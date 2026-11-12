"""Public packing/placeholder/attention oracle and connected visual gradients."""
import os,unittest
from types import SimpleNamespace
import torch
from torch import nn
from model import Model,Attention
from multimodal import multimodal_positions,inject_features,attention,MultimodalModel
from vision import VisionModel,VisionConfig,text_mrope
from reference import load_reference
from test_recipe import tiny,upstream

REF=load_reference()


class MultimodalTests(unittest.TestCase):
    def setUp(self):torch.manual_seed(111);self.device=os.environ.get('LAB_TEST_DEVICE','cpu')
    def test_public_image_video_padding_positions_and_placeholder_replacement(self):
        row=[1]+[94]*4+[2]+[95]*4+[3]+[95]*4+[4]
        ids=torch.tensor([row,[1]+[94]*4+[2]+[0]*10])
        types=torch.where(ids==94,1,torch.where(ids==95,2,0));mask=torch.ones_like(ids);mask[1,6:]=0
        images=torch.tensor([[1,4,4],[1,4,4]]);videos=torch.tensor([[2,4,4]])
        ref=REF['Qwen4ExpModel']();ref.config=SimpleNamespace(vision_config=SimpleNamespace(spatial_merge_size=2),image_token_id=94,video_token_id=95)
        expected=ref.get_rope_index(ids,types,images,videos,mask)
        actual=multimodal_positions(ids.to(self.device),types.to(self.device),images,videos,mask.to(self.device))
        for a,b in zip(actual,expected,strict=True):self.assertTrue(torch.equal(a.cpu(),b))
        embedding=nn.Embedding(97,32).to(self.device)
        image=torch.randn(8,32,device=self.device,requires_grad=True);video=torch.randn(8,32,device=self.device,requires_grad=True)
        base=embedding(ids.to(self.device));a,b=ref.get_placeholder_mask(ids.to(self.device),base,image,video)
        expected=base.masked_scatter(a,image).masked_scatter(b,video)
        actual=inject_features(ids.to(self.device),embedding,image,video,94,95)
        torch.testing.assert_close(actual,expected)
        actual.sum().backward();torch.testing.assert_close(image.grad,torch.ones_like(image));torch.testing.assert_close(video.grad,torch.ones_like(video))
        self.assertTrue(torch.equal(embedding.weight.grad[94:96],torch.zeros_like(embedding.weight.grad[94:96])))
        with self.assertRaises(ValueError):inject_features(ids.to(self.device),embedding,image[:7],video,94,95)

    def test_three_axis_attention_matches_public_module_and_gradients(self):
        c=tiny(heads=12,head_dim=256,index_dim=128,rotary_dim=64,budget=128)
        ours=Attention(c).to(self.device);cfg=upstream(c)
        cfg.head_dim=c.head_dim;cfg.num_attention_heads=c.heads;cfg.num_key_value_heads=c.kv_heads
        cfg.attention_dropout=0.;cfg.attention_bias=False;cfg._attn_implementation='eager'
        ref=REF['Qwen4ExpTextAttention'](cfg,3)
        mapping={'q_proj.weight':'q_gate.weight','k_proj.weight':'k.weight','v_proj.weight':'v.weight','o_proj.weight':'out.weight',
            'q_norm.weight':'qnorm.weight','k_norm.weight':'knorm.weight','indexer.index_qk_proj.weight':'indexer.qk.weight',
            'indexer.q_layernorm.weight':'indexer.qnorm.weight','indexer.k_layernorm.weight':'indexer.knorm.weight'}
        ref.load_state_dict({key:ours.state_dict()[value].cpu() for key,value in mapping.items()})
        x=torch.randn(1,17,c.width,requires_grad=True);actual_x=x.detach().to(self.device).requires_grad_()
        positions=torch.randint(0,32,(3,1,17));cos,sin=text_mrope(positions)
        mask=torch.triu(torch.full((17,17),torch.finfo(torch.float32).min),1)[None,None]
        expected=ref(x,(cos,sin),mask)[0]
        actual=attention(ours,actual_x,positions.to(self.device),'sparse')
        torch.testing.assert_close(actual.cpu(),expected,atol=3e-5,rtol=3e-4)
        actual.square().sum().backward();expected.square().sum().backward()
        torch.testing.assert_close(actual_x.grad.cpu(),x.grad,atol=5e-5,rtol=5e-4)

    def test_connected_image_forward_gradients_and_text_identity(self):
        c=tiny();base=Model(c).to(self.device).eval()
        visual=VisionModel(VisionConfig(hidden_size=16,intermediate_size=32,num_heads=2,depth=1,patch_size=2,out_hidden_size=c.width)).to(self.device)
        model=MultimodalModel(base,visual,94,95)
        ids=torch.randint(0,94,(1,23),device=self.device);types=torch.zeros_like(ids)
        with torch.no_grad():
            for stage in ('pretrain','sparse'):
                torch.testing.assert_close(model(ids,types,stage=stage)[0],base(ids,stage)[0],atol=3e-5,rtol=3e-4)
        ids=torch.tensor([[1]+[94]*4+[2,3]],device=self.device);types=torch.where(ids==94,1,0)
        patches=torch.randn(16,24,device=self.device,requires_grad=True);grid=torch.tensor([[1,4,4]],device=self.device)
        logits,_,_=model(ids,types,image_patches=patches,image_grid=grid)
        logits.square().mean().backward();self.assertTrue(torch.isfinite(patches.grad).all());self.assertGreater(float(patches.grad.abs().sum()),0)

    def test_padded_batch_valid_logits_positions_and_gradients_match_independent_sequences(self):
        c=tiny();base=Model(c).to(self.device).eval()
        visual=VisionModel(VisionConfig(hidden_size=16,intermediate_size=32,num_heads=2,depth=1,patch_size=2,out_hidden_size=c.width)).to(self.device)
        model=MultimodalModel(base,visual,94,95)
        # One left-padded image sequence and one right-padded text sequence.
        ids=torch.tensor([[0,0,1,94,94,94,94,2,3],[4,5,6,7,8,0,0,0,0]],device=self.device)
        mask=torch.tensor([[0,0,1,1,1,1,1,1,1],[1,1,1,1,1,0,0,0,0]],device=self.device)
        types=torch.where(ids==94,1,0);grid=torch.tensor([[1,4,4]],device=self.device)
        patches=torch.randn(16,24,device=self.device,requires_grad=True)
        for stage in ('pretrain','sparse'):
            base.zero_grad();visual.zero_grad();patches.grad=None
            actual,positions,deltas=model(ids,types,image_patches=patches,image_grid=grid,stage=stage,mask=mask)
            self.assertTrue(torch.equal(actual[~mask.bool()],torch.zeros_like(actual[~mask.bool()])))
            actual.square().sum().backward()
            actual_image_grad=patches.grad.detach().clone();actual_embedding_grad=base.embedding.weight.grad.detach().clone()
            base.zero_grad();visual.zero_grad();patches.grad=None;loss=0
            for b in range(2):
                valid=mask[b].bool();kwargs=dict(image_patches=patches,image_grid=grid) if b==0 else {}
                expected,expected_positions,expected_delta=model(ids[b:b+1,valid],types[b:b+1,valid],stage=stage,**kwargs)
                torch.testing.assert_close(actual[b,valid],expected[0],atol=3e-5,rtol=3e-4)
                self.assertTrue(torch.equal(positions[:,b:b+1,valid],expected_positions))
                self.assertTrue(torch.equal(deltas[b:b+1],expected_delta));loss=loss+expected.square().sum()
            loss.backward()
            torch.testing.assert_close(actual_image_grad,patches.grad,atol=5e-5,rtol=5e-4)
            torch.testing.assert_close(actual_embedding_grad,base.embedding.weight.grad,atol=5e-5,rtol=5e-4)
        with self.assertRaises(ValueError):model(ids,types,image_patches=patches,image_grid=grid,mask=torch.zeros_like(mask))


if __name__=='__main__':unittest.main()
