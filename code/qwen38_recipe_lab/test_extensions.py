"""Public vision oracle and MTP architecture tests; no claim of official MTP training."""
import os,unittest
import torch
from torch import nn
from vision import VisionConfig,VisionModel,text_mrope
from types import SimpleNamespace
from mtp import MultiTokenPredictor
from reference import load_reference
from test_recipe import tiny

REF=load_reference()
class ExtensionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(5); self.device=os.environ.get('LAB_TEST_DEVICE','cpu')
    def test_text_multiaxis_rope_matches_upstream_and_text_special_case(self):
        config=SimpleNamespace(max_position_embeddings=262144,head_dim=256,
            rope_parameters=dict(rope_type='default',rope_theta=1e7,partial_rotary_factor=.25,mrope_section=[11,11,10]))
        ref=REF['Qwen4ExpTextRotaryEmbedding'](config)
        positions=torch.randint(0,1024,(3,2,17))
        expected=ref(torch.zeros(2,17,64),positions)
        actual=text_mrope(positions.to(self.device))
        for a,b in zip(actual,expected,strict=True):
            torch.testing.assert_close(a.cpu(),b,atol=3e-5,rtol=3e-5)
        from model import rope
        text=torch.arange(17)[None,None].expand(3,1,-1).to(self.device)
        cos,sin=text_mrope(text)
        x=torch.randn(1,17,2,256,device=self.device)
        expected=REF['apply_rotary_pos_emb'](x,cos=cos,sin=sin,unsqueeze_dim=2)
        torch.testing.assert_close(rope(x,torch.arange(17,device=self.device),64,1e7),expected,atol=3e-5,rtol=3e-5)
    def test_full_vision_geometry_blocks_merger_and_gradients(self):
        c=VisionConfig(hidden_size=16,intermediate_size=32,num_heads=2,depth=2,patch_size=2,out_hidden_size=32)
        model=VisionModel(c).to(self.device)
        grid=torch.tensor([[1,4,4],[2,6,4]])
        total=sum(t*h*w for t,h,w in grid.tolist())
        x=torch.randn(total,3*2*2*2,requires_grad=True)
        actual_x=x.detach().to(self.device).requires_grad_()
        positions=REF['get_vision_position_ids'](grid,2)
        indices,weights=REF['get_vision_interpolation_indices_and_weights'](grid,4,mode='bilinear',align_corners=True,spatial_merge_size=2)
        cu=REF['get_vision_cu_seqlens'](grid)
        ours_pos,ours_emb,ours_cu=model.geometry(grid.to(self.device))
        self.assertTrue(torch.equal(ours_pos.cpu(),positions)); self.assertTrue(torch.equal(ours_cu.cpu(),cu))
        expected_emb=(model.pos_embed.weight.detach().cpu()[indices]*weights[:,:,None]).sum(1)
        torch.testing.assert_close(ours_emb.cpu(),expected_emb,atol=2e-6,rtol=1e-5)
        patch=REF['Qwen4ExpVisionPatchEmbed'](c); patch.load_state_dict({k:v.cpu() for k,v in model.patch_embed.state_dict().items()})
        blocks=[]
        for block in model.blocks:
            ref=REF['Qwen4ExpVisionBlock'](c); ref.load_state_dict({k:v.cpu() for k,v in block.state_dict().items()}); blocks.append(ref)
        merger=REF['Qwen4ExpVisionPatchMerger'](c); merger.load_state_dict({k:v.cpu() for k,v in model.merger.state_dict().items()})
        expected=patch(x)+expected_emb
        rotary=REF['Qwen4ExpVisionRotaryEmbedding'](c)(expected,positions)
        for block in blocks: expected=block(expected,cu,rotary)
        merged=merger(expected)
        a,b=model(actual_x,grid.to(self.device))
        torch.testing.assert_close(a.cpu(),expected,atol=2e-5,rtol=2e-4)
        torch.testing.assert_close(b.cpu(),merged,atol=2e-5,rtol=2e-4)
        b.square().sum().backward(); merged.square().sum().backward()
        torch.testing.assert_close(actual_x.grad.cpu(),x.grad,atol=5e-5,rtol=5e-4)
    def test_mtp_shared_branch_fusion_and_selection_reuse(self):
        c=tiny(); m=MultiTokenPredictor(c).to(self.device)
        embedding=torch.randn(1,19,c.width,device=self.device)
        hidden=torch.randn(1,19,4*c.width,device=self.device)
        perm=torch.tensor([3,1,0,2],device=self.device)
        fused=m.fusion(embedding,hidden)
        permuted=hidden.unflatten(-1,(4,c.width))[:,:,perm].flatten(-2)
        torch.testing.assert_close(m.fusion(embedding,permuted),fused.unflatten(-1,(4,c.width))[:,:,perm].flatten(-2),atol=1e-6,rtol=1e-5)
        calls=[]
        hook=m.layer.mix.indexer.register_forward_hook(lambda *args:calls.append(1))
        single,multi,tokens,kl,_=m(embedding,hidden)
        reused=m(embedding,hidden,reuse_tokens=tokens)
        hook.remove()
        self.assertEqual(len(calls),1)
        torch.testing.assert_close(single,reused[0],atol=3e-6,rtol=3e-5)
        torch.testing.assert_close(multi,reused[1],atol=3e-6,rtol=3e-5)
        self.assertEqual(tuple(multi.shape),(1,19,4*c.width))
        (single.square().mean()+kl).backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None))
        with self.assertRaises(ValueError): m(embedding,hidden,reuse_tokens=torch.full_like(tokens,18))
    def test_mtp_hidden_norm_is_global_not_per_branch(self):
        c=tiny(); m=MultiTokenPredictor(c).to(self.device)
        with torch.no_grad(): m.fc_hidden.weight.copy_(torch.eye(c.width,device=self.device))
        hidden=torch.randn(1,3,4,c.width,device=self.device)
        hidden[:,:,0]*=10
        flat=hidden.flatten(-2)
        expected=flat*torch.rsqrt(flat.square().mean(-1,keepdim=True)+1e-6)
        actual=m.fusion(torch.zeros(1,3,c.width,device=self.device),flat)
        torch.testing.assert_close(actual,expected,atol=2e-6,rtol=2e-5)

if __name__=='__main__': unittest.main()
