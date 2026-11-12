import io,json,math,os,unittest
from types import SimpleNamespace
import torch
from torch import nn
from model import Config,Model,GDN,GatedResidual,Indexer,Attention,NGramEmbedding,PLE,RMSNorm,rope,delta_scan
from optim import RecipeOptimizer,ownership,polar,head_pieces,piece,COEFFICIENTS
from reference import load_reference

torch.set_num_threads(1)
REF=load_reference()

def tiny(**kwargs):
    cfg=dict(vocab_size=97,eos=96,width=32,heads=2,head_dim=32,rotary_dim=8,
             gdn_k_dim=16,gdn_v_dim=16,hidden=16,experts=4,index_dim=32,
             ngram_base=31,budget=8,rank=4)
    cfg.update(kwargs); return Config(**cfg)

def upstream(c):
    return SimpleNamespace(hidden_size=c.width,linear_num_value_heads=c.gdn_v_heads,
        linear_num_key_heads=c.gdn_k_heads,linear_key_head_dim=c.gdn_k_dim,linear_value_head_dim=c.gdn_v_dim,
        linear_conv_kernel_dim=4,hidden_act='silu',rms_norm_eps=1e-6,output_gate_type='sigmoid',
        layer_types=['linear_attention']*3+['indexed_attention'],hc_count=4,hc_lowrank=c.rank,
        ngram_size=3,heads_per_ngram=c.ngram_heads,vocab_size=c.vocab_size,ngram_vocab_size_base=c.ngram_base,
        seed=c.hash_seed,eos_token_id=c.eos,make_ngram_vocab_size_divisible_by=128,
        ple_embed_dim=c.width,ple_conv_kernel_size=4,indexer_n_heads=c.index_heads,indexer_kv_heads=1,
        indexer_head_dim=c.index_dim,indexer_budget=c.budget,indexer_compress_ratio=c.block)

class RecipeTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(123); self.device=os.environ.get('LAB_TEST_DEVICE','cpu')
        self.c=tiny()
    def test_gdn_upstream_chunk_output_gradients_and_cache(self):
        c=self.c; model=GDN(c).to(self.device)
        ref=REF['Qwen4ExpTextGatedDeltaNet'](upstream(c),0)
        mapping={k.replace('conv.weight','conv1d.weight').replace('norm_weight','norm.weight'):v.cpu() for k,v in model.state_dict().items()}
        ref.load_state_dict(mapping)
        x=torch.randn(1,19,c.width,requires_grad=True)
        y=x.detach().to(self.device).requires_grad_()
        expected=ref(x); actual=model(y)
        torch.testing.assert_close(actual.cpu(),expected,atol=3e-5,rtol=3e-4)
        expected.square().sum().backward(); actual.square().sum().backward()
        torch.testing.assert_close(y.grad.cpu(),x.grad,atol=8e-5,rtol=8e-4)
        cache={}
        with torch.no_grad():
            chunked=torch.cat((model(y[:,:7],cache),model(y[:,7:13],cache),model(y[:,13:],cache)),1)
            torch.testing.assert_close(chunked,actual,atol=3e-5,rtol=3e-4)
    def test_gr_and_final_read_match_upstream(self):
        c=self.c
        for write in (True,False):
            own=GatedResidual(c,write).to(self.device)
            ref=REF['Qwen4ExpTextGatedResidual'](upstream(c),use_combine=write)
            names={'norm.weight':'hc_norm.weight','down.weight':'input_mix_weight_down.weight',
                   'up.weight':'input_mix_weight_up.weight','inject.weight':'block_inject_weight.weight'}
            ref.load_state_dict({names[k]:v.cpu() for k,v in own.state_dict().items()})
            x=torch.randn(2,7,4*c.width)
            a,b=own(x.to(self.device)),ref(x)
            if write:
                torch.testing.assert_close(a[0].cpu(),b[0]); torch.testing.assert_close(a[1].cpu(),b[2])
            else: torch.testing.assert_close(a.cpu(),b)
    def test_ngram_hash_eos_and_ple_match_upstream(self):
        c=self.c; own=PLE(c).to(self.device)
        ref=REF['Qwen4ExpTextPLELayer'](upstream(c),1,0)
        mapping={}
        for k,v in own.state_dict().items():
            key=k.replace('embedding.ngram_embedding','ple_embedding.ngram_embedding')
            key=key.replace('key.','key_proj.').replace('value.','value_proj.').replace('qnorm.','norm_query.').replace('knorm.','norm_key.').replace('cnorm.','norm_conv.')
            if k.startswith('conv.'): key=k.replace('conv.','conv1d.',1)
            mapping[key]=v.cpu()
        missing=ref.load_state_dict(mapping,strict=False)
        self.assertEqual(set(missing.missing_keys),{'ple_embedding.layer_multipliers','ple_embedding.ngram_heads_vocab_sizes','ple_embedding.ngram_heads_offsets'})
        self.assertFalse(missing.unexpected_keys)
        ids=torch.tensor([[5,6,c.eos,7,8,c.eos,9,10,11,12,13]])
        torch.testing.assert_close(own.embedding(ids.to(self.device)).cpu(),ref.ple_embedding(ids,None))
        x=torch.randn(1,ids.shape[1],4*c.width)
        result=own(x.to(self.device),ids.to(self.device))
        torch.testing.assert_close(result.cpu(),ref(x,ids,None),atol=1e-5,rtol=1e-4)
        cache={}
        parts=[own(x[:,a:b].to(self.device),ids[:,a:b].to(self.device),cache) for a,b in ((0,4),(4,7),(7,11))]
        torch.testing.assert_close(torch.cat(parts,1),result,atol=1e-5,rtol=1e-4)
    def test_rope_and_index_selection_match_upstream(self):
        c=self.c; own=Indexer(c).to(self.device)
        ref=REF['Qwen4ExpTextQSAIndexer'](upstream(c),3)
        ref.load_state_dict({'index_qk_proj.weight':own.qk.weight.cpu(),'q_layernorm.weight':own.qnorm.weight.cpu(),'k_layernorm.weight':own.knorm.weight.cpu()})
        x=torch.randn(2,19,c.width)
        scores,valid,selected,tokens=own(x.to(self.device))
        freq=c.rope_theta**(-torch.arange(0,c.rotary_dim,2)/c.rotary_dim)
        angle=torch.arange(19)[:,None]*freq
        cos=torch.cat((angle.cos(),angle.cos()),-1)[None].expand(2,-1,-1)
        sin=torch.cat((angle.sin(),angle.sin()),-1)[None].expand(2,-1,-1)
        mask=torch.ones(2,1,19,19,dtype=torch.bool).tril()
        expected=ref(x,(cos,sin),mask,None)
        actual=torch.zeros(2,19,20,device=self.device,dtype=torch.bool).scatter(-1,torch.where(tokens>=0,tokens,19),True)[...,:19]
        self.assertTrue(torch.equal(actual.cpu(),expected[:,0]))
        self.assertFalse(actual.triu(1).any())
        self.assertTrue((actual.sum(-1)<torch.arange(1,20,device=self.device)).any())
        probe=torch.randn(1,19,2,32)
        ours=rope(probe,torch.arange(19),c.rotary_dim,c.rope_theta)
        oracle=REF['apply_rotary_pos_emb'](probe,cos=cos[:1],sin=sin[:1],unsqueeze_dim=2)
        torch.testing.assert_close(ours,oracle)
    def test_sparse_gather_matches_dense_mask_outputs_and_gradients(self):
        model=Attention(tiny(budget=32)).to(self.device)
        x=torch.randn(2,39,self.c.width,device=self.device,requires_grad=True)
        out,kl=model(x,'sparse'); loss=out.square().sum()+kl
        loss.backward(); grad=x.grad.clone()
        own=[p.grad.clone() if p.grad is not None else None for p in model.parameters()]
        model.zero_grad(); x.grad=None
        oracle,klo=model(x,'sparse',dense_oracle=True)
        torch.testing.assert_close(out,oracle,atol=3e-6,rtol=3e-5)
        torch.testing.assert_close(kl,klo,atol=3e-6,rtol=3e-5)
        (oracle.square().sum()+klo).backward()
        torch.testing.assert_close(grad,x.grad,atol=2e-5,rtol=2e-4)
        for p,g in zip(model.parameters(),own):
            if g is not None: torch.testing.assert_close(p.grad,g,atol=3e-5,rtol=3e-4)
    def test_zero_score_ties_do_not_read_future(self):
        indexer=Indexer(self.c).to(self.device)
        nn.init.zeros_(indexer.qk.weight)
        x=torch.randn(1,39,self.c.width,device=self.device)
        long=indexer(x)[-1]
        short=indexer(x[:,:23])[-1]
        self.assertTrue(torch.equal(long[:,:23],short))
    def test_full_model_causality_cache_and_stage_freezing(self):
        m=Model(self.c).to(self.device)
        ids=torch.randint(0,90,(1,19),device=self.device)
        with torch.no_grad():
            for stage in ('pretrain','distill','sparse'):
                a=m(ids,stage)[0]
                b=ids.clone(); b[:,9:]=(b[:,9:]+1)%90
                torch.testing.assert_close(a[:,:9],m(b,stage)[0][:,:9],atol=2e-5,rtol=2e-4)
                torch.testing.assert_close(a[:,:9],m(ids[:,:9],stage)[0],atol=2e-5,rtol=2e-4)
            cache={}
            cached=torch.cat([m(ids[:,i:i+1],cache=cache)[0] for i in range(19)],1)
            torch.testing.assert_close(cached,m(ids)[0],atol=3e-5,rtol=3e-4)
        m.stage('distill'); before={k:v.detach().clone() for k,v in m.named_parameters()}
        opt=RecipeOptimizer(ownership(m),lr=1e-4)
        _,kl,_=m(ids,'distill'); kl.backward(); opt.step()
        changed=[k for k,p in m.named_parameters() if not torch.equal(before[k],p)]
        self.assertTrue(changed and all('.indexer.' in k for k in changed))
        m.stage('sparse'); m.zero_grad(); m(ids,'sparse')[1].backward()
        self.assertTrue(all('.indexer.' in k for k,p in m.named_parameters() if p.grad is not None))
    def test_polar_matrix_polynomial_svd_and_zero(self):
        for shape in ((7,11),(11,7),(5,5)):
            x=torch.randn(*shape,dtype=torch.float64)
            u,s,vh=torch.linalg.svd(x,full_matrices=False)
            z=s/x.norm().clamp_min(1e-14)
            for a,b,c in COEFFICIENTS: z=a*z+b*z**3+c*z**5
            torch.testing.assert_close(polar(x,dtype=torch.float64),(u*z)@vh,atol=1e-10,rtol=1e-9)
            self.assertLess((polar(x,dtype=torch.float64)-u@vh).norm().item()/(u@vh).norm().item(),0.05)
            actual=polar(x.float().to(self.device),dtype=torch.float32).cpu()
            torch.testing.assert_close(actual,polar(x.float(),dtype=torch.float32),atol=3e-4,rtol=3e-3)
        torch.testing.assert_close(polar(torch.zeros(5,7,device=self.device)),torch.zeros(5,7,device=self.device,dtype=torch.bfloat16))
        x=torch.diag(torch.tensor([1.,1e-2,1e-4,1e-8,0.],dtype=torch.float64))
        z=x.diag()/x.norm().clamp_min(1e-14)
        for a,b,c in COEFFICIENTS: z=a*z+b*z**3+c*z**5
        torch.testing.assert_close(polar(x,dtype=torch.float64),z.diag(),atol=1e-10,rtol=1e-9)
        self.assertTrue(torch.isfinite(polar(x.float().to(self.device))).all())
    def test_optimizer_multistep_cpu_mps_trajectory(self):
        p=nn.Parameter(torch.randn(7,11)); q=nn.Parameter(p.detach().clone().to(self.device))
        a=RecipeOptimizer([('weight',p,[piece('muon')])],lr=1e-3,ns_dtype='float32')
        b=RecipeOptimizer([('weight',q,[piece('muon')])],lr=1e-3,ns_dtype='float32')
        for _ in range(8):
            grad=torch.randn_like(p); p.grad=grad; q.grad=grad.to(self.device)
            a.step(); b.step()
        torch.testing.assert_close(p,q.cpu(),atol=1e-5,rtol=1e-4)
    def test_ungated_ablation_keeps_branch_norm_and_removes_gates(self):
        c=tiny(gr_mode='ungated'); gr=GatedResidual(c).to(self.device)
        x=torch.randn(1,7,4*c.width,device=self.device,requires_grad=True)
        y,write=gr(x)
        expected=gr.norm(x).unflatten(-1,(4,c.width)).mean(-2)
        torch.testing.assert_close(y,expected); torch.testing.assert_close(write,torch.ones_like(write))
        y.square().sum().backward()
        self.assertIsNone(gr.down.weight.grad); self.assertIsNone(gr.up.weight.grad); self.assertIsNone(gr.inject.weight.grad)
    def test_fused_ownership_updates_adam_and_resume(self):
        p=nn.Parameter(torch.randn(16,8,device=self.device))
        p._recipe_pieces=head_pieces(2,4,gated=True)
        module=nn.Module(); module.register_parameter('fused',p)
        split=[nn.Parameter(x.detach().clone()) for x in p.split(4)]
        plans=[(str(i),v,[piece('muon' if i%2==0 else 'adamw')]) for i,v in enumerate(split)]
        a=RecipeOptimizer(ownership(module),lr=1e-3); b=RecipeOptimizer(plans,lr=1e-3)
        for step in range(4):
            grad=torch.randn_like(p); p.grad=grad
            for v,g in zip(split,grad.split(4)): v.grad=g.clone()
            a.step(); b.step()
            torch.testing.assert_close(p,torch.cat(split),atol=1e-6,rtol=1e-5)
        memory=io.BytesIO(); torch.save(a.state_dict(),memory); memory.seek(0)
        clone=RecipeOptimizer(ownership(module),lr=1e-3)
        clone.load_state_dict(torch.load(memory,weights_only=True,map_location=self.device))
        x=nn.Parameter(torch.randn(5,7,device=self.device)); y=nn.Parameter(x.detach().clone())
        own=RecipeOptimizer([('adam',x,[piece('adamw')])],lr=1e-3,weight_decay=.01)
        ref=torch.optim.AdamW([y],lr=1e-3,betas=(0.9,0.95),eps=1e-8,weight_decay=0.01,foreach=False)
        for _ in range(3):
            x.grad=torch.randn_like(x); y.grad=x.grad.clone(); own.step(); ref.step()
        torch.testing.assert_close(x,y,atol=2e-7,rtol=2e-6)

if __name__=='__main__': unittest.main()
