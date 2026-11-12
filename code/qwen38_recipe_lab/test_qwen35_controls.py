"""Pending component acceptance: execute serially on CPU/MPS with the global lock."""
import ast,copy,hashlib,json,os
from pathlib import Path
from types import SimpleNamespace
import unittest
import torch
from torch.nn import functional as F
from model import rope
from qwen35_controls import ControlConfig,GDN35,PlainAttention,PlainLayer,Qwen35Control,causal_window_mask,gated_norm35


def tiny(**kwargs):
    fields=dict(vocab_size=32,width=8,heads=2,kv_heads=1,head_dim=8,rotary_dim=2,gdn_k_dim=8,gdn_v_dim=8,hidden=4,experts=4,top_k=2,query_chunk=7,scan_chunk=3,activation_checkpoint=False)
    fields.update(kwargs);return ControlConfig(**fields)


def published_tree():
    rows=json.loads(Path('qwen35_reference_sources.lock.json').read_text());record=next(r for r in rows if r['path'].endswith('modeling_qwen3_5_moe.py'));raw=Path(record['file']).read_bytes();assert hashlib.sha256(raw).hexdigest()==record['sha256'];return ast.parse(raw)


class Qwen35ControlsTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1);torch.manual_seed(123);self.device=torch.device(os.environ.get('LAB_TEST_DEVICE','cpu'))
        if self.device.type=='mps':self.assertTrue(torch.backends.mps.is_available())
    def test_gate_matches_published_silu_norm(self):
        node=next(n for n in published_tree().body if isinstance(n,ast.ClassDef) and n.name=='Qwen3_5MoeRMSNormGated');node.decorator_list=[]  # Reference the Torch fallback, not hub kernel dispatch.
        namespace=dict(torch=torch,nn=torch.nn,ACT2FN={'silu':F.silu})
        exec(compile(ast.Module(body=[node],type_ignores=[]),'pinned_qwen35_norm','exec'),namespace)
        reference=namespace['Qwen3_5MoeRMSNormGated'](8).to(self.device);hidden=torch.randn(19,8,device=self.device);gate=torch.linspace(-3,3,152,device=self.device).reshape(19,8)
        actual=gated_norm35(hidden,gate,reference.weight);torch.testing.assert_close(actual,reference(hidden,gate),rtol=1e-6,atol=1e-6)
        self.assertTrue(torch.any(actual[gate<0]!=0));self.assertEqual(actual.device.type,self.device.type)
    def test_ordinary_layer_composition_matches_published_forward(self):
        layer_node=next(n for n in published_tree().body if isinstance(n,ast.ClassDef) and n.name=='Qwen3_5MoeDecoderLayer');function=copy.deepcopy(next(n for n in layer_node.body if isinstance(n,ast.FunctionDef) and n.name=='forward'));function.returns=None;function.decorator_list=[]
        for node in ast.walk(function):
            if isinstance(node,ast.arg):node.annotation=None
        namespace={};exec(compile(ast.fix_missing_locations(ast.Module(body=[function],type_ignores=[])),'pinned_qwen35_layer','exec'),namespace)
        for kind in ('full','swa','gdn'):
            with self.subTest(kind=kind):
                layer=PlainLayer(tiny(),kind).to(self.device);x=torch.randn(2,11,8,device=self.device)
                reference=SimpleNamespace(block_type='linear_attention' if kind=='gdn' else 'full_attention',input_layernorm=layer.input_norm,post_attention_layernorm=layer.post_norm,mlp=layer.moe,linear_attn=lambda hidden_states,**kw:layer.mixer(hidden_states),self_attn=lambda hidden_states,**kw:(layer.mixer(hidden_states),None))
                expected=namespace['forward'](reference,x,None);actual=layer(x)[0];torch.testing.assert_close(actual,expected,rtol=2e-4,atol=2e-5)
    def test_window128_actually_excludes_history_and_future(self):
        pos=torch.arange(259,device=self.device);mask=causal_window_mask(pos,pos,128).cpu();self.assertEqual(int(mask[-1].sum()),128);self.assertFalse(bool(mask[128,0]));self.assertFalse(bool(mask[10,11]));self.assertTrue(bool(mask[128,1]))
        swa=PlainAttention(tiny(),128).to(self.device).eval();full=PlainAttention(tiny()).to(self.device).eval();full.load_state_dict(swa.state_dict());x=torch.randn(1,259,8,device=self.device);changed=x.clone();changed[:,:128]+=4
        with torch.no_grad():
            before=swa(x);after=swa(changed);torch.testing.assert_close(before[:,-1],after[:,-1],rtol=1e-5,atol=1e-6)
            self.assertFalse(torch.equal(full(x)[:,-1],full(changed)[:,-1]));torch.testing.assert_close(before[:,:128],full(x)[:,:128],rtol=2e-4,atol=2e-5)
    def test_checkpoint_preserves_recurrent_state_gradients(self):
        reference=GDN35(tiny(scan_chunk=17)).to(self.device);actual=GDN35(tiny(scan_chunk=3,activation_checkpoint=True)).to(self.device);actual.load_state_dict(reference.state_dict());x=torch.randn(1,17,8,device=self.device,requires_grad=True);z=x.detach().clone().requires_grad_(True)
        y=reference(x);v=actual(z);torch.testing.assert_close(v,y,rtol=2e-4,atol=2e-5)
        expected=torch.autograd.grad(y.square().mean(),[x,*reference.parameters()]);observed=torch.autograd.grad(v.square().mean(),[z,*actual.parameters()])
        for a,b in zip(expected,observed,strict=True):self.assertTrue(bool(torch.isfinite(b).all()));torch.testing.assert_close(b,a,rtol=5e-4,atol=5e-5)
    def test_gqa8_matches_independent_dense_attention(self):
        for heads,kv_heads in ((8,1),(16,2)):
            with self.subTest(heads=heads,kv_heads=kv_heads):
                c=tiny(heads=heads,kv_heads=kv_heads,head_dim=16,rotary_dim=4,query_chunk=3);layer=PlainAttention(c).to(self.device);x=torch.randn(2,19,c.width,device=self.device);b,t,_=x.shape;positions=torch.arange(t,device=self.device)
                q,gate=layer.q_gate(x).reshape(b,t,c.heads,2*c.head_dim).chunk(2,-1)
                q=rope(layer.qnorm(q),positions,c.rotary_dim,c.rope_theta).transpose(1,2)
                k=rope(layer.knorm(layer.k(x).reshape(b,t,c.kv_heads,c.head_dim)),positions,c.rotary_dim,c.rope_theta).repeat_interleave(8,2).transpose(1,2)
                v=layer.v(x).reshape(b,t,c.kv_heads,c.head_dim).repeat_interleave(8,2).transpose(1,2)
                expected=F.scaled_dot_product_attention(q,k,v,is_causal=True).transpose(1,2)*gate.sigmoid();expected=layer.out(expected.flatten(-2))
                torch.testing.assert_close(layer(x),expected,rtol=2e-4,atol=2e-5)
    def test_common_initialization_and_arm_layout(self):
        models={arm:Qwen35Control(tiny(),arm) for arm in ('full','swa','gdn')};weights={arm:dict(m.named_parameters()) for arm,m in models.items()}
        self.assertEqual([m.kind for m in models['gdn'].layers],['gdn','gdn','gdn','full']);self.assertEqual([m.kind for m in models['swa'].layers],['swa','swa','swa','full'])
        for name,p in weights['full'].items():
            if '.mixer.' not in name or name.startswith('layers.3.'):
                for arm in ('swa','gdn'):self.assertTrue(torch.equal(p,weights[arm][name]),name)
            self.assertTrue(torch.equal(p,weights['swa'][name]),name)
        self.assertFalse(any('indexer' in name or 'ngram' in name or 'mix_residual' in name for name in weights['gdn']))
    def test_whole_model_causality_all_arms(self):
        ids=torch.randint(0,32,(1,17),device=self.device);changed=ids.clone();changed[:,11:]=(changed[:,11:]+7)%32
        for arm in ('full','swa','gdn'):
            with self.subTest(arm=arm):
                model=Qwen35Control(tiny(),arm).to(self.device).eval()
                with torch.no_grad():a=model(ids)[0];b=model(changed)[0]
                torch.testing.assert_close(a[:,:11],b[:,:11],rtol=2e-4,atol=2e-5);self.assertEqual(a.device.type,self.device.type)


if __name__=='__main__':unittest.main()
