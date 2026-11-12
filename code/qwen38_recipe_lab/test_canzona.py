"""Direct mathematical comparison to the paper author's implementation, not a copy oracle."""
import ast,json,hashlib,math,os,types,unittest
from pathlib import Path
import torch
from torch import nn
from optim import RecipeOptimizer,polar,piece


def author_reference():
    file=Path('.cache/references/canzona_muon.py')
    spec=next(s for s in json.loads(Path('sources.lock.json').read_text()) if s['name']==file.name)
    assert hashlib.sha256(file.read_bytes()).hexdigest()==spec['sha256']
    tree=ast.parse(file.read_text()); selected=[]
    for node in tree.body:
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='_COEFFICIENT_SETS' for t in node.targets): selected.append(node)
        if isinstance(node,ast.FunctionDef) and node.name=='zeropower_via_newtonschulz5': selected.append(node)
        if isinstance(node,ast.ClassDef) and node.name=='Muon':
            selected += [m for m in node.body if isinstance(m,ast.FunctionDef) and m.name in ('get_muon_scale_factor','_inner_single_param_step','_single_param_update')]
    module=ast.Module(body=selected,type_ignores=[]); ns=dict(torch=torch,math=math)
    exec(compile(ast.fix_missing_locations(module),str(file),'exec'),ns)
    return ns

class CanzonaTests(unittest.TestCase):
    def test_ns_matches_author_bfloat16(self):
        device=os.environ.get('LAB_TEST_DEVICE','cpu'); torch.manual_seed(17)
        ref=author_reference()['zeropower_via_newtonschulz5']
        for shape in ((7,11),(11,7),(16,16),(128,64)):
            x=torch.randn(*shape,device=device)
            actual=polar(x)
            expected=ref(x,steps=8,ns_coefficient_type='qwen_express',ns_norm_eps=1e-14)
            torch.testing.assert_close(actual,expected,atol=0,rtol=0)
    def test_parameter_update_and_momentum_match_author(self):
        device=os.environ.get('LAB_TEST_DEVICE','cpu'); torch.manual_seed(19)
        p=nn.Parameter(torch.randn(7,11,device=device)); q=nn.Parameter(p.detach().clone())
        opt=RecipeOptimizer([('weight',p,[piece('muon')])],lr=1e-3)
        ns=author_reference(); fake=types.SimpleNamespace(state={q:{}})
        fake.get_muon_scale_factor=types.MethodType(ns['get_muon_scale_factor'],fake)
        group=dict(momentum=.95,nesterov=True,ns_steps=8,ns_coefficient_type='qwen_express',ns_norm_eps=1e-14,weight_decay=.1,lr=.001)
        for _ in range(8):
            grad=torch.randn_like(p); p.grad=grad.clone()
            expected=ns['_inner_single_param_step'](fake,'',q,grad.clone(),group).to(grad.dtype)
            with torch.no_grad(): ns['_single_param_update'](fake,q,expected,group)
            opt.step()
            torch.testing.assert_close(p,q,atol=0,rtol=0)
            torch.testing.assert_close(opt.state[p]['pieces'][0]['momentum'],fake.state[q]['momentum_buffer'],atol=0,rtol=0)

if __name__=='__main__': unittest.main()
