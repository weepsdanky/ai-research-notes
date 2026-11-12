"""Check all three rematerialized arms against the original numerical model."""
import os,unittest
import torch
from torch.nn import functional as F
from qwen35_controls import ControlConfig,Qwen35Control
from qwen35_layer_checkpoint import LayerCheckpointControl
from train_qwen35_controls import balance_loss
from optim import RecipeOptimizer,ownership


def nodes(tensor):
    seen=set();stack=[tensor.grad_fn]
    while stack:
        node=stack.pop()
        if node is None or node in seen:continue
        seen.add(node);stack.extend(child for child,_ in node.next_functions)
    return len(seen)


class LayerCheckpointTests(unittest.TestCase):
    def test_values_gradients_optimizer_and_graph_size(self):
        torch.set_num_threads(1);torch.manual_seed(332);device=os.environ.get('LAB_TEST_DEVICE','cpu')
        c=ControlConfig(vocab_size=32,width=8,heads=2,kv_heads=1,head_dim=8,rotary_dim=2,gdn_k_dim=8,gdn_v_dim=8,hidden=4,experts=4,top_k=2,query_chunk=17,scan_chunk=17)
        ids=torch.randint(0,32,(1,129),device=device);labels=torch.randint(0,32,(1,129),device=device)
        for arm in ['full','swa','gdn']:
            with self.subTest(arm=arm):
                reference=Qwen35Control(c,arm).to(device).train();actual=LayerCheckpointControl(c,arm).to(device).train();actual.load_state_dict(reference.state_dict())
                self.assertEqual(reference.state_dict().keys(),actual.state_dict().keys())
                a,ar=actual(ids);b,br=reference(ids)
                torch.testing.assert_close(a,b,atol=2e-6,rtol=2e-5)
                self.assertLess(nodes(a),nodes(b)//2)
                la=F.cross_entropy(a.flatten(0,1),labels.flatten())+.001*balance_loss(ar,c.experts,c.top_k)
                lb=F.cross_entropy(b.flatten(0,1),labels.flatten())+.001*balance_loss(br,c.experts,c.top_k)
                la.backward();lb.backward()
                for pa,pb in zip(actual.parameters(),reference.parameters(),strict=True):
                    if pa.grad is None or pb.grad is None:self.assertIs(pa.grad,pb.grad)
                    else:torch.testing.assert_close(pa.grad,pb.grad,atol=2e-6,rtol=2e-4)
                # CPU logical Muon/Adam state is also compared after one actual update.
                if device=='cpu':
                    oa=RecipeOptimizer(ownership(actual),3e-4);ob=RecipeOptimizer(ownership(reference),3e-4);oa.step();ob.step()
                    for pa,pb in zip(actual.parameters(),reference.parameters(),strict=True):torch.testing.assert_close(pa,pb,atol=2e-6,rtol=2e-4)


if __name__=='__main__':unittest.main()
