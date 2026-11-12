import os,unittest
import torch
from model import Attention,Config
from inference import Inference
from bounded_dense_inference import BoundedDenseInference


class BoundedDenseTests(unittest.TestCase):
    @torch.no_grad()
    def test_group_mapping_absolute_mask_and_partitioned_cache(self):
        device=os.environ.get('LAB_TEST_DEVICE','cpu');torch.manual_seed(17)
        for kv_heads in (1,2):
            with self.subTest(kv_heads=kv_heads):
                config=Config(width=16,heads=4,kv_heads=kv_heads,head_dim=16,index_dim=16,rotary_dim=8)
                layer=Attention(config).to(device);x=torch.randn(2,19,16,device=device)
                expected=layer(x,'pretrain')[0]
                runner=BoundedDenseInference(None);cache={};pieces=[];position=0
                for count in (3,7,1,8):
                    chunk=x[:,position:position+count];before={key:value.clone() for key,value in cache.items()}
                    actual=runner.attention(layer,chunk,cache)
                    reference=Inference(None,'pretrain').attention(layer,chunk,before)
                    torch.testing.assert_close(actual,reference,rtol=2e-4,atol=2e-5)
                    pieces.append(actual);position+=count
                torch.testing.assert_close(torch.cat(pieces,1),expected,rtol=2e-4,atol=2e-5)
                self.assertEqual(cache['k'].shape,(2,19,kv_heads,16))


if __name__=='__main__':unittest.main()
