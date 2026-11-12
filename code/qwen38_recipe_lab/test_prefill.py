import os
import unittest
import torch
from model import Model
from inference import Inference
from prefill_inference import PrefillInference
from test_recipe import tiny


class PrefillTests(unittest.TestCase):
    def test_public_chunk_prefill_and_recurrent_decode_match(self):
        torch.manual_seed(211);device = os.environ.get('LAB_TEST_DEVICE', 'cpu')
        c = tiny(heads=12, head_dim=256, rotary_dim=64, index_dim=128, gdn_k_dim=128, gdn_v_dim=128)
        model = Model(c).to(device).eval();ids = torch.randint(0, c.vocab_size, (2, 145), device=device);ids[:, 63] = c.eos
        with torch.no_grad():
            for stage in ['pretrain', 'sparse']:
                for chunk in [31, 128]:
                    old = Inference(model, stage, prefill_chunk=chunk);new = PrefillInference(model, stage, prefill_chunk=chunk)
                    a, b = {}, {};left = old(ids, a);right = new(ids, b)
                    torch.testing.assert_close(left, right, atol=5e-5, rtol=5e-4)
                    for i in range(4):
                        for key in ['conv', 'state']:
                            if key in a[str(i)]['mix']:
                                torch.testing.assert_close(a[str(i)]['mix'][key], b[str(i)]['mix'][key], atol=5e-5, rtol=5e-4)
                    for step in range(16):
                        token = left.argmax(-1)[:, None]
                        self.assertTrue(torch.equal(token[:, 0], right.argmax(-1)))
                        left = old(token, a);right = new(token, b)
                        torch.testing.assert_close(left, right, atol=5e-5, rtol=5e-4)
