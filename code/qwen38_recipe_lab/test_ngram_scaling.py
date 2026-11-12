"""Verify constructed parameter totals and shared initialization for Tables 8/9."""
from pathlib import Path
import unittest

import torch

from model import Config
from ngram_scaling import NGramScaling
from optim import ownership
from qwen35_quality_data import digest, load


class ScalingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.profile = load(Path('profiles/ngram_table8_9_mac_capacity.json'))
        self.assertTrue(all(digest(path) == sha for path, sha in self.profile['source_hashes'].items()))
        self.config = Config(**self.profile['config'])

    def check_family(self, name, fixed):
        reference = None
        total = None
        for index, row in enumerate(self.profile[name]):
            model = NGramScaling(self.config, str(index), row, self.profile['position'], seed=42)
            size = sum(p.numel() for p in model.parameters())
            if reference is None:
                reference = model.state_dict()
                total = size
            else:
                expected = total if fixed else total+row['table']['parameters']+row['added_ple_adapter_parameters']
                self.assertEqual(size, expected)
                for key, value in model.state_dict().items():
                    if '.ple.' in key:
                        continue
                    other = reference[key]
                    if key.endswith('.moe.router.weight'):
                        other = other[:value.shape[0]]
                    self.assertTrue(torch.equal(value, other), key)
            plans = {name: pieces for name, _, pieces in ownership(model)}
            self.assertFalse(any('.indexer.' in key for key in plans))
            if row['table']['rows']:
                name = f"layers.{self.profile['position']-1}.ple.embedding.ngram_embedding.weight"
                self.assertEqual([piece['kind'] for piece in plans[name]], ['adam'])

    def test_table8_constructed_total_is_exact(self):
        self.check_family('table8_fixed_total', fixed=True)

    def test_table9_growth_keeps_the_experts(self):
        self.check_family('table9_growing_total', fixed=False)


if __name__ == '__main__':
    unittest.main()
