"""Numerical checks for the frozen-target MTP training path."""
import io
import unittest

import torch

from model import Model
from mtp import MultiTokenPredictor
from optim import RecipeOptimizer, ownership
from qwen35_long_phase import exact
from test_recipe import tiny
from train_mtp_draft import losses


class DraftTrainingTests(unittest.TestCase):
    def test_exact_resume_and_frozen_target(self):
        torch.set_num_threads(1)
        torch.manual_seed(189)
        config = tiny()
        target = Model(config).eval().requires_grad_(False)
        frozen = {name: value.clone() for name, value in target.state_dict().items()}
        profile = dict(context=16, depths=4)

        def initialize():
            torch.manual_seed(42)
            draft = MultiTokenPredictor(config)
            return draft, RecipeOptimizer(ownership(draft), 3e-4), torch.Generator().manual_seed(79)

        def update(draft, optimizer, rng):
            ids = torch.randint(config.vocab_size, (1, 21), generator=rng)
            optimizer.zero_grad(set_to_none=True)
            ce, kl = losses(target, draft, ids, profile)
            self.assertEqual(ce.shape, (4,))
            self.assertTrue(torch.isfinite(ce).all() and torch.isfinite(kl))
            (ce.mean() + 0.1*kl).backward()
            indexer = draft.layer.mix.indexer.qk.weight
            self.assertTrue(torch.any(indexer.grad != 0))
            optimizer.step()
            return ce.detach().clone(), kl.detach().clone()

        expected = initialize()
        steps = [update(*expected) for _ in range(4)]
        first = initialize()
        for _ in range(2):
            update(*first)
        snapshot = io.BytesIO()
        torch.save((first[0].state_dict(), first[1].state_dict(), first[2].get_state()), snapshot)
        snapshot.seek(0)
        model, opt, rng = torch.load(snapshot, weights_only=True)
        second = initialize()
        second[0].load_state_dict(model)
        second[1].load_state_dict(opt)
        second[2].set_state(rng)
        for expected_row in steps[2:]:
            self.assertTrue(all(torch.equal(a, b) for a, b in zip(expected_row, update(*second))))
        for name, value in second[0].state_dict().items():
            self.assertTrue(torch.equal(value, expected[0].state_dict()[name]))
        exact(expected[1].state_dict(), second[1].state_dict())
        self.assertTrue(torch.equal(expected[2].get_state(), second[2].get_state()))
        self.assertTrue(all(parameter.grad is None for parameter in target.parameters()))
        self.assertTrue(all(torch.equal(value, frozen[name]) for name, value in target.state_dict().items()))


if __name__ == '__main__':
    unittest.main()
