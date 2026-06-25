import unittest

import numpy as np
import torch

import train_mnist_ddpm as ddpm


class ReproducibilityTest(unittest.TestCase):
    def test_seed_setup_is_inline_not_helper_functions(self):
        self.assertFalse(hasattr(ddpm, "set_seed"))
        self.assertFalse(hasattr(ddpm, "seed_worker"))

    def test_forward_diffusion_repeats_with_same_seed(self):
        schedule = ddpm.NoiseSchedule(timesteps=10)
        x0 = torch.linspace(-1.0, 1.0, steps=2 * ddpm.IN_CHANNELS * ddpm.IMG_SIZE * ddpm.IMG_SIZE)
        x0 = x0.view(2, ddpm.IN_CHANNELS, ddpm.IMG_SIZE, ddpm.IMG_SIZE)
        t = torch.tensor([0, 9], dtype=torch.long)

        np.random.seed(123)
        torch.manual_seed(123)
        xt1, eps1 = ddpm.forward_diffusion(x0, t, schedule)

        np.random.seed(123)
        torch.manual_seed(123)
        xt2, eps2 = ddpm.forward_diffusion(x0, t, schedule)

        self.assertTrue(torch.equal(eps1, eps2))
        self.assertTrue(torch.equal(xt1, xt2))


if __name__ == "__main__":
    unittest.main()
