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


class DDPMPaperTricksTest(unittest.TestCase):
    def test_training_stability_config_matches_paper_style_defaults(self):
        self.assertEqual(ddpm.LR, 2e-4)
        self.assertEqual(ddpm.GRAD_CLIP_NORM, 1.0)
        self.assertAlmostEqual(ddpm.EMA_DECAY, 0.9999)

    def test_noise_schedule_exposes_posterior_variance(self):
        schedule = ddpm.NoiseSchedule(timesteps=10)

        self.assertTrue(hasattr(schedule, "posterior_variance"))
        self.assertEqual(schedule.posterior_variance.shape, schedule.beta.shape)
        self.assertEqual(schedule.posterior_variance[0].item(), 0.0)
        self.assertTrue(torch.all(schedule.posterior_variance[1:] > 0))
        self.assertTrue(torch.all(schedule.posterior_variance[1:] <= schedule.beta[1:]))

    def test_unet_uses_resblocks_with_groupnorm(self):
        model = ddpm.UNet(base_channels=8, time_emb_dim=16)

        self.assertTrue(any(isinstance(module, ddpm.ResBlock) for module in model.modules()))
        self.assertTrue(any(isinstance(module, torch.nn.GroupNorm) for module in model.modules()))

        x = torch.randn(2, ddpm.IN_CHANNELS, ddpm.IMG_SIZE, ddpm.IMG_SIZE)
        t = torch.tensor([0, 9], dtype=torch.long)
        y = model(x, t)
        self.assertEqual(y.shape, x.shape)

    def test_ema_updates_and_copies_shadow_weights(self):
        model = torch.nn.Linear(2, 1, bias=False)
        ema = ddpm.EMA(model, decay=0.5)

        with torch.no_grad():
            original = model.weight.detach().clone()
            model.weight.add_(2.0)

        ema.update(model)
        expected_shadow = original * 0.5 + model.weight.detach() * 0.5
        self.assertTrue(torch.allclose(ema.shadow["weight"], expected_shadow))

        target = torch.nn.Linear(2, 1, bias=False)
        ema.copy_to(target)
        self.assertTrue(torch.allclose(target.weight, expected_shadow))

    def test_sampling_supports_beta_and_posterior_variance(self):
        schedule = ddpm.NoiseSchedule(timesteps=4)
        model = ddpm.UNet(base_channels=8, time_emb_dim=16)

        beta_samples = ddpm.sample(model, schedule, num_images=2, variance_type="beta", device="cpu")
        posterior_samples = ddpm.sample(model, schedule, num_images=2, variance_type="posterior", device="cpu")

        self.assertEqual(beta_samples.shape, (2, ddpm.IN_CHANNELS, ddpm.IMG_SIZE, ddpm.IMG_SIZE))
        self.assertEqual(posterior_samples.shape, beta_samples.shape)
        self.assertTrue(torch.isfinite(beta_samples).all())
        self.assertTrue(torch.isfinite(posterior_samples).all())

        with self.assertRaises(ValueError):
            ddpm.sample(model, schedule, num_images=2, variance_type="unknown", device="cpu")


if __name__ == "__main__":
    unittest.main()
