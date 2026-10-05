"""Small numerical checks of the existing baseline, requiring CPU PyTorch."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "run in trdp2-r2 for CPU PyTorch checks")
class BaselineTest(unittest.TestCase):
    def test_tv_ramp_value_axis_symmetry_and_gradient(self):
        from r2_gaussian.utils.loss_utils import tv_3d_loss
        vol = torch.arange(4., dtype=torch.float64)[:, None, None].expand(4, 5, 6).clone().requires_grad_()
        loss = tv_3d_loss(vol, "mean")
        self.assertAlmostEqual(loss.item(), 90 / (90 + 96 + 100))
        self.assertAlmostEqual(loss.item(), tv_3d_loss(vol.permute(2, 0, 1), "mean").item())
        loss.backward()
        self.assertTrue(torch.isfinite(vol.grad).all())
        self.assertGreater(vol.grad.abs().sum().item(), 0)

    def test_constant_tv_zero_with_finite_gradients(self):
        from r2_gaussian.utils.loss_utils import tv_3d_loss
        vol = torch.ones(4, 5, 6, requires_grad=True)
        loss = tv_3d_loss(vol, "mean")
        loss.backward()
        self.assertEqual(loss.item(), 0)
        self.assertTrue(torch.isfinite(vol.grad).all())
        self.assertEqual(vol.grad.abs().sum().item(), 0)

    def test_repository_psnr_fixed_range_no_clipping(self):
        from r2_gaussian.utils.image_utils import metric_vol
        gt = torch.ones(4, 5, 6)
        pred = gt + 0.1
        value, _ = metric_vol(gt, pred, "psnr")
        self.assertAlmostEqual(value, 20, places=4)
        doubled_range, _ = metric_vol(gt, pred, "psnr", pixel_max=2)
        self.assertAlmostEqual(doubled_range - value, 6.0206, places=4)


if __name__ == "__main__":
    unittest.main()
