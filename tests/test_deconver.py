import unittest

import torch
from torch import nn
import factorizer as ft


class TestDeconverModules(unittest.TestCase):
    def setUp(self) -> None:
        # Common setup for all tests
        self.batch_size = 1
        self.spatial_size = (48, 48, 48)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_deconv(self):
        # Deconv initialization
        channels = 16
        x = torch.rand(
            self.batch_size, channels, *self.spatial_size, requires_grad=True
        ).to(self.device)

        deconv = ft.Deconv(
            channels=channels,
            source_channels=1,
            groups=8,
            kernel_size=(3, 3, 3),
            update_source=True,
            update_filter=True,
            num_iters=5,
            num_grad_iters=None,
        ).to(self.device)

        # Parameter count
        num_params = sum(p.numel() for p in deconv.parameters() if p.requires_grad)
        self.assertGreater(num_params, 0, "Deconv should have trainable parameters")

        # Forward pass
        y = deconv(x)
        self.assertEqual(y.shape, x.shape, "Deconv output shape should match input shape")
        self.assertIsNotNone(y, "Deconv output should not be None")

    def test_deconv_mixer(self):
        # DeconvMixer initialization
        in_channels = 16
        out_channels = 16
        x = torch.rand(
            self.batch_size, in_channels, *self.spatial_size, requires_grad=True
        ).to(self.device)

        deconv_mixer = ft.DeconvMixer(
            in_channels=in_channels,
            out_channels=out_channels,
            act=nn.ReLU,
            kernel_size=(3, 3, 3),
            update_source=True,
            update_filter=False,
            num_iters=5,
            num_grad_iters=1,
            dropout=0.1,
        ).to(self.device)

        # Parameter count
        num_params = sum(p.numel() for p in deconv_mixer.parameters() if p.requires_grad)
        self.assertGreater(num_params, 0, "DeconvMixer should have trainable parameters")

        # Forward pass
        y = deconv_mixer(x)
        expected_shape = (self.batch_size, out_channels, *self.spatial_size)
        self.assertEqual(y.shape, expected_shape, "DeconvMixer output shape mismatch")
        self.assertTrue(
            torch.isfinite(y).all(), "DeconvMixer output should not contain NaNs or Infs"
        )

    def test_deconver_block(self):
        # DeconverBlock initialization
        channels = 16
        x = torch.rand(
            self.batch_size, channels, *self.spatial_size, requires_grad=True
        ).to(self.device)

        deconver_block = ft.DeconverBlock(
            channels=channels,
            kernel_size=(3, 3, 3),
            update_filter=False,
            num_iters=3,
            num_grad_iters=1,
            mlp_ratio=3,
        ).to(self.device)

        # Forward pass
        y = deconver_block(x)
        self.assertEqual(y.shape, x.shape, "DeconverBlock output shape mismatch")
        self.assertTrue(
            torch.isfinite(y).all(),
            "DeconverBlock output should not contain NaNs or Infs",
        )

    def test_deconver_stage(self):
        # Test DeconverStage initialization
        in_channels = 16
        out_channels = 32
        x = torch.rand(
            self.batch_size, in_channels, *self.spatial_size, requires_grad=True
        ).to(self.device)

        deconver_stage = ft.DeconverStage(
            in_channels=in_channels,
            out_channels=out_channels,
            depth=2,
            kernel_size=(3, 3, 3),
            update_filter=False,
            num_iters=3,
            num_grad_iters=1,
            mlp_ratio=2,
        ).to(self.device)

        # Forward pass
        y = deconver_stage(x)
        expected_shape = (self.batch_size, out_channels, *self.spatial_size)
        self.assertEqual(y.shape, expected_shape, "DeconverStage output shape mismatch")
        self.assertTrue(
            torch.isfinite(y).all(),
            "DeconverStage output should not contain NaNs or Infs",
        )

    def test_deconver_model(self):
        # Deconver main model initialization
        in_channels = 4
        out_channels = 3
        deconver = ft.Deconver(
            in_channels=in_channels,
            out_channels=out_channels,
            spatial_dims=3,
            encoder_depth=(1, 1, 1, 1, 1),
            encoder_width=(32, 64, 128, 256, 512),
            strides=(1, 2, 2, 2, 2),
            decoder_depth=(1, 1, 1, 1),
            act=nn.ReLU,
            groups=-1,
            ratio=1,
            kernel_size=(3, 3, 3),
            update_filter=False,
            num_iters=5,
            num_grad_iters=1,
            mlp_ratio=2,
            dropout=0.1,
        ).to(self.device)

        # Parameter count
        num_params = sum(p.numel() for p in deconver.parameters() if p.requires_grad)
        self.assertGreater(
            num_params, 0, "Deconver model should have trainable parameters"
        )

        # Forward pass
        x = torch.rand(
            self.batch_size, in_channels, *self.spatial_size, requires_grad=True
        ).to(self.device)
        y = deconver(x)
        expected_shape = (self.batch_size, out_channels, *self.spatial_size)
        self.assertEqual(y.shape, expected_shape, "Deconver model output shape mismatch")
        self.assertTrue(
            torch.isfinite(y).all(),
            "Deconver model output should not contain NaNs or Infs",
        )

        # Test different batch sizes
        for batch_size in [2, 3]:
            x = torch.rand(batch_size, in_channels, *self.spatial_size).to(self.device)
            y = deconver(x)
            self.assertEqual(
                y.shape[0],
                batch_size,
                f"Output batch size mismatch for batch size {batch_size}",
            )


if __name__ == "__main__":
    unittest.main()
