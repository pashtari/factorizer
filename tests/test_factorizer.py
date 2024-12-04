import unittest
import torch
from torch import nn
import factorizer as ft


class TestFactorizerModules(unittest.TestCase):
    def setUp(self) -> None:
        self.batch_size = 1
        self.spatial_size = (64, 64, 64)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_factmixer(self):
        in_channels = 16
        out_channels = 16
        x = torch.rand(
            self.batch_size, in_channels, *self.spatial_size, requires_grad=True
        ).to(self.device)

        factmixer = ft.FactMixer(
            in_channels=in_channels,
            out_channels=out_channels,
            spatial_size=self.spatial_size,
            reshape=(ft.Matricize, {"num_heads": 1, "grid_size": 1}),
            act=nn.ReLU,
            factorize=ft.NMF,
            rank=1,
            num_iters=5,
            num_grad_steps=None,
            init="uniform",
            solver="mu",
            dropout=0.1,
        ).to(self.device)

        # Parameter count
        num_params = sum(p.numel() for p in factmixer.parameters() if p.requires_grad)
        self.assertGreater(num_params, 0, "FactMixer should have trainable parameters")

        # Forward pass
        y = factmixer(x)
        self.assertEqual(
            y.shape, x.shape, "FactMixer output shape should match input shape"
        )
        self.assertTrue(
            torch.isfinite(y).all(), "FactMixer output should not contain NaNs or Infs"
        )

    def test_factorizer_block(self):
        channels = 16
        x = torch.rand(
            self.batch_size, channels, *self.spatial_size, requires_grad=True
        ).to(self.device)

        factorizer_block = ft.FactorizerBlock(
            channels=channels,
            spatial_size=self.spatial_size,
            reshape=(ft.Matricize, {"num_heads": 1, "grid_size": 1}),
            act=nn.ReLU,
            factorize=ft.NMF,
            rank=1,
            num_iters=5,
            num_grad_steps=None,
            init="uniform",
            solver="mu",
            mlp_ratio=2,
            dropout=0.1,
        ).to(self.device)

        # Forward pass
        y = factorizer_block(x)
        self.assertEqual(
            y.shape, x.shape, "FactorizerBlock output shape should match input shape"
        )
        self.assertTrue(
            torch.isfinite(y).all(),
            "FactorizerBlock output should not contain NaNs or Infs",
        )

    def test_factorizer_stage(self):
        in_channels = 16
        out_channels = 32
        x = torch.rand(
            self.batch_size, in_channels, *self.spatial_size, requires_grad=True
        ).to(self.device)

        factorizer_stage = ft.FactorizerStage(
            in_channels=in_channels,
            out_channels=out_channels,
            spatial_size=self.spatial_size,
            depth=2,
            reshape=(ft.Matricize, {"num_heads": 1, "grid_size": 1}),
            act=nn.ReLU,
            factorize=ft.NMF,
            rank=1,
            num_iters=5,
            init="uniform",
            solver="mu",
            mlp_ratio=3,
            dropout=0.1,
        ).to(self.device)

        # Forward pass
        y = factorizer_stage(x)
        expected_shape = (self.batch_size, out_channels, *self.spatial_size)
        self.assertEqual(y.shape, expected_shape, "FactorizerStage output shape mismatch")
        self.assertTrue(
            torch.isfinite(y).all(),
            "FactorizerStage output should not contain NaNs or Infs",
        )

    def test_factorizer_model(self):
        in_channels = 4
        out_channels = 3
        factorizer_model = ft.Factorizer(
            in_channels=in_channels,
            out_channels=out_channels,
            spatial_size=self.spatial_size,
            encoder_depth=(1, 1, 1, 1),
            encoder_width=(32, 64, 128, 256),
            strides=(1, 2, 2, 2),
            decoder_depth=(1, 1, 1),
            reshape=(ft.SWMatricize, {"num_heads": 8, "patch_size": 4}),
            act=nn.ReLU,
            factorize=ft.NMF,
            rank=1,
            num_iters=5,
            num_grad_steps=None,
            init="uniform",
            solver="hals",
            mlp_ratio=2,
            dropout=0.1,
        ).to(self.device)

        # Parameter count
        num_params = sum(
            p.numel() for p in factorizer_model.parameters() if p.requires_grad
        )
        self.assertGreater(
            num_params, 0, "Factorizer model should have trainable parameters"
        )

        # Forward pass
        x = torch.rand(
            self.batch_size, in_channels, *self.spatial_size, requires_grad=True
        ).to(self.device)
        y = factorizer_model(x)
        expected_shape = (self.batch_size, out_channels, *self.spatial_size)
        self.assertEqual(
            y.shape, expected_shape, "Factorizer model output shape mismatch"
        )
        self.assertTrue(
            torch.isfinite(y).all(),
            "Factorizer model output should not contain NaNs or Infs",
        )

        # Test with different batch sizes
        for batch_size in [2, 3]:
            x = torch.rand(batch_size, in_channels, *self.spatial_size).to(self.device)
            y = factorizer_model(x)
            self.assertEqual(
                y.shape[0],
                batch_size,
                f"Output batch size mismatch for batch size {batch_size}",
            )


if __name__ == "__main__":
    unittest.main()
