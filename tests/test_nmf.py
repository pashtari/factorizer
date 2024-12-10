import unittest

import torch
import factorizer as ft


class TestNMF(unittest.TestCase):
    def setUp(self):
        self.size = (2, 4, 8, 16)
        *_, M, N = self.size
        self.rank = 3
        self.nmf = ft.NMF(size=(M, N), rank=self.rank, init="uniform", solver="hals")

    def test_decompose(self):
        x = torch.rand(self.size, requires_grad=True)
        u, v = self.nmf.decompose(x)
        self.assertEqual(u.shape, (*self.size[:-2], self.size[-2], self.rank))
        self.assertEqual(v.shape, (*self.size[:-2], self.size[-1], self.rank))
        self.assertTrue((u >= 0).all())
        self.assertTrue((v >= 0).all())

    def test_forward(self):
        x = torch.rand(self.size, requires_grad=True)
        y = self.nmf(x)
        self.assertEqual(y.shape, x.shape)

    def test_reconstruct(self):
        u = torch.rand((*self.size[:-2], self.size[-2], self.rank), requires_grad=True)
        v = torch.rand((*self.size[:-2], self.size[-1], self.rank), requires_grad=True)
        y = self.nmf.reconstruct(u, v)
        self.assertEqual(y.shape, self.size)

    def test_loss(self):
        x = torch.rand(self.size, requires_grad=True)
        u = torch.rand((*self.size[:-2], self.size[-2], self.rank), requires_grad=True)
        v = torch.rand((*self.size[:-2], self.size[-1], self.rank), requires_grad=True)
        loss = self.nmf.loss(x, u, v)
        self.assertEqual(loss.shape, self.size[:1])
        self.assertTrue((loss >= 0).all())


if __name__ == "__main__":
    unittest.main()
