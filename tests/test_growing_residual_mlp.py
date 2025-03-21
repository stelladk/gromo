from unittest import TestCase, main

import torch

from gromo.containers.growing_residual_mlp import GrowingResidualMLP
from gromo.utils.utils import global_device


class TestGrowingResidualMLP(TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.input_shape = (3, 32, 32)
        self.x = torch.randn(2, *self.input_shape, device=global_device())
        self.model = GrowingResidualMLP(
            in_features=self.input_shape,
            out_features=10,
            num_features=16,
            hidden_features=8,
            num_blocks=2,
            activation=torch.nn.ReLU(),
        )

    def test_init(self):
        l1 = GrowingResidualMLP(
            in_features=self.input_shape,
            out_features=10,
            num_features=16,
            hidden_features=8,
            num_blocks=2,
            activation=torch.nn.ReLU(),
        )

        self.assertIsInstance(l1, GrowingResidualMLP)
        self.assertIsInstance(l1, torch.nn.Module)


if __name__ == "__main__":
    main()
