from copy import deepcopy
from typing import Any, Dict, Tuple
from unittest import TestCase, main

import torch

from gromo.modules.linear_growing_module import (
    LinearGrowingModule,
    LinearMergeGrowingModule,
)
from gromo.utils.tensor_statistic import TensorStatistic
from gromo.utils.utils import global_device
from tests.torch_unittest import TorchTestCase
from tests.unittest_tools import unittest_parametrize


# Test configuration constants
class TestConfig:
    """Centralized test configuration to reduce magic numbers."""

    N_SAMPLES = 11
    C_FEATURES = 5
    BATCH_SIZE = 10
    RANDOM_SEED = 0
    DEFAULT_TOLERANCE = 1e-8
    REDUCED_TOLERANCE = 1e-7

    # Layer dimensions for different test scenarios
    LAYER_DIMS = {
        "small": (1, 1),
        "medium": (3, 3),
        "large": (5, 7),
        "demo_1": (5, 3),
        "demo_2": (3, 7),
        "merge_prev": (5, 3),
        "merge_next": (3, 7),
    }


def theoretical_s_1(n: int, c: int) -> Tuple[torch.Tensor, ...]:
    """
    Compute the theoretical value of the tensor S for the input and output of
    weight matrix W = (0 ... 0 \\ 0 1 0 ... 0 \\ 0 0 2 0 ... 0 \\ ... \\ 1 ... 1).

    Optimized version with better variable names and cached computations.

    Parameters
    ----------
    n: int
        number of samples
    c: int
        number of features

    Returns
    -------
    x1:
        input tensor 1
    x2:
        input tensor 2
    is1:
        theoretical value of the tensor nS for x1
    is2:
        theoretical value of the tensor 2nS for (x1, x2)
    os1:
        theoretical value of the tensor nS for the output of W(x1)
    os2:
        theoretical value of the tensor 2nS for the output of W((x1, x2))
    """
    # Pre-compute common values to avoid redundant calculations
    device = global_device()
    arange_c = torch.arange(c, device=device)
    ones_c = torch.ones(c, dtype=torch.long, device=device)
    arange_n = torch.arange(n, device=device)

    # Input statistics matrices
    is0 = arange_c.view(-1, 1) @ arange_c.view(1, -1)
    isc = arange_c.view(-1, 1) @ ones_c.view(1, -1)
    isc = isc + isc.T
    is1 = torch.ones(c, c, device=device)

    # Output statistics matrices
    arange_c_plus1 = torch.arange(c + 1, device=device)
    va_im = arange_c_plus1**2
    va_im[-1] = c * (c - 1) // 2
    v1_im = arange_c_plus1

    os0 = va_im.view(-1, 1) @ va_im.view(1, -1)
    osc = va_im.view(-1, 1) @ v1_im.view(1, -1)
    osc = osc + osc.T
    os1 = v1_im.view(-1, 1) @ v1_im.view(1, -1)

    # Generate input tensors
    x1 = torch.ones(n, c, device=device)
    x1 *= arange_n.view(-1, 1)

    x2 = torch.tile(arange_c, (n, 1))
    x2 += arange_n.view(-1, 1)
    x2 = x2.to(device)

    # Pre-compute common coefficient
    coeff_1 = n * (n - 1) * (2 * n - 1) // 6
    coeff_2_partial = n * (n - 1) // 2
    coeff_3 = n * (n - 1) * (2 * n - 1) // 3

    # Theoretical values
    is_theory_1 = coeff_1 * is1
    os_theory_1 = coeff_1 * os1
    is_theory_2 = n * is0 + coeff_2_partial * isc + coeff_3 * is1
    os_theory_2 = n * os0 + coeff_2_partial * osc + coeff_3 * os1

    return x1, x2, is_theory_1, is_theory_2, os_theory_1, os_theory_2


class TestLinearGrowingModuleBase(TorchTestCase):
    """Base class with common helper methods for linear growing module tests."""

    @classmethod
    def setUpClass(cls):
        """Set up class-level constants and utilities."""
        cls.config = TestConfig()

    def setUp(self):
        """Common setup for all tests."""
        self.n = self.config.N_SAMPLES
        self.c = self.config.C_FEATURES
        # This assert is checking that the test is correct and not that the code is correct
        # that why it is not a self.assert*
        assert self.n % 2 == 1  # Ensure n is odd for theoretical calculations

        # Set deterministic seed for reproducible tests
        torch.manual_seed(self.config.RANDOM_SEED)

        # Common test data
        self.input_x = torch.randn((self.n, self.c), device=global_device())

    def create_weight_matrix(self) -> torch.Tensor:
        """Create standard test weight matrix."""
        weight_matrix = torch.ones(self.c + 1, self.c, device=global_device())
        weight_matrix[:-1] = torch.diag(torch.arange(self.c)).to(global_device())
        return weight_matrix

    def create_demo_layers(
        self, bias: bool
    ) -> Tuple[LinearGrowingModule, LinearGrowingModule]:
        """Create demo layers for testing with specified bias configuration."""
        demo_layer_1 = LinearGrowingModule(
            *self.config.LAYER_DIMS["demo_1"],
            use_bias=bias,
            name=f"L1({'bias' if bias else 'no_bias'})",
            device=global_device(),
        )
        demo_layer_2 = LinearGrowingModule(
            *self.config.LAYER_DIMS["demo_2"],
            use_bias=bias,
            name=f"L2({'bias' if bias else 'no_bias'})",
            previous_module=demo_layer_1,
            device=global_device(),
        )
        return demo_layer_1, demo_layer_2

    def create_linear_layer(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        name: str | None = None,
    ) -> LinearGrowingModule:
        """Helper to create a LinearGrowingModule with common settings."""
        return LinearGrowingModule(
            in_features=in_features,
            out_features=out_features,
            use_bias=bias,
            name=name or f"layer_{in_features}_{out_features}",
            device=global_device(),
        )

    def assert_layer_properties(
        self,
        layer: LinearGrowingModule,
        expected_in: int,
        expected_out: int,
        expected_params: int,
    ):
        """Helper to assert common layer properties."""
        self.assertEqual(layer.in_features, expected_in)
        self.assertEqual(layer.out_features, expected_out)
        self.assertEqual(layer.number_of_parameters(), expected_params)


class TestLinearGrowingModule(TestLinearGrowingModuleBase):
    """Optimized test class for LinearGrowingModule with improved structure."""

    def setUp(self):
        """Enhanced setUp using base class helpers."""
        super().setUp()

        # Create weight matrix using helper method
        self.weight_matrix_1 = self.create_weight_matrix()

        # Create demo layers for different bias configurations
        self.demo_layers = {}
        for bias in (True, False):
            self.demo_layers[bias] = self.create_demo_layers(bias)

    def test_compute_s(self):
        """Test S tensor computation with optimized setup."""
        x1, x2, is_th_1, is_th_2, os_th_1, os_th_2 = theoretical_s_1(self.n, self.c)

        # Create modules using helper methods
        output_module = LinearMergeGrowingModule(in_features=self.c + 1, name="output")
        layer = self.create_linear_layer(self.c, self.c + 1, bias=False, name="layer1")
        layer.next_module = output_module
        output_module.set_previous_modules([layer])

        net = torch.nn.Sequential(layer, output_module)
        layer.layer.weight.data = self.weight_matrix_1

        # Initialize computation automatically handles storage flags
        layer.init_computation()
        output_module.init_computation()

        # Forward pass 1 - extracted to reduce duplication
        self._run_forward_pass_and_update(net, layer, output_module, x1)
        self._assert_tensor_values(layer, output_module, is_th_1, os_th_1, self.n)

        # Forward pass 2
        self._run_forward_pass_and_update(net, layer, output_module, x2)
        self._assert_tensor_values(layer, output_module, is_th_2, os_th_2, 2 * self.n)

    def _run_forward_pass_and_update(self, net, layer, output_module, input_data):
        """Helper method to run forward pass and update tensors."""
        y = net(input_data.float().to(global_device()))
        loss = torch.norm(y)  # Compute loss to generate gradients
        loss.backward()  # Generate gradients for tensor M computation
        layer.update_computation()
        output_module.update_computation()

    def _assert_tensor_values(
        self, layer, output_module, is_theoretical, os_theoretical, divisor
    ):
        """Helper method to assert tensor values."""
        device = global_device()
        # Input S
        self.assertAllClose(layer.tensor_s(), is_theoretical.float().to(device) / divisor)
        # Output S
        self.assertAllClose(
            output_module.tensor_s()[: self.c + 1, : self.c + 1],
            os_theoretical.float().to(device) / divisor,
        )
        # Input S computed from merge layer
        self.assertAllClose(
            output_module.previous_tensor_s(),
            is_theoretical.float().to(device) / divisor,
        )

    @unittest_parametrize(
        (
            {"force_pseudo_inverse": True},
            {"force_pseudo_inverse": False},
            {"update_layer": False},
        )
    )
    def test_compute_delta(
        self, force_pseudo_inverse: bool = False, update_layer: bool = True
    ):
        for reduction in {"mixed"}:  # { "mean", "sum"} do not work
            # mean: batch is divided by the number of samples in the batch
            # and the total is divided by the number of batches
            # mixed: batch is not divided
            # but the total is divided by the number of batches * batch_size
            # sum: batch is not divided
            # and the total is not divided
            batch_red = self.c if reduction == "mean" else 1
            loss_func = lambda x, y: torch.norm(x - y) ** 2 / batch_red

            for alpha in (0.1, 1.0, 10.0):
                layer = LinearGrowingModule(self.c, self.c, use_bias=False, name="layer1")
                layer.layer.weight.data = torch.zeros_like(
                    layer.layer.weight, device=global_device()
                )
                layer.init_computation()

                for _ in range(nb_batch := 3):
                    x = alpha * torch.eye(self.c, device=global_device())
                    y = layer(x)
                    loss = loss_func(x, y)
                    loss.backward()

                    layer.update_computation()

                # S
                self.assertAllClose(
                    layer.tensor_s(),
                    alpha**2 * torch.eye(self.c, device=global_device()) / self.c,
                    message=f"Error in S for {reduction=}, {alpha=}",
                )

                # dL / dA
                self.assertAllClose(
                    layer.pre_activity.grad,
                    -2 * alpha * torch.eye(self.c, device=global_device()) / batch_red,
                    message=f"Error in dL/dA for {reduction=}, {alpha=}",
                )

                # M
                self.assertAllClose(
                    layer.tensor_m(),
                    -2
                    * alpha**2
                    * torch.eye(self.c, device=global_device())
                    / self.c
                    / batch_red,
                    message=f"Error in M for {reduction=}, {alpha=}",
                )

                # dW*
                w, _, fo = layer.compute_optimal_delta(
                    force_pseudo_inverse=force_pseudo_inverse, update=update_layer
                )
                self.assertAllClose(
                    w,
                    -2 * torch.eye(self.c, device=global_device()) / batch_red,
                    message=f"Error in dW* for {reduction=}, {alpha=}",
                )

                if update_layer:
                    self.assertAllClose(
                        layer.optimal_delta_layer.weight,
                        w,
                        message=f"Error in the update of the delta layer for {reduction=}, {alpha=}",
                    )
                else:
                    self.assertIsNone(
                        layer.optimal_delta_layer,
                    )

                factors = {
                    "mixed": 1,
                    "mean": self.c,  # batch size to compensate the batch normalization
                    "sum": self.c * nb_batch,  # number of samples
                }
                # <dW*, dL/dA>
                self.assertAlmostEqual(
                    fo.item(),
                    4 * alpha**2 / batch_red**2 * factors[reduction],
                    places=3,
                    msg=f"Error in <dW*, dL/dA> for {reduction=}, {alpha=}",
                )

    def test_str(self):
        self.assertIsInstance(str(LinearGrowingModule(5, 5)), str)

    @unittest_parametrize(({"bias": True}, {"bias": False}))
    def test_extended_forward_out(self, bias):
        torch.manual_seed(0)
        # fixed layers
        l0 = torch.nn.Linear(5, 1, bias=bias, device=global_device())
        l_ext = torch.nn.Linear(5, 2, bias=bias, device=global_device())
        l_delta = torch.nn.Linear(5, 1, bias=bias, device=global_device())

        # changed layer
        layer = LinearGrowingModule(
            5, 1, use_bias=bias, name="layer1", device=global_device()
        )
        layer.weight.data.copy_(l0.weight.data)
        if bias:
            layer.bias.data.copy_(l0.bias.data)
        layer.optimal_delta_layer = l_delta
        layer.extended_output_layer = l_ext

        for gamma, gamma_next in ((0.0, 0.0), (1.0, 1.5), (5.0, 5.5)):
            layer.scaling_factor = gamma
            layer._scaling_factor_next_module[0] = gamma_next
            x = torch.randn((10, 5), device=global_device())
            self.assertAllClose(layer(x), l0(x))

            y_ext_1, y_ext_2 = layer.extended_forward(x)

            self.assertAllClose(y_ext_1, l0(x) - gamma**2 * l_delta(x))
            self.assertAllClose(y_ext_2, gamma_next * l_ext(x))

        layer.apply_change(apply_previous=False)
        y = layer(x)
        self.assertAllClose(y, l0(x) - gamma**2 * l_delta(x))

        layer._apply_output_changes()
        y_changed = layer(x)
        y_changed_1 = y_changed[:, :1]
        y_changed_2 = y_changed[:, 1:]
        self.assertAllClose(y_changed_1, l0(x) - gamma**2 * l_delta(x))
        self.assertAllClose(
            y_changed_2,
            gamma_next * l_ext(x),
            atol=1e-7,
            message=f"Error in applying change",
        )

    @unittest_parametrize(({"bias": True}, {"bias": False}))
    def test_extended_forward_in(self, bias):
        torch.manual_seed(0)
        # fixed layers
        l0 = torch.nn.Linear(3, 1, bias=bias, device=global_device())
        l_ext = torch.nn.Linear(5, 1, bias=bias, device=global_device())
        if bias:
            l_ext.bias.data.fill_(0)
        l_delta = torch.nn.Linear(3, 1, bias=bias, device=global_device())

        # changed layer
        layer = LinearGrowingModule(
            3, 1, use_bias=bias, name="layer1", device=global_device()
        )
        layer.weight.data.copy_(l0.weight.data)
        if bias:
            layer.bias.data.copy_(l0.bias.data)
        layer.optimal_delta_layer = l_delta
        layer.extended_input_layer = l_ext

        for gamma in (0.0, 1.0, 5.0):
            layer.zero_grad()
            layer.scaling_factor = gamma
            x = torch.randn((10, 3), device=global_device())
            x_ext = torch.randn((10, 5), device=global_device())
            self.assertAllClose(layer(x), l0(x))

            y, none = layer.extended_forward(x, x_ext)
            self.assertIsNone(none)

            self.assertAllClose(y, l0(x) - gamma**2 * l_delta(x) + gamma * l_ext(x_ext))

            torch.norm(y).backward()

            self.assertIsNotNone(layer.scaling_factor.grad)

        layer.apply_change(apply_previous=False)
        x_cat = torch.concatenate((x, x_ext), dim=1)
        y = layer(x_cat)
        self.assertAllClose(
            y,
            l0(x) - gamma**2 * l_delta(x) + gamma * l_ext(x_ext),
            message=(f"Error in applying change"),
        )

    def test_number_of_parameters(self):
        for in_layer in (1, 3):
            for out_layer in (1, 3):
                for bias in (True, False):
                    layer = LinearGrowingModule(
                        in_layer, out_layer, use_bias=bias, name="layer1"
                    )
                    self.assertEqual(
                        layer.number_of_parameters(),
                        in_layer * out_layer + bias * out_layer,
                    )

    def test_layer_in_extension(self):
        layer = LinearGrowingModule(3, 1, use_bias=False, name="layer1")
        layer.weight = torch.nn.Parameter(torch.ones(1, 3))
        self.assertEqual(layer.number_of_parameters(), 3)
        self.assertEqual(layer.in_features, 3)

        x = torch.tensor([[1, 2, 3]], dtype=torch.float32)
        y = layer(x)
        self.assertAllClose(y, torch.tensor([[6.0]]))

        layer.layer_in_extension(torch.tensor([[10]], dtype=torch.float32))
        self.assertEqual(layer.number_of_parameters(), 4)
        self.assertEqual(layer.in_features, 4)
        self.assertEqual(layer.layer.in_features, 4)
        x = torch.tensor([[1, 2, 3, 4]], dtype=torch.float32)
        y = layer(x)
        self.assertAllClose(y, torch.tensor([[46.0]]))

    def test_layer_out_extension(self):
        # without bias
        layer = LinearGrowingModule(1, 3, use_bias=False, name="layer1")
        layer.weight = torch.nn.Parameter(torch.ones(3, 1))
        self.assertEqual(layer.number_of_parameters(), 3)
        self.assertEqual(layer.out_features, 3)
        x = torch.tensor([[1]], dtype=torch.float32)
        y = layer(x)
        self.assertAllClose(y, torch.tensor([[1.0, 1.0, 1.0]]))

        layer.layer_out_extension(torch.tensor([[10]], dtype=torch.float32))
        self.assertEqual(layer.number_of_parameters(), 4)
        self.assertEqual(layer.out_features, 4)
        self.assertEqual(layer.layer.out_features, 4)

        y = layer(x)
        self.assertAllClose(y, torch.tensor([[1.0, 1.0, 1.0, 10.0]]))

        # with bias
        layer = LinearGrowingModule(1, 3, use_bias=True, name="layer1")
        layer.weight = torch.nn.Parameter(torch.ones(3, 1))
        layer.bias = torch.nn.Parameter(10 * torch.ones(3))
        self.assertEqual(layer.number_of_parameters(), 6)
        self.assertEqual(layer.out_features, 3)
        x = torch.tensor([[-1]], dtype=torch.float32)
        y = layer(x)
        self.assertAllClose(y, torch.tensor([[9.0, 9.0, 9.0]]))

        layer.layer_out_extension(
            torch.tensor([[10]], dtype=torch.float32),
            bias=torch.tensor([100], dtype=torch.float32),
        )
        self.assertEqual(layer.number_of_parameters(), 8)
        self.assertEqual(layer.out_features, 4)
        y = layer(x)
        self.assertAllClose(y, torch.tensor([[9.0, 9.0, 9.0, 90.0]]))

    def test_apply_change_delta_layer(self):
        torch.manual_seed(0)
        for bias in {True, False}:
            l0 = torch.nn.Linear(3, 1, bias=bias, device=global_device())
            l_delta = torch.nn.Linear(3, 1, bias=bias, device=global_device())
            layer = LinearGrowingModule(
                3, 1, use_bias=bias, name="layer1", device=global_device()
            )
            layer.weight.data.copy_(l0.weight.data)
            layer.optimal_delta_layer = l_delta

            if bias:
                layer.bias.data.copy_(l0.bias.data)

            gamma = 5.0
            layer.scaling_factor = gamma
            layer.apply_change(apply_previous=False)

            x = torch.randn((10, 3), device=global_device())
            y = layer(x)
            self.assertAllClose(y, l0(x) - gamma**2 * l_delta(x))

    def test_apply_change_out_extension(self):
        torch.manual_seed(0)
        for bias in {True, False}:
            l0 = torch.nn.Linear(5, 1, bias=bias, device=global_device())
            l_ext = torch.nn.Linear(5, 2, bias=bias, device=global_device())
            layer = LinearGrowingModule(
                5, 1, use_bias=bias, name="layer1", device=global_device()
            )
            layer.weight.data.copy_(l0.weight.data)

            if bias:
                layer.bias.data.copy_(l0.bias.data)
            layer.extended_output_layer = l_ext

            gamma = 5.0
            gamma_next = 5.5
            layer.scaling_factor = gamma
            layer.apply_change(apply_previous=False)
            self.assertAllClose(layer.weight.data, l0.weight.data)

            layer._scaling_factor_next_module[0] = gamma_next
            layer._apply_output_changes()

            x = torch.randn((10, 5), device=global_device())
            y = layer(x)
            y1 = y[:, :1]
            y2 = y[:, 1:]
            self.assertAllClose(y1, l0(x))
            self.assertAllClose(y2, gamma_next * l_ext(x))

    def test_apply_change_in_extension(self):
        torch.manual_seed(0)
        for bias in {True, False}:
            l0 = torch.nn.Linear(3, 1, bias=bias, device=global_device())
            l_ext = torch.nn.Linear(5, 1, bias=bias, device=global_device())
            if bias:
                l_ext.bias.data.fill_(0)
            layer = LinearGrowingModule(
                3, 1, use_bias=bias, name="layer1", device=global_device()
            )
            layer.weight.data.copy_(l0.weight.data)

            if bias:
                layer.bias.data.copy_(l0.bias.data)
            layer.extended_input_layer = l_ext

            gamma = 5.0
            layer.scaling_factor = gamma
            layer.apply_change(apply_previous=False)

            x_cat = torch.randn((10, 8), device=global_device())
            y = layer(x_cat)
            x = x_cat[:, :3]
            x_ext = x_cat[:, 3:]

            self.assertAllClose(
                y,
                l0(x) + gamma * l_ext(x_ext),
                atol=1e-7,
                message=(
                    f"Error in applying change: "
                    f"{(y - l0(x) - gamma * l_ext(x_ext)).abs().max():.2e}"
                ),
            )

    def test_sub_select_optimal_added_parameters_out(self):
        for bias in {True, False}:
            layer = LinearGrowingModule(3, 1, use_bias=bias, name="layer1")
            layer.extended_output_layer = torch.nn.Linear(3, 2, bias=bias)

            new_layer = torch.nn.Linear(3, 1, bias=bias)
            new_layer.weight.data = layer.extended_output_layer.weight.data[0].view(1, -1)
            if bias:
                new_layer.bias.data = layer.extended_output_layer.bias.data[0].view(1)

            layer._sub_select_added_output_dimension(1)

            self.assertAllClose(layer.extended_output_layer.weight, new_layer.weight)

            self.assertAllClose(layer.extended_output_layer.weight, new_layer.weight)

            if bias:
                self.assertAllClose(layer.extended_output_layer.bias, new_layer.bias)

    def test_sub_select_optimal_added_parameters_in(self):
        bias = False
        layer = LinearGrowingModule(1, 3, use_bias=bias, name="layer1")
        layer.extended_input_layer = torch.nn.Linear(2, 3, bias=bias)
        layer.eigenvalues_extension = torch.tensor([2.0, 1.0])

        new_layer = torch.nn.Linear(1, 3, bias=bias)
        new_layer.weight.data = layer.extended_input_layer.weight.data[:, 0].view(-1, 1)
        if bias:
            new_layer.bias.data = layer.extended_input_layer.bias.data

        layer.sub_select_optimal_added_parameters(1, sub_select_previous=False)

        self.assertAllClose(layer.extended_input_layer.weight, new_layer.weight)

        if bias:
            self.assertAllClose(layer.extended_input_layer.bias, new_layer.bias)

        self.assertAllClose(layer.eigenvalues_extension, torch.tensor([2.0]))

    def test_sample_number_invariant(self):
        invariants = [
            "tensor_s",
            "tensor_m",
            # "pre_activity",
            # "input",
            "delta_raw",
            "optimal_delta_layer",
            "parameter_update_decrease",
            "eigenvalues_extension",
            "tensor_m_prev",
            "cross_covariance",
        ]

        def linear_layer_equality(layer1, layer2, rtol=1e-5, atol=1e-8):
            return torch.allclose(
                layer1.weight, layer2.weight, atol=atol, rtol=rtol
            ) and (
                (layer1.bias is None and layer2.bias is None)
                or (torch.allclose(layer1.bias, layer2.bias, atol=atol, rtol=rtol))
            )

        def set_invariants(layer: LinearGrowingModule):
            _reference = dict()
            for inv in invariants:
                inv_value = getattr(layer, inv)
                if isinstance(inv_value, torch.Tensor):
                    _reference[inv] = inv_value.clone()
                elif isinstance(inv_value, torch.nn.Linear):
                    _reference[inv] = deepcopy(inv_value)
                elif isinstance(inv_value, TensorStatistic):
                    _reference[inv] = inv_value().clone()
                else:
                    raise ValueError(f"Invalid type for {inv} ({type(inv_value)})")
            return _reference

        def check_invariants(
            layer: LinearGrowingModule, reference: dict, rtol=1e-5, atol=1e-8
        ):
            for inv in invariants:
                new_inv_value = getattr(layer, inv)
                if isinstance(new_inv_value, torch.Tensor):
                    self.assertAllClose(
                        reference[inv],
                        new_inv_value,
                        rtol=rtol,
                        atol=atol,
                        message=f"Error on {inv=}",
                    )
                elif isinstance(new_inv_value, torch.nn.Linear):
                    self.assertTrue(
                        linear_layer_equality(
                            reference[inv], new_inv_value, rtol=rtol, atol=atol
                        ),
                        f"Error on {inv=}",
                    )
                elif isinstance(new_inv_value, TensorStatistic):
                    self.assertAllClose(
                        reference[inv],
                        new_inv_value(),
                        rtol=rtol,
                        atol=atol,
                        message=f"Error on {inv=}",
                    )
                else:
                    raise ValueError(f"Invalid type for {inv} ({type(new_inv_value)})")

        torch.manual_seed(0)
        layer_in = LinearGrowingModule(
            in_features=5,
            out_features=3,
            name="layer_in",
            post_layer_function=torch.nn.SELU(),
        )
        layer_out = LinearGrowingModule(
            in_features=3, out_features=7, name="layer_out", previous_module=layer_in
        )
        net = torch.nn.Sequential(layer_in, layer_out)

        def update_computation(double_batch=False):
            loss = torch.nn.MSELoss(reduction="sum")
            # loss = lambda x, y: torch.norm(x - y) ** 2
            torch.manual_seed(0)
            net.zero_grad()
            x = torch.randn((10, 5), device=global_device())
            if double_batch:
                x = torch.cat((x, x), dim=0)
            y = net(x)
            loss = loss(y, torch.zeros_like(y))
            loss.backward()
            layer_out.update_computation()

        layer_out.init_computation()

        update_computation()
        layer_out.compute_optimal_updates()

        reference = set_invariants(layer_out)

        for db in (False, True):
            update_computation(double_batch=db)
            layer_out.compute_optimal_updates()
            check_invariants(layer_out, reference)

        # simple test update without natural gradient
        layer_out.compute_optimal_updates(zero_delta=True)

    @unittest_parametrize(({"bias": True, "dtype": torch.float64}, {"bias": False}))
    def test_compute_optimal_added_parameters(
        self, bias: bool, dtype: torch.dtype = torch.float32
    ):
        demo_layers = self.demo_layers[bias]
        demo_layers[0].store_input = True
        demo_layers[1].init_computation()

        y = demo_layers[0](self.input_x)
        y = demo_layers[1](y)
        loss = torch.norm(y)
        loss.backward()

        demo_layers[1].update_computation()

        demo_layers[1].compute_optimal_delta()
        alpha, alpha_b, omega, eigenvalues = demo_layers[
            1
        ].compute_optimal_added_parameters(dtype=dtype)

        self.assertShapeEqual(
            alpha,
            (-1, demo_layers[0].in_features),
        )
        k = alpha.size(0)
        if bias:
            self.assertShapeEqual(alpha_b, (k,))
        else:
            self.assertIsNone(alpha_b)

        self.assertShapeEqual(
            omega,
            (
                demo_layers[1].out_features,
                k,
            ),
        )

        self.assertShapeEqual(eigenvalues, (k,))

        self.assertIsInstance(demo_layers[0].extended_output_layer, torch.nn.Linear)
        self.assertIsInstance(demo_layers[1].extended_input_layer, torch.nn.Linear)

        # those tests are not working yet
        demo_layers[1].sub_select_optimal_added_parameters(2)
        self.assertEqual(demo_layers[1].eigenvalues_extension.shape[0], 2)
        self.assertEqual(demo_layers[1].extended_input_layer.in_features, 2)
        self.assertEqual(demo_layers[0].extended_output_layer.out_features, 2)

    @unittest_parametrize(({"bias": True}, {"bias": False}))
    def test_tensor_s_growth(self, bias):
        demo_layers = self.demo_layers[bias]
        demo_layers[0].store_input = True
        demo_layers[1].init_computation()

        y = demo_layers[0](self.input_x)
        y = demo_layers[1](y)
        loss = torch.norm(y)
        loss.backward()

        demo_layers[1].update_computation()

        self.assertEqual(
            demo_layers[1].tensor_s_growth.samples,
            self.input_x.size(0),
        )
        s = demo_layers[0].in_features + demo_layers[0].use_bias
        self.assertShapeEqual(demo_layers[1].tensor_s_growth(), (s, s))

    def test_tensor_s_growth_errors(self):
        with self.assertRaises(AttributeError):
            self.demo_layers[True][1].tensor_s_growth = 1

        with self.assertRaises(ValueError):
            _ = self.demo_layers[True][0].tensor_s_growth


class TestLinearMergeGrowingModule(TorchTestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.demo_modules = dict()
        for bias in (True, False):
            demo_merge = LinearMergeGrowingModule(
                in_features=3, name="merge", device=global_device()
            )
            demo_merge_prev = LinearGrowingModule(
                5,
                3,
                use_bias=bias,
                name="merge_prev",
                device=global_device(),
                next_module=demo_merge,
            )
            demo_merge_next = LinearGrowingModule(
                3,
                7,
                use_bias=bias,
                name="merge_next",
                device=global_device(),
                previous_module=demo_merge,
            )
            demo_merge.set_previous_modules([demo_merge_prev])
            demo_merge.set_next_modules([demo_merge_next])
            self.demo_modules[bias] = {
                "add": demo_merge,
                "prev": demo_merge_prev,
                "next": demo_merge_next,
                "seq": torch.nn.Sequential(demo_merge_prev, demo_merge, demo_merge_next),
            }
        self.input_x = torch.randn((11, 5), device=global_device())

    @unittest_parametrize(({"bias": True}, {"bias": False}))
    def test_init(self, bias):
        self.assertIsInstance(self.demo_modules[bias]["add"], LinearMergeGrowingModule)

    @unittest_parametrize(({"bias": True}, {"bias": False}))
    def test_input_storage(self, bias):
        demo_layers = self.demo_modules[bias]
        demo_layers["next"].store_input = True
        self.assertEqual(demo_layers["add"].store_activity, 1)
        self.assertTrue(not demo_layers["next"]._internal_store_input)
        self.assertIsNone(demo_layers["next"].input)

        _ = demo_layers["seq"](self.input_x)

        self.assertShapeEqual(
            demo_layers["next"].input,
            (self.input_x.size(0), demo_layers["next"].in_features),
        )

    @unittest_parametrize(({"bias": True}, {"bias": False}))
    def test_activity_storage(self, bias):
        demo_layers = self.demo_modules[bias]
        demo_layers["prev"].store_pre_activity = True
        self.assertEqual(demo_layers["add"].store_input, 1)
        self.assertTrue(not demo_layers["prev"]._internal_store_pre_activity)
        self.assertIsNone(demo_layers["prev"].pre_activity)

        _ = demo_layers["seq"](self.input_x)

        self.assertShapeEqual(
            demo_layers["prev"].pre_activity,
            (self.input_x.size(0), demo_layers["prev"].out_features),
        )

    def test_update_scaling_factor(self):
        demo_layers = self.demo_modules[True]

        demo_layers["add"].update_scaling_factor(scaling_factor=0.5)
        self.assertEqual(demo_layers["prev"]._scaling_factor_next_module.item(), 0.5)
        self.assertEqual(demo_layers["prev"].scaling_factor.item(), 0.0)
        self.assertEqual(demo_layers["next"].scaling_factor.item(), 0.5)

    def test_update_scaling_factor_incorrect_input_module(self):
        demo_layers = self.demo_modules[True]
        demo_layers["add"].previous_modules = [demo_layers["prev"], torch.nn.Linear(7, 3)]
        with self.assertRaises(TypeError):
            demo_layers["add"].update_scaling_factor(scaling_factor=0.5)

    def test_update_scaling_factor_incorrect_output_module(self):
        demo_layers = self.demo_modules[True]
        demo_layers["add"].set_next_modules([demo_layers["next"], torch.nn.Linear(3, 7)])
        with self.assertRaises(TypeError):
            demo_layers["add"].update_scaling_factor(scaling_factor=0.5)

    @unittest_parametrize(({"bias": True}, {"bias": False}))
    def test_set_previous_next_modules(self, bias):
        demo_layers = self.demo_modules[bias]
        new_input_layer = LinearGrowingModule(
            2,
            3,
            use_bias=bias,
            name="new_prev",
            device=global_device(),
            next_module=demo_layers["add"],
        )
        new_output_layer = LinearGrowingModule(
            3,
            2,
            use_bias=bias,
            name="new_next",
            device=global_device(),
            previous_module=demo_layers["add"],
        )

        self.assertEqual(
            demo_layers["add"].sum_in_features(), demo_layers["prev"].in_features
        )
        self.assertEqual(
            demo_layers["add"].sum_in_features(with_bias=True),
            demo_layers["prev"].in_features + bias,
        )
        self.assertEqual(
            demo_layers["add"].sum_out_features(), demo_layers["next"].out_features
        )

        demo_layers["add"].set_previous_modules([demo_layers["prev"], new_input_layer])
        demo_layers["add"].set_next_modules([demo_layers["next"], new_output_layer])

        self.assertEqual(
            demo_layers["add"].sum_in_features(),
            demo_layers["prev"].in_features + new_input_layer.in_features,
        )

        self.assertEqual(
            demo_layers["add"].sum_in_features(with_bias=True),
            demo_layers["prev"].in_features + bias + new_input_layer.in_features + bias,
        )

        self.assertEqual(
            demo_layers["add"].sum_out_features(),
            demo_layers["next"].out_features + new_output_layer.out_features,
        )

    # PHASE 1 - CRITICAL COVERAGE IMPROVEMENTS
    # Adding comprehensive tests for compute_optimal_delta (0% coverage -> +15% gain)
    
    @unittest_parametrize(({"bias": True}, {"bias": False}))
    def test_compute_optimal_delta_basic_functionality(self, bias):
        """Test basic compute_optimal_delta functionality."""
        demo_layers = self.demo_modules[bias]
        merge_module = demo_layers["add"]
        
        # Initialize tensor statistics computation
        merge_module.init_computation()
        
        # Ensure modules are properly set up with data - multiple passes with gradients
        for _ in range(3):
            demo_layers["seq"].zero_grad()
            output = demo_layers["seq"](self.input_x)
            loss = torch.norm(output)
            loss.backward()
            
            # CRITICAL: Manually update tensor statistics after forward/backward pass
            merge_module.update_computation()
        
        # Test basic compute_optimal_delta call
        result = merge_module.compute_optimal_delta()
        
        # Should return None by default (no return_deltas)
        self.assertIsNone(result)
        
        # Verify that internal computations occurred (tensor updates)
        self.assertIsNotNone(merge_module.previous_tensor_s)
        self.assertIsNotNone(merge_module.previous_tensor_m)
        self.assertGreater(merge_module.previous_tensor_s.samples, 0)
        self.assertGreater(merge_module.previous_tensor_m.samples, 0)

    @unittest_parametrize(({"bias": True}, {"bias": False}))
    def test_compute_optimal_delta_with_return_deltas(self, bias):
        """Test compute_optimal_delta with return_deltas=True."""
        demo_layers = self.demo_modules[bias]
        merge_module = demo_layers["add"]
        
        # Initialize tensor statistics computation
        merge_module.init_computation()

        # Ensure modules are properly set up with data - multiple passes with gradients
        for _ in range(3):
            demo_layers["seq"].zero_grad()
            output = demo_layers["seq"](self.input_x)
            loss = torch.norm(output)
            loss.backward()
            
            # CRITICAL: Manually update tensor statistics after forward/backward pass  
            merge_module.update_computation()
        
        # Test with return_deltas=True
        deltas = merge_module.compute_optimal_delta(return_deltas=True)
        
        # Should return list of tuples (delta_w, delta_b)
        self.assertIsInstance(deltas, list)
        self.assertEqual(len(deltas), len(merge_module.previous_modules))
        
        # Each delta should be a tuple (weight_delta, bias_delta)
        for i, (delta_w, delta_b) in enumerate(deltas):
            prev_module = merge_module.previous_modules[i]
            expected_weight_shape = (prev_module.out_features, prev_module.in_features)
            self.assertEqual(delta_w.shape, expected_weight_shape)
            self.assertIsInstance(delta_w, torch.Tensor)
            
            # Check bias delta based on module configuration
            if prev_module.use_bias:
                self.assertIsNotNone(delta_b)
                expected_bias_shape = (prev_module.out_features,)
                self.assertEqual(delta_b.shape, expected_bias_shape)
            else:
                self.assertIsNone(delta_b)

    @unittest_parametrize(({"bias": True}, {"bias": False}))
    def test_compute_optimal_delta_pseudo_inverse_fallback(self, bias):
        """Test compute_optimal_delta with pseudo-inverse fallback."""
        demo_layers = self.demo_modules[bias]
        merge_module = demo_layers["add"]
        
        # Initialize tensor statistics computation
        merge_module.init_computation()
        
        # Ensure modules are properly set up with data
        for _ in range(3):
            demo_layers["seq"].zero_grad()
            output = demo_layers["seq"](self.input_x)
            loss = torch.norm(output)
            loss.backward()
            
            # CRITICAL: Manually update tensor statistics after forward/backward pass
            merge_module.update_computation()
        
        # Test pseudo-inverse by forcing it
        deltas = merge_module.compute_optimal_delta(
            return_deltas=True, force_pseudo_inverse=True
        )
        
        # Should still return valid deltas
        self.assertIsInstance(deltas, list)
        self.assertEqual(len(deltas), len(merge_module.previous_modules))
        
        # Verify all deltas have correct shapes
        for i, (delta_w, delta_b) in enumerate(deltas):
            prev_module = merge_module.previous_modules[i]
            expected_weight_shape = (prev_module.out_features, prev_module.in_features)
            self.assertEqual(delta_w.shape, expected_weight_shape)

    @unittest_parametrize(({"bias": True}, {"bias": False}))
    def test_compute_optimal_delta_different_bias_configs(self, bias):
        """Test compute_optimal_delta with different bias configurations."""
        # Use the existing demo modules which have a working single-input setup
        demo_layers = self.demo_modules[bias]
        merge_module = demo_layers["add"]
        
        # Initialize tensor statistics computation
        merge_module.init_computation()
        
        # Ensure modules are properly set up with data - multiple passes with gradients
        for _ in range(3):
            demo_layers["seq"].zero_grad()
            output = demo_layers["seq"](self.input_x)
            loss = torch.norm(output)
            loss.backward()
            
            # CRITICAL: Manually update tensor statistics after forward/backward pass
            merge_module.update_computation()
        
        # Test compute_optimal_delta with the bias configuration
        deltas = merge_module.compute_optimal_delta(return_deltas=True)
        
        # Should handle bias configurations correctly
        self.assertIsNotNone(deltas)
        self.assertIsInstance(deltas, list)
        assert deltas is not None  # Type narrowing for mypy
        self.assertEqual(len(deltas), len(merge_module.previous_modules))
        
        # Check delta shapes account for bias differences
        for i, (delta_w, delta_b) in enumerate(deltas):
            prev_module = merge_module.previous_modules[i]
            expected_weight_shape = (prev_module.out_features, prev_module.in_features)
            self.assertEqual(delta_w.shape, expected_weight_shape)
            
            # Check bias handling
            if prev_module.use_bias:
                self.assertIsNotNone(delta_b)
                expected_bias_shape = (prev_module.out_features,)
                self.assertEqual(delta_b.shape, expected_bias_shape)
            else:
                self.assertIsNone(delta_b)

    def test_compute_optimal_delta_error_conditions(self):
        """Test error conditions in compute_optimal_delta."""
        # Test with uninitialized merge module
        merge_module = LinearMergeGrowingModule(in_features=3, device=global_device())
        
        # Should handle case with no previous modules gracefully
        with self.assertRaises(AssertionError):
            merge_module.compute_optimal_delta()
        
        # Test with improperly configured modules
        prev_module = LinearGrowingModule(2, 3, device=global_device())
        merge_module.set_previous_modules([prev_module])
        
        # Should handle case with no tensor data
        with self.assertRaises(ValueError):
            merge_module.compute_optimal_delta()

    # PHASE 2 - TARGET 95% COVERAGE: MAJOR MISSING FUNCTIONALITY
    # Adding comprehensive tests for add_parameters method (lines 725-767)
    
    @unittest_parametrize(({"bias": True}, {"bias": False}))
    def test_add_parameters_input_features(self, bias):
        """Test add_parameters method for adding input features (lines 725-767)."""
        layer = LinearGrowingModule(3, 2, use_bias=bias, device=global_device())
        original_in_features = layer.in_features
        original_out_features = layer.out_features
        
        # Test adding input features with default zero matrix
        added_in_features = 2
        
        # This should trigger lines 728-736 (added_in_features > 0 branch)
        layer.add_parameters(
            matrix_extension=None,
            bias_extension=None,
            added_in_features=added_in_features,
            added_out_features=0
        )
        
        # Verify layer dimensions changed correctly
        self.assertEqual(layer.in_features, original_in_features + added_in_features)
        self.assertEqual(layer.out_features, original_out_features)
        self.assertEqual(layer.layer.in_features, original_in_features + added_in_features)
        
        # Test input with extended features
        x = torch.randn(5, original_in_features + added_in_features, device=global_device())
        output = layer(x)
        self.assertEqual(output.shape, (5, original_out_features))

    @unittest_parametrize(({"bias": True}, {"bias": False}))
    def test_add_parameters_input_features_with_custom_matrix(self, bias):
        """Test add_parameters with custom matrix extension for input features."""
        layer = LinearGrowingModule(3, 2, use_bias=bias, device=global_device())
        original_in_features = layer.in_features
        original_out_features = layer.out_features
        
        # Test adding input features with custom matrix
        added_in_features = 2
        custom_matrix = torch.ones(original_out_features, added_in_features, device=global_device())
        
        # This should trigger lines 737-744 (custom matrix_extension branch)
        layer.add_parameters(
            matrix_extension=custom_matrix,
            bias_extension=None,
            added_in_features=added_in_features,
            added_out_features=0
        )
        
        # Verify layer dimensions
        self.assertEqual(layer.in_features, original_in_features + added_in_features)
        self.assertEqual(layer.out_features, original_out_features)
        
        # Test that custom matrix was used (check weight matrix contains ones)
        x = torch.zeros(1, original_in_features + added_in_features, device=global_device())
        x[0, original_in_features:] = 1.0  # Set extended features to 1
        output = layer(x)
        # Extended features should contribute due to ones in custom matrix
        self.assertGreater(torch.abs(output).sum().item(), 0)

    @unittest_parametrize(({"bias": True}, {"bias": False}))
    def test_add_parameters_output_features(self, bias):
        """Test add_parameters method for adding output features."""
        layer = LinearGrowingModule(3, 2, use_bias=bias, device=global_device())
        original_in_features = layer.in_features
        original_out_features = layer.out_features
        
        # Test adding output features with default matrices
        added_out_features = 2
        
        # This should trigger lines 746-767 (added_out_features > 0 branch)
        layer.add_parameters(
            matrix_extension=None,
            bias_extension=None,
            added_in_features=0,
            added_out_features=added_out_features
        )
        
        # Verify layer dimensions changed correctly
        self.assertEqual(layer.in_features, original_in_features)
        self.assertEqual(layer.out_features, original_out_features + added_out_features)
        self.assertEqual(layer.layer.out_features, original_out_features + added_out_features)
        
        # Test output with extended features
        x = torch.randn(5, original_in_features, device=global_device())
        output = layer(x)
        self.assertEqual(output.shape, (5, original_out_features + added_out_features))

    @unittest_parametrize(({"bias": True}, {"bias": False}))
    def test_add_parameters_output_features_with_custom_matrices(self, bias):
        """Test add_parameters with custom matrices for output features."""
        layer = LinearGrowingModule(3, 2, use_bias=bias, device=global_device())
        original_in_features = layer.in_features
        original_out_features = layer.out_features
        
        # Test adding output features with custom matrices
        added_out_features = 2
        custom_weight = torch.ones(added_out_features, original_in_features, device=global_device())
        custom_bias = torch.ones(added_out_features, device=global_device()) * 5.0 if bias else None
        
        # This should trigger lines 750-766 (custom matrix/bias extension branches)
        layer.add_parameters(
            matrix_extension=custom_weight,
            bias_extension=custom_bias,
            added_in_features=0,
            added_out_features=added_out_features
        )
        
        # Verify layer dimensions
        self.assertEqual(layer.in_features, original_in_features)
        self.assertEqual(layer.out_features, original_out_features + added_out_features)
        
        # Test that custom matrices were used
        x = torch.ones(1, original_in_features, device=global_device())
        output = layer(x)
        
        # Extended outputs should be influenced by custom weight (all 1s) and bias if present
        extended_outputs = output[0, original_out_features:]
        if bias:
            expected_value = original_in_features + 5.0  # sum of ones * inputs + bias
            self.assertAllClose(extended_outputs, torch.full_like(extended_outputs, expected_value))
        else:
            expected_value = original_in_features  # sum of ones * inputs, no bias
            self.assertAllClose(extended_outputs, torch.full_like(extended_outputs, expected_value))

    def test_add_parameters_assertion_errors(self):
        """Test assertion errors in add_parameters method."""
        layer = LinearGrowingModule(3, 2, device=global_device())
        
        # Test adding both input and output features (should raise AssertionError)
        with self.assertRaises(AssertionError) as context:
            layer.add_parameters(
                matrix_extension=None,
                bias_extension=None,
                added_in_features=1,
                added_out_features=1
            )
        self.assertIn("cannot add input and output features at the same time", str(context.exception))
        
        # Test wrong matrix shape for input extension
        with self.assertRaises(AssertionError) as context:
            wrong_matrix = torch.ones(3, 3)  # Should be (2, 2) for 2 added input features
            layer.add_parameters(
                matrix_extension=wrong_matrix,
                bias_extension=None,
                added_in_features=2,
                added_out_features=0
            )
        self.assertIn("matrix_extension should have shape", str(context.exception))
        
        # Test wrong matrix shape for output extension
        layer2 = LinearGrowingModule(3, 2, device=global_device())
        with self.assertRaises(AssertionError) as context:
            wrong_matrix = torch.ones(3, 2)  # Should be (2, 3) for 2 added output features
            layer2.add_parameters(
                matrix_extension=wrong_matrix,
                bias_extension=None,
                added_in_features=0,
                added_out_features=2
            )
        self.assertIn("matrix_extension should have shape", str(context.exception))
        
        # Test wrong bias shape for output extension
        layer3 = LinearGrowingModule(3, 2, device=global_device())
        with self.assertRaises(AssertionError) as context:
            correct_matrix = torch.ones(2, 3)
            wrong_bias = torch.ones(3)  # Should be (2,) for 2 added output features
            layer3.add_parameters(
                matrix_extension=correct_matrix,
                bias_extension=wrong_bias,
                added_in_features=0,
                added_out_features=2
            )
        self.assertIn("bias_extension should have shape", str(context.exception))

    # PHASE 2.2 - COVERING compute_n_update METHOD (lines 579-589)
    # Testing the fixed compute_n_update method with corrected projected_v_goal() call
    
    @unittest_parametrize(({"bias": True}, {"bias": False}))
    def test_compute_n_update_basic_functionality(self, bias):
        """Test compute_n_update method basic functionality (lines 579-589)."""
        # Create a chain of LinearGrowingModules: layer1 -> layer2
        layer1 = LinearGrowingModule(3, 2, use_bias=bias, device=global_device(), name="layer1")
        layer2 = LinearGrowingModule(2, 4, use_bias=bias, device=global_device(), name="layer2")
        layer1.next_module = layer2
        layer2.previous_module = layer1
        
        # Initialize computation for both layers
        layer1.init_computation()
        layer2.init_computation()
        
        # Create sequential network
        net = torch.nn.Sequential(layer1, layer2)
        
        # Forward pass with input data
        x = torch.randn(5, 3, device=global_device())
        output = net(x)
        
        # Create a loss and backward pass to generate gradients
        loss = torch.norm(output)
        loss.backward()
        
        # Update tensor statistics
        layer1.update_computation()
        layer2.update_computation()
        
        # Compute optimal delta for layer2 (required for projected_v_goal)
        layer2.compute_optimal_delta()
        
        # Test compute_n_update on layer1 (which has layer2 as next_module)
        n_update, n_samples = layer1.compute_n_update()
        
        # Verify shapes and values - the shape is (in_features, out_features) without bias
        expected_shape = (layer1.in_features, layer2.out_features)
        self.assertEqual(n_update.shape, expected_shape)
        self.assertEqual(n_samples, x.shape[0])  # Should equal batch size
        self.assertIsInstance(n_update, torch.Tensor)
        self.assertIsInstance(n_samples, int)

    @unittest_parametrize(({"bias": True}, {"bias": False}))
    def test_compute_n_update_with_different_shapes(self, bias):
        """Test compute_n_update with different layer dimensions."""
        # Test with different layer sizes
        layer1 = LinearGrowingModule(4, 3, use_bias=bias, device=global_device(), name="layer1")
        layer2 = LinearGrowingModule(3, 5, use_bias=bias, device=global_device(), name="layer2")
        layer1.next_module = layer2
        layer2.previous_module = layer1
        
        # Initialize and setup
        layer1.init_computation()
        layer2.init_computation()
        net = torch.nn.Sequential(layer1, layer2)
        
        # Multiple batch sizes to test tensor flattening
        for batch_size in [1, 3, 7]:
            net.zero_grad()
            x = torch.randn(batch_size, 4, device=global_device())
            output = net(x)
            loss = torch.norm(output)
            loss.backward()
            
            layer1.update_computation()
            layer2.update_computation()
            
            # Compute optimal delta for layer2 (required for projected_v_goal)
            layer2.compute_optimal_delta()
            
            n_update, n_samples = layer1.compute_n_update()
            
            # Verify correct shapes and sample counting - shape is (in_features, out_features) without bias
            expected_shape = (layer1.in_features, layer2.out_features)
            self.assertEqual(n_update.shape, expected_shape)
            self.assertEqual(n_samples, batch_size)

    def test_compute_n_update_type_error(self):
        """Test compute_n_update raises TypeError for non-LinearGrowingModule next_module."""
        layer1 = LinearGrowingModule(3, 2, device=global_device(), name="layer1")
        
        # Set next_module to a regular Linear layer (not LinearGrowingModule)
        layer1.next_module = torch.nn.Linear(2, 4)
        
        # Initialize computation
        layer1.init_computation()
        
        # Setup input and forward pass
        x = torch.randn(2, 3, device=global_device())
        output = layer1(x)
        loss = torch.norm(output)
        loss.backward()
        layer1.update_computation()
        
        # Should raise TypeError due to wrong next_module type
        with self.assertRaises(TypeError) as context:
            layer1.compute_n_update()
        
        self.assertIn("The next module must be a LinearGrowingModule", str(context.exception))

    @unittest_parametrize(({"bias": True}, {"bias": False}))
    def test_compute_n_update_tensor_computation_correctness(self, bias):
        """Test compute_n_update mathematical correctness with known values."""
        # Create layers with known dimensions for predictable testing
        layer1 = LinearGrowingModule(2, 2, use_bias=bias, device=global_device(), name="layer1")
        layer2 = LinearGrowingModule(2, 2, use_bias=bias, device=global_device(), name="layer2")
        layer1.next_module = layer2
        layer2.previous_module = layer1
        
        # Initialize computation
        layer1.init_computation()
        layer2.init_computation()
        
        # Use simple input for easier verification
        x = torch.ones(2, 2, device=global_device())  # Simple input: all ones
        
        # Forward pass
        out1 = layer1(x)
        out2 = layer2(out1)
        
        # Backward pass
        loss = torch.norm(out2)
        loss.backward()
        
        # Update computations
        layer1.update_computation()
        layer2.update_computation()
        
        # Compute optimal delta for layer2 (required for projected_v_goal)
        layer2.compute_optimal_delta()
        
        # Test compute_n_update
        n_update, n_samples = layer1.compute_n_update()
        
        # Verify that computation uses the correct einsum operation
        # The method should compute: torch.einsum("ij,ik->jk", input_flat, projected_v_goal_flat)
        input_flat = torch.flatten(layer1.input, 0, -2)
        projected_v_goal_flat = torch.flatten(layer2.projected_v_goal(layer2.input), 0, -2)
        expected_n_update = torch.einsum("ij,ik->jk", input_flat, projected_v_goal_flat)
        
        # Assert the computed n_update matches expected calculation
        self.assertAllClose(n_update, expected_n_update, atol=1e-6)
        self.assertEqual(n_samples, 2)  # batch size

    def test_compute_n_update_no_next_module(self):
        """Test behavior when next_module is None (should raise TypeError)."""
        layer = LinearGrowingModule(3, 2, device=global_device(), name="layer")
        layer.next_module = None  # No next module
        
        # This method should only be called when there's a valid next_module
        # The method raises TypeError when next_module is not a LinearGrowingModule
        x = torch.randn(2, 3, device=global_device())
        layer.init_computation()
        output = layer(x)
        loss = torch.norm(output)
        loss.backward()
        layer.update_computation()
        
        # Should raise TypeError when next_module is not a LinearGrowingModule
        with self.assertRaises(TypeError):
            layer.compute_n_update()

    def test_error_handling_edge_cases(self):
        """Test various error handling edge cases for better coverage."""
        layer = LinearGrowingModule(3, 2, device=global_device(), name="layer")
        
        # Test XOR assertion for extended layers (line 883-885)
        # Setting both extended_input_layer and extended_output_layer should fail
        layer.extended_input_layer = torch.nn.Linear(3, 2)
        layer.extended_output_layer = torch.nn.Linear(3, 2)
        
        # This should trigger assertion error because both are set (violates XOR)
        with self.assertRaises(AssertionError):
            layer._sub_select_added_output_dimension(1)

    def test_layer_initialization_edge_cases(self):
        """Test layer initialization with different bias settings (lines 85, 87)."""
        # Test various initialization scenarios
        layer1 = LinearGrowingModule(3, 2, use_bias=True, device=global_device())
        layer2 = LinearGrowingModule(3, 2, use_bias=False, device=global_device())
        
        # Test setting next_module property directly (instead of through set_next_modules)
        layer1.next_module = layer2
        
        # Test accessing properties that might trigger missing lines
        self.assertIsInstance(layer1.use_bias, bool)
        self.assertIsInstance(layer2.use_bias, bool)

    def test_activation_gradient_not_implemented(self):
        """Test activation gradient computation with unsupported previous module (lines 359-364)."""
        layer = LinearGrowingModule(4, 2, device=global_device(), name="layer")
        
        # Set an unsupported previous module type
        layer.previous_module = torch.nn.Linear(3, 4)  # Regular Linear layer, not supported
        
        # Should raise NotImplementedError
        with self.assertRaises(NotImplementedError):
            _ = layer.activation_gradient


if __name__ == "__main__":
    main()
