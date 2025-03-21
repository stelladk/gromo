"""
Module to define a two layer block similar to a BasicBlock in ResNet.
"""

import torch
import torch.nn as nn

from gromo.containers.growing_container import GrowingContainer
from gromo.modules.linear_growing_module import (
    LinearAdditionGrowingModule,
    LinearGrowingModule,
)


all_layer_types = {
    "linear": {"layer": LinearGrowingModule, "addition": LinearAdditionGrowingModule},
}


class GrowingResidualMLP(GrowingContainer):
    def __init__(
        self,
        in_features: torch.Size | tuple[int, ...],
        out_features: int,
        num_features: int,
        hidden_features: int,
        num_blocks: int,
        activation: torch.nn.Module = torch.nn.ReLU(),
        device: torch.device = None,
    ) -> None:

        in_features = torch.tensor(in_features).prod().int().item()
        super(GrowingResidualMLP, self).__init__(
            in_features=torch.tensor(in_features).prod().int().item(),
            out_features=out_features,
            device=device,
        )
        self.num_features = num_features
        self.hidden_features = hidden_features
        self.num_blocks = num_blocks

        # embedding
        # print(f"Embedding: {self.in_features} -> {self.num_features}")
        self.embedding = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(self.in_features, num_features, device=self.device),
        )

        # blocks
        self.blocks = torch.nn.ModuleList(
            [
                GrowingResidualBlock(
                    num_features,
                    hidden_features,
                    activation=activation,
                    name=f"block {i}",
                )
                for i in range(num_blocks)
            ]
        )

        # final projection
        self.projection = nn.Linear(num_features, self.out_features, device=self.device)
        self.set_growing_layers()

    def set_growing_layers(self):
        self.growing_layers = nn.ModuleList(block.second_layer for block in self.blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        for block in self.blocks:
            x = block(x)
        x = self.projection(x)
        return x

    def extended_forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        for block in self.blocks:
            x = block.extended_forward(x)
        x = self.projection(x)
        return x

    def select_update(self, layer_index: int, verbose: bool = False) -> int:
        for i, layer in enumerate(self.growing_layers):
            if verbose:
                print(f"Block {i} improvement: {layer.first_order_improvement}")
                print(
                    f"Block {i} parameter improvement: {layer.parameter_update_decrease}"
                )
                print(f"Block {i} eigenvalues extension: {layer.eigenvalues}")
            if i != layer_index:
                if verbose:
                    print(f"Deleting block {i}")
                layer.delete_update()
            else:
                self.currently_updated_layer_index = i
        return self.currently_updated_layer_index

    def number_of_parameters(self):
        num_param = sum(p.numel() for p in self.embedding.parameters())
        for block in self.blocks:
            num_param += block.number_of_parameters()
        num_param += sum(p.numel() for p in self.projection.parameters())
        return num_param
        # return sum(p.numel() for p in self.parameters())

    @staticmethod
    def tensor_statistics(tensor) -> dict[str, float]:
        min_value = tensor.min().item()
        max_value = tensor.max().item()
        mean_value = tensor.mean().item()
        if tensor.numel() > 1:
            std_value = tensor.std().item()
        else:
            std_value = -1
        return {
            "min": min_value,
            "max": max_value,
            "mean": mean_value,
            "std": std_value,
        }

    def weights_statistics(self) -> dict[int, dict[str, dict[str, float]]]:
        statistics = {}
        for i, block in enumerate(self.blocks):
            statistics[i] = {"weight_0": self.tensor_statistics(block.first_layer.weight)}
            if block.first_layer.bias is not None:
                statistics[i]["bias_0"] = self.tensor_statistics(block.first_layer.bias)

            statistics[i]["weight_1"] = self.tensor_statistics(block.second_layer.weight)
            if block.second_layer.bias is not None:
                statistics[i]["bias_1"] = self.tensor_statistics(block.second_layer.bias)

            statistics[i]["hidden_shape"] = block.hidden_features
        return statistics

    def update_information(self):
        information = dict()
        for i, layer in enumerate(self.growing_layers):
            layer_information = dict()
            layer_information["update_value"] = layer.first_order_improvement
            layer_information["parameter_improvement"] = layer.parameter_update_decrease
            layer_information["eigenvalues_extension"] = layer.eigenvalues_extension
            information[i] = layer_information
        return information


class GrowingResidualBlock(GrowingContainer):
    """
    Represents a block of a growing network.

    Sequence of layers:
    - Activation pre
    - Layer first
    - Activation mid
    - Layer second
    """

    def __init__(
        self,
        num_features: int,
        hidden_features: int = 0,
        activation: torch.nn.Module | None = None,
        name: str = "block",
        kwargs_layer: dict | None = None,
    ) -> None:
        """
        Initialise the block.

        Parameters
        ----------
        num_features: int
            number of input and output features, in cas of convolutional layer, the number of channels
        hidden_features: int
            number of hidden features, if zero the block is the zero function
        layer_type: str
            type of layer to use either "linear" or "conv"
        activation: torch.nn.Module | None
            activation function to use, if None use the identity function
        name: str
            name of the block
        kwargs_layer: dict | None
            dictionary of arguments for the layers (e.g. bias, ...)
        """
        if kwargs_layer is None:
            kwargs_layer = {}

        super(GrowingResidualBlock, self).__init__(
            in_features=num_features,
            out_features=num_features,
        )
        self.name = name
        self.num_features = num_features
        self.hidden_features = hidden_features

        self.norm = nn.LayerNorm(
            num_features, elementwise_affine=False, device=self.device
        )
        self.activation: torch.nn.Module = activation
        self.first_layer = LinearGrowingModule(
            num_features,
            hidden_features,
            post_layer_function=self.activation,
            name=f"first_layer",
            **kwargs_layer,
        )
        self.second_layer = LinearGrowingModule(
            hidden_features,
            num_features,
            post_layer_function=torch.nn.Identity(),
            previous_module=self.first_layer,
            name=f"second_layer",
            **kwargs_layer,
        )

        self.enable_extended_forward = False

        # self.activation_derivative = torch.func.grad(mid_activation)(torch.tensor(1e-5))
        # TODO: FIX this
        self.activation_derivative = 1

    def extended_forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the block with the current modifications.

        Parameters
        ----------
        x: torch.Tensor
            input tensor

        Returns
        -------
        torch.Tensor
            output tensor
        """

        if self.hidden_features > 0:
            x = self.norm(x)
            y = self.activation(x)
            y, y_ext = self.first_layer.extended_forward(y)
            y, _ = self.second_layer.extended_forward(y, y_ext)
            assert (
                _ is None
            ), f"The output of layer 2 {self.second_layer.name} should not be extended."
            del y_ext
            x = y + x
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the block.

        Parameters
        ----------
        x: torch.Tensor
            input tensor

        Returns
        -------
        torch.Tensor
            output tensor
        """
        if self.hidden_features > 0:
            x = self.norm(x)
            y = self.activation(x)
            y = self.first_layer(y)
            y = self.second_layer(y)
            x = y + x
        return x

    def sub_select_optimal_added_parameters(
        self,
        keep_neurons: int,
    ) -> None:
        """
        Select the first keep_neurons neurons of the optimal added parameters.

        Parameters
        ----------
        keep_neurons: int
            number of neurons to keep
        """
        self.currently_updated_layer.eigenvalues = self.eigenvalues[:keep_neurons]
        self.currently_updated_layer.second_layer.sub_select_optimal_added_parameters(
            keep_neurons, sub_select_previous=True
        )

    def number_of_parameters(self):
        num_param = self.first_layer.number_of_parameters()
        num_param += self.second_layer.number_of_parameters()
        return num_param

    @staticmethod
    def tensor_statistics(tensor) -> dict[str, float]:
        min_value = tensor.min().item()
        max_value = tensor.max().item()
        mean_value = tensor.mean().item()
        if tensor.numel() > 1:
            std_value = tensor.std().item()
        else:
            std_value = -1
        return {
            "min": min_value,
            "max": max_value,
            "mean": mean_value,
            "std": std_value,
        }
