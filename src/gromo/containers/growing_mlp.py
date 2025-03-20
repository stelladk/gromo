import torch
from torch import nn

from gromo.containers.growing_container import GrowingContainer
from gromo.modules.linear_growing_module import LinearGrowingModule


class GrowingMLP(GrowingContainer):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_shape: int,
        number_hidden_layers: int,
        activation: nn.Module = nn.SELU(),
        use_bias: bool = True,
        device: torch.device = None,
    ):
        super(GrowingMLP, self).__init__(
            in_features=in_features,
            out_features=out_features,
            device=device,
        )

        self.num_features = torch.tensor(self.in_features).prod().int().item()

        # flatten input
        self.flatten = nn.Flatten(start_dim=1)
        self.layers = nn.ModuleList()
        self.layers.append(
            LinearGrowingModule(
                self.num_features,
                hidden_shape,
                post_layer_function=activation,
                use_bias=use_bias,
                name="Layer 0",
            )
        )
        for i in range(number_hidden_layers - 1):
            self.layers.append(
                LinearGrowingModule(
                    hidden_shape,
                    hidden_shape,
                    post_layer_function=activation,
                    previous_module=self.layers[-1],
                    use_bias=use_bias,
                    name=f"Layer {i + 1}",
                )
            )
        self.layers.append(
            LinearGrowingModule(
                hidden_shape,
                self.out_features,
                previous_module=self.layers[-1],
                use_bias=use_bias,
                name=f"Layer {number_hidden_layers}",
            )
        )

        self.set_growing_layers()
        self.updates_values = None
        self.currently_updated_layer_index = None

    def set_growing_layers(self):
        self.growing_layers = self.layers[1:]

    def number_of_parameters(self) -> int:
        """
        Return the number of parameters of the model.

        Returns
        -------
        int
            number of parameters
        """
        return sum(layer.number_of_parameters() for layer in self.layers)

    def __str__(self):
        return "\n".join(str(layer) for layer in self.layers)

    def __repr__(self):
        return self.__str__()

    def __getitem__(self, item: int):
        assert (
            0 <= item < len(self.layers)
        ), f"{item=} should be in [0, {len(self.layers)})"
        return self.layers[item]

    @staticmethod
    def normalisation_factor(values: torch.Tensor):
        """
        Compute normalisation factor for the values in the tensor i.e.
        factors such that the product of the factors is 1 and each value
        multiplied by the factor is equal.

        Parameters
        ----------
        values: torch.Tensor of float, shape (N)
            Values to be normalised

        Returns
        -------
        torch.Tensor of float, shape (N)
            Normalisation factors
        """
        normalisation = values.prod().pow(1 / values.numel())
        return normalisation.repeat(values.shape) / values

    def normalise(self, verbose: bool = False):
        max_values = torch.zeros(len(self.layers), device=self.device)
        for i, layer in enumerate(self.layers):
            max_values[i] = layer.weight.abs().max()
        normalisation = self.normalisation_factor(max_values)
        if verbose:
            print(f"Normalisation: {list(enumerate(normalisation))}")
        current_normalisation = torch.ones(1, device=self.device)
        for i, layer in enumerate(self.layers):
            layer.weight.data = layer.weight.data * normalisation[i]
            current_normalisation *= normalisation[i]
            if layer.bias is not None:
                layer.bias.data = layer.bias.data * current_normalisation

    def forward(self, x):
        x = self.flatten(x)
        for layer in self.layers:
            x = layer(x)
        return x

    def extended_forward(self, x):
        x = self.flatten(x)
        x_ext = None
        for layer in self.layers:
            x, x_ext = layer.extended_forward(x, x_ext)
        return x

    def update_information(self):
        information = dict()
        for i, layer in enumerate(self.growing_layers):
            layer_information = dict()
            layer_information["update_value"] = layer.first_order_improvement
            layer_information["parameter_improvement"] = layer.parameter_update_decrease
            layer_information["eigenvalues_extension"] = layer.eigenvalues_extension
            information[i] = layer_information
        return information

    def select_update(self, layer_index: int, verbose: bool = False) -> int:
        for i, layer in enumerate(self.growing_layers):
            if verbose:
                print(f"Layer {i} update: {layer.first_order_improvement}")
                print(
                    f"Layer {i} parameter improvement: {layer.parameter_update_decrease}"
                )
                print(f"Layer {i} eigenvalues extension: {layer.eigenvalues_extension}")
            if i != layer_index:
                if verbose:
                    print(f"Deleting layer {i}")
                layer.delete_update()
            else:
                self.currently_updated_layer_index = i
        return self.currently_updated_layer_index

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
        for i, layer in enumerate(self.layers):
            statistics[i] = {
                "weight": self.tensor_statistics(layer.weight),
            }
            if layer.bias is not None:
                statistics[i]["bias"] = self.tensor_statistics(layer.bias)
            statistics[i]["input_shape"] = layer.in_features
            statistics[i]["output_shape"] = layer.out_features
        return statistics
