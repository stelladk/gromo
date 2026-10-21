import torch

from gromo.modules.conv2d_growing_module import Conv2dGrowingModule
from gromo.modules.linear_growing_module import LinearGrowingModule


class ConstantModule(LinearGrowingModule):
    """Placeholder Module with constant zero output.
    Used to simulate zero connection between modules

    Parameters
    ----------
    in_features : int
        input features
    out_features : int
        output features
    device : torch.device | None
        default device, by default None
    """

    def __init__(
        self, in_features: int, out_features: int, device: torch.device | None = None
    ) -> None:
        super(ConstantModule, self).__init__(
            in_features=in_features,
            out_features=out_features,
            use_bias=False,
            device=device,
        )
        # Store the constant tensor as a buffer (non-trainable parameter)
        # self.register_buffer('constant', torch.zeros(in_features, out_features))
        # Remove the inert weight so the zero function holds no trainable parameters
        del self.layer.weight
        self.layer.register_buffer(
            "weight", torch.zeros(out_features, in_features, device=device)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Placeholder function that ignores the input and always returns the constant tensor

        Parameters
        ----------
        x : torch.Tensor
            input tensor

        Returns
        -------
        torch.Tensor
            constant tensor
        """
        # Ignore the input x and always return the constant tensor
        self.register_buffer(
            "constant",
            torch.zeros(
                len(x), self.out_features, requires_grad=True, device=self.device
            ),
        )
        return self.constant

    def __setattr__(self, key, value):
        if key == "optimal_delta_layer":
            torch.nn.Module.__setattr__(self, "_hidden_optimal_delta_layer", value)
        else:
            super().__setattr__(key, value)

    @property
    def optimal_delta_layer(self) -> torch.nn.Linear:
        """Placeholder function that returns a zero weight module as a mock update to the constant module

        Returns
        -------
        torch.nn.Linear
            zero-weight linear layer
        """
        return self.layer_of_tensor(
            torch.zeros_like(self._hidden_optimal_delta_layer.weight, device=self.device)
        )


class ConstantConvModule(Conv2dGrowingModule):
    """Placeholder Module with constant zero output for convolutional nodes.
    Used to simulate zero connection between convolution modules.
    Unlike the linear ConstantModule, it is shaped like a convolution (matching
    output channels) so it can be wired as an edge between Conv2dMergeGrowingModule
    nodes.

    Parameters
    ----------
    in_channels : int
        input channels
    out_channels : int
        output channels
    kernel_size : int | tuple[int, int]
        convolution kernel size
    input_size : tuple[int, int]
        spatial size (h, w) of the input feature map
    device : torch.device | None
        default device, by default None
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        input_size: tuple[int, int],
        device: torch.device | None = None,
    ) -> None:
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        # "same" padding keeps the spatial size, matching a real conv edge
        padding = ((kernel_size[0] - 1) // 2, (kernel_size[1] - 1) // 2)
        super(ConstantConvModule, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            input_size=input_size,
            padding=padding,
            use_bias=False,
            device=device,
        )
        # Remove the inert weight so the zero function holds no trainable parameters
        del self.layer.weight
        self.layer.register_buffer(
            "weight",
            torch.zeros(out_channels, in_channels, *kernel_size, device=device),
        )

    def _output_size(self) -> tuple[int, int]:
        """Spatial size (h, w) of the output feature map"""
        k = self.layer.kernel_size
        s = self.layer.stride
        p = self.layer.padding
        d = self.layer.dilation
        return tuple(
            (self._input_size[i] + 2 * p[i] - d[i] * (k[i] - 1) - 1) // s[i] + 1
            for i in range(2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Placeholder function that ignores the input and always returns the constant tensor

        Parameters
        ----------
        x : torch.Tensor
            input tensor

        Returns
        -------
        torch.Tensor
            constant tensor of shape (batch, out_channels, h, w)
        """
        # Ignore the input x and always return the constant tensor
        h, w = self._output_size()
        self.register_buffer(
            "constant",
            torch.zeros(
                len(x),
                self.out_channels,
                h,
                w,
                requires_grad=True,
                device=self.device,
            ),
        )
        return self.constant

    def __setattr__(self, key, value):
        if key == "optimal_delta_layer":
            torch.nn.Module.__setattr__(self, "_hidden_optimal_delta_layer", value)
        else:
            super().__setattr__(key, value)

    @property
    def optimal_delta_layer(self) -> torch.nn.Conv2d:
        """Placeholder function that returns a zero weight module as a mock update to the constant module

        Returns
        -------
        torch.nn.Conv2d
            zero-weight convolution layer
        """
        layer = torch.nn.Conv2d(
            self.in_channels,
            self.out_channels,
            self.layer.kernel_size,
            stride=self.layer.stride,
            padding=self.layer.padding,
            dilation=self.layer.dilation,
            bias=False,
            device=self.device,
        )
        torch.nn.init.zeros_(layer.weight)
        return layer
