from torch import nn
from .utils import compute_output_size

class CNNclassifer(nn.Module):
    def __init__(
        self,
        in_dim: int = None,
        n_classes: int = None,
        in_channels: int = 1,
        n_filters: list = None,
        n_neurons: list = None,
        kernel_size: list = None,
        pool_size: tuple = None,
        stride: int = 1,
        dropout: float = 0.0,
    ) -> None:

        super().__init__()

        # Convolutional layers
        self._conv = []
        # Max Pooling layers
        self._pool = []
        # Linear layers
        self._linear = []
        # Batch normalization layers
        self._bnorm = []

        # Compute input to dense layer
        in_dense = compute_output_size(
            pool_size[0], stride, in_dim, kernel_size, n_filters
        )
        # n_filters = [in_channels] + n_filters
        # Create convolutional and batch normalization layers
        for i in range(len(n_filters)):
            if i == 0:
                self._conv += [
                    nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=n_filters[i],
                        kernel_size=kernel_size[i],
                        stride=stride,
                    )
                ]
            else:
                self._conv += [
                    nn.Conv2d(
                        in_channels=n_filters[i - 1],
                        out_channels=n_filters[i],
                        kernel_size=kernel_size[i],
                        stride=stride,
                    )
                ]
            self._bnorm += [nn.BatchNorm2d(num_features=n_filters[i])]

        n_neurons = [in_dense] + n_neurons + [n_classes]

        # Create dense layers
        for i in range(len(n_neurons) - 1):
            self._linear += [
                nn.Linear(in_features=n_neurons[i],
                          out_features=n_neurons[i + 1])
            ]

        # Convert lists to ModuleList
        self._conv = nn.ModuleList(self._conv)
        self._bnorm = nn.ModuleList(self._bnorm)
        self._linear = nn.ModuleList(self._linear)

        # Dropout layer
        self.dropout = nn.Dropout(0.2)
        # Activation layer
        self.activation = nn.ReLU()
        # Max Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=pool_size)
        # Flattening layer
        self.flatten = nn.Flatten()
        self.out = nn.Sigmoid()

    def forward(self, X):

        n_conv = len(self._conv)
        n_dense = len(self._linear)

        for i in range(n_conv):
            X = self.pool(self._bnorm[i](self.activation(self._conv[i](X))))

        X = self.flatten(X)
        for i in range(n_dense):
            X = self._linear[i](X)
            if i < len(self._linear) - 1:
                X = self.dropout(self.activation(X))
        return X
