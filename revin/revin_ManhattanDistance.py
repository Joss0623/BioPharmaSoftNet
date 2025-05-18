import torch
import torch.nn as nn
import random


class RevIN(nn.Module):
    """
    Reversible Instance Normalization for Accurate Time-Series Forecasting
    against Distribution Shift, ICLR 2021.

    Parameters
    ----------
    num_features: int
        Number of features or channels.
    eps: float
        Small value added for numerical stability. Default: 1e-5.
    affine: bool
        If True (default), RevIN has learnable affine parameters.
    """

    def __init__(self, num_features: int, eps=1e-5, affine=True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self._init_params()

    def forward(self, x, mode: str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else:
            raise NotImplementedError('Only modes "norm" and "denorm" are supported.')
        return x

    def _init_params(self):
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = [0]
        # Calculate mean over batch dimension and detach from graph
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        # Calculate mean absolute deviation (Manhattan distance) as estimate of std deviation
        distances = torch.mean(torch.abs(x - self.mean), dim=dim2reduce)
        self.stdev = (distances + self.eps).detach()

    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps * self.eps)
        x = x * self.stdev
        x = x + self.mean
        return x


if __name__ == '__main__':
    random.seed(0)
    torch.manual_seed(0)

    # Define tensor shape (batch_size, features)
    tensor_shape = (64, 6)

    # Generate random tensor
    x = torch.rand(tensor_shape)

    layer = RevIN(6)
    y = layer(x, mode='norm')
    z = layer(y, mode='denorm')

    print("Original tensor x:\n", x)
    print("Normalized tensor y:\n", y)
    print("Denormalized tensor z:\n", z)
