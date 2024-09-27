
import torch

from torch import nn

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers):
        super().__init__()

        self.layers = nn.ModuleList()

        for layer in range(n_layers):
            dim = input_dim if layer == 0 else hidden_dim
            self.layers.append(nn.Sequential(
                               nn.Linear(dim, hidden_dim),
                               nn.BatchNorm1d(hidden_dim),
                               nn.ReLU())
                               )

        self.layers.append(nn.Sequential(
                           nn.Linear(hidden_dim, output_dim))
                           )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
