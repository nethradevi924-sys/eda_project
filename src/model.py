"""
model.py
Deep Learning Multi-Layer Perceptron (MLP) architecture in PyTorch for
tabular employee attrition risk prediction.
"""

import torch
import torch.nn as nn


class AttritionMLP(nn.Module):
    """
    Multi-Layer Perceptron with Batch Normalization, LeakyReLU,
    and Dropout Regularization designed for tabular risk classification.
    """
    def __init__(self, in_features: int, hidden_dims: list = [128, 64, 32], dropout_rates: list = [0.3, 0.2, 0.1]):
        super().__init__()
        
        layers = []
        prev_dim = in_features
        
        # Hidden Layer 1
        layers.append(nn.Linear(prev_dim, hidden_dims[0]))
        layers.append(nn.BatchNorm1d(hidden_dims[0]))
        layers.append(nn.LeakyReLU(0.1))
        layers.append(nn.Dropout(dropout_rates[0]))
        prev_dim = hidden_dims[0]
        
        # Hidden Layer 2
        layers.append(nn.Linear(prev_dim, hidden_dims[1]))
        layers.append(nn.BatchNorm1d(hidden_dims[1]))
        layers.append(nn.LeakyReLU(0.1))
        layers.append(nn.Dropout(dropout_rates[1]))
        prev_dim = hidden_dims[1]
        
        # Hidden Layer 3
        layers.append(nn.Linear(prev_dim, hidden_dims[2]))
        layers.append(nn.LeakyReLU(0.1))
        layers.append(nn.Dropout(dropout_rates[2]))
        prev_dim = hidden_dims[2]
        
        # Output Classification Layer
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(-1)


def count_parameters(model: nn.Module) -> int:
    """Returns total trainable parameter count."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    dummy_input = torch.randn(16, 45)
    model = AttritionMLP(in_features=45)
    output = model(dummy_input)
    print(model)
    print(f"Output shape: {output.shape}")
    print(f"Total Trainable Parameters: {count_parameters(model)}")
