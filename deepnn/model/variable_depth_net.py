import torch.nn as nn
import torch.nn.functional as F

class VariableDepthNet(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, dropout=0.0):
        super(VariableDepthNet, self).__init__()
        self.dropout = dropout
        self.layers = nn.ModuleList()

        # Input layer
        self.layers.append(nn.Linear(input_size, hidden_sizes[0]))

        # Hidden layers
        for i in range(1, len(hidden_sizes)):
            self.layers.append(nn.Linear(hidden_sizes[i - 1], hidden_sizes[i]))

        # Output layer
        self.layers.append(nn.Linear(hidden_sizes[-1], output_size))

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
            x = F.dropout(x, self.dropout)
        x = self.layers[-1](x)  # No activation function on the output layer
        x = F.tanh(x)
        return x

