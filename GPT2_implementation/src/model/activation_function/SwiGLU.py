import torch
import torch.nn as nn
import torch.nn.functional as F

# SwiGLU is a activation function that is used in the Modern transformer models.
# Originally GPT-2 uses GELU as activation function.
# SwiGLU is a combination of Swish and GELU.
# It splits a linear transformation's output into two parts, applies Swish to the gate
# and multiplies the result with the value part.
class SwiGLU(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear_gate = nn.Linear(in_features, out_features)
        self.linear_value = nn.Linear(in_features, out_features)
        
    def forward(self, x):
        # the gate determines which part in the "value" part are important.
        gate = self.linear_gate(x)
        value = self.linear_value(x)
        return F.silu(gate) * value
    