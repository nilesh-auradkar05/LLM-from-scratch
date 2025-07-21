import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, embedding_dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        # The gain is a learnable parameter for model. Similar to scale parameter in layer norm.
        self.gain = nn.Parameter(torch.ones(embedding_dim))
        
    def forward(self, x):
        # Calculate the root mean square of the last dimension.
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        # Normalize and apply gain.
        return x / rms * self.gain
    