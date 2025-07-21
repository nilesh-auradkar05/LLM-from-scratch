import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.activation_function.SwiGLU import SwiGLU

class FeedForwardNNBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        embedding_dim = cfg["embedding_dim"]
        # neurons in hidden layer is 4 times of embedding_dim in original GPT-2.
        # SwiGLU expands by 8/3 factor.
        hidden_dim = int((4 * embedding_dim) * (2/3))
        
        self.layers = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            SwiGLU(hidden_dim, embedding_dim),
        )
        
    def forward(self, x):
        return self.layers(x)