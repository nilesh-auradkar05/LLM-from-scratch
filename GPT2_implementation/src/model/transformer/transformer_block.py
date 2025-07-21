import torch.nn as nn

from src.model.transformer.mha import MultiHeadAttention
from src.model.transformer.ffn import FeedForwardNNBlock
from src.model.normalization.rmsnorm import RMSNorm

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.mask_attn = MultiHeadAttention(
            d_in=cfg["embedding_dim"],
            d_out=cfg["embedding_dim"],
            num_heads=cfg["num_heads"],
            dropout=cfg["drop_rate"],
            context_length=cfg["context_length"],
            qkv_bias=cfg["qkv_bias"],
        )
        
        self.ffn_block = FeedForwardNNBlock(cfg)
        self.norm_1 = RMSNorm(cfg["embedding_dim"])
        self.norm_2 = RMSNorm(cfg["embedding_dim"])
        self.drop_shorcut = nn.Dropout(cfg["drop_rate"])
        
    def forward(self, x):
        # 1. Shortcut connection from input droput to layer norm in transformer block
        shortcut = x
        x = self.norm_1(x)
        x = self.mask_attn(x)
        x = self.drop_shorcut(x)
        x = x + shortcut
        
        # 2. Shortcut connection from dropout in transformer to layer norm in 2 phase of transformer
        shortcut = x
        x = self.norm_2(x)
        x = self.ffn_block(x)
        x = self.drop_shorcut(x)
        x = x + shortcut
        
        return x
        