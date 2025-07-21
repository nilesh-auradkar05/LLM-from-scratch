import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.transformer.transformer_block import TransformerBlock
from src.model.normalization.rmsnorm import RMSNorm

class ModifiedGPT2ModelArch(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_embeddings = nn.Embedding(cfg["vocab_size"], cfg["embedding_dim"])
        self.pos_embeddings = nn.Embedding(cfg["context_length"], cfg["embedding_dim"])
        self.dropout = nn.Dropout(cfg["drop_rate"])
        
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["num_layers"])]
        )
        
        self.final_rms_norm = RMSNorm(cfg["embedding_dim"])
        self.out_head = nn.Linear(cfg["embedding_dim"], cfg["vocab_size"], bias=False)
        
    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeddings = self.tok_embeddings(in_idx)
        pos_embeddings = self.pos_embeddings(torch.arange(seq_len, device=in_idx.device))
        
        x = tok_embeddings + pos_embeddings
        x = self.dropout(x)
        x = self.trf_blocks(x)
        x = self.final_rms_norm(x)
        logits = self.out_head(x)
        return logits