import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, num_heads, dropout, context_length, qkv_bias=True):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"
        
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        
        # 1. Initialize the Weight matrices
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )
        
    def forward(self, x):
        batch_size, num_tokens, d_in = x.shape
        
        # 2. Input * weight matrices
        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)
        
        # 3. Rolling last dimension d_out to num_heads, head_dim
        # (batch_size, num_tokens, d_out) -> (batch_size, num_tokens, num_heads, head_dim)
        # Transpose to group weights by num_heads so attention can be computed in parallel.
        queries = queries.view(batch_size, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        keys = keys.view(batch_size, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        values = values.view(batch_size, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 4. Calculate the attention scores
        # Queries @ Keys.T
        # (batch_size, num_heads, num_tokens, head_dim) @ (batch_size, num_heads, head_dim, num_tokens)
        # Output: (batch_size, num_heads, num_tokens, num_tokens)
        attn_scores = queries @ keys.transpose(-2, -1)
        
        # 5. Apply mask to future tokens
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask_bool, -torch.inf)
        
        # 6. Apply softmax to get attention weights
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        
        # 7. Apply dropout to attention weights
        attn_weights = self.dropout(attn_weights)
        
        # 8. Compute context vector
        # attn_weights @ values
        # (batch_size, num_heads, num_tokens, num_tokens) @ (batch_size, num_heads, num_tokens, head_dim)
        # Output: (batch_size, num_heads, num_tokens, head_dim)
        context_vec = attn_weights @ values
        
        # 9. Reshape context vector to (batch_size, num_tokens, num_heads, head_dim) to match d_out shape
        # Concatenate all heads and project to d_out
        context_vec = context_vec.transpose(1, 2)
        context_vec = context_vec.contiguous().view(batch_size, num_tokens, self.d_out)
        
        # 10. Project to d_out
        context_vec = self.out_proj(context_vec)
        
        return context_vec