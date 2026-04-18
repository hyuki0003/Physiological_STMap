import torch
import torch.nn as nn

class CrossAttention(nn.Module):
    def __init__(self, dim, heads=8, dropout=0.2):
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)
        self.to_out = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(self, query, key, value):
        B, N, C = query.shape

        q = self.to_q(query).reshape(B, N, self.heads, -1).transpose(1, 2)  # (B, heads, N, dim//heads)
        k = self.to_k(key).reshape(B, N, self.heads, -1).transpose(1, 2)
        v = self.to_v(value).reshape(B, N, self.heads, -1).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, heads, N, N)
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        out = attn @ v  # (B, heads, N, dim//heads)
        out = out.transpose(1, 2).reshape(B, N, C)  # (B, N, dim)
        out = self.to_out(out)
        return self.norm(out + query)  # Residual + Norm

