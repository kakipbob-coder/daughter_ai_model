import torch
import torch.nn as nn
from model.attention import MultiHeadAttention

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_head, d_ff, rope_theta=10000):
        super().__init__()

        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_head, rope_theta)

        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x, mask=None):
        # Attention
        attn_out = self.attn(self.ln1(x), mask)
        x = x + attn_out

        # FFN
        ffn_out = self.ffn(self.ln2(x))
        x = x + ffn_out

        return x
