import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head, rope_theta=10000):
        super().__init__()
        assert d_model % n_head == 0

        self.d_model = d_model
        self.n_head = n_head
        self.head_dim = d_model // n_head

        # QKV projection
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)

        # Output projection
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

        # RoPE base frequency
        self.rope_theta = rope_theta

    def forward(self, x, mask=None):
        B, T, C = x.size()

        # Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # reshape to (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Apply RoPE
        q, k = self.apply_rope(q, k)

        # Scaled dot-product attention
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if mask is not None:
            att = att.masked_fill(mask == 0, float('-inf'))

        att = torch.softmax(att, dim=-1)
        out = att @ v

        # reshape back
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        return self.o_proj(out)

    def apply_rope(self, q, k):
        # q, k: (B, n_head, T, head_dim)
        B, H, T, D = q.shape
        half = D // 2

        # 周波数
        freq = 1.0 / (self.rope_theta ** (torch.arange(0, half, 2, device=q.device) / half))

        # 位置 t = 0..T-1
        t = torch.arange(T, device=q.device).float()

        # 回転角 θ = t * freq
        theta = torch.einsum("t,f->tf", t, freq)

        # cos, sin を作る
        cos = torch.repeat_interleave(torch.cos(theta), 2, dim=-1)  # (T, half)
        sin = torch.repeat_interleave(torch.sin(theta), 2, dim=-1)

        # reshape for broadcasting
        cos = cos.unsqueeze(0).unsqueeze(0)  # (1,1,T,half)
        sin = sin.unsqueeze(0).unsqueeze(0)

        # q, k を前半・後半に分割
        q1, q2 = q[..., :half], q[..., half:]
        k1, k2 = k[..., :half], k[..., half:]

        # 回転
        q_rot = q1 * cos - q2 * sin
        q_rot2 = q1 * sin + q2 * cos

        k_rot = k1 * cos - k2 * sin
        k_rot2 = k1 * sin + k2 * cos

        # 結合
        q_out = torch.cat([q_rot, q_rot2], dim=-1)
        k_out = torch.cat([k_rot, k_rot2], dim=-1)

        return q_out, k_out
