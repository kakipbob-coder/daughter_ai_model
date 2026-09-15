# File : model.py
# Description :
#   Core implementation of a decoder-only GPT model.
#   Defines token/positional embeddings, multi-head self-attention,
#   feed-forward network, transformer blocks, and the final LM head.
#   Provides both forward() for training and generate() for autoregressive inference.
#
# 説明 :
#   Decoder-only（自己回帰型）GPT モデルの本体実装。
#   ・トークン埋め込み（token_emb）
#   ・位置埋め込み（pos_emb）
#   ・Multi-Head Self-Attention（自己注意）
#   ・FeedForward（中間層）
#   ・TransformerBlock（思考の層）
#   ・LayerNorm（安定化）
#   ・LM Head（次トークン予測）
#   ・重み共有（token_emb と lm_head）
#
#   forward() では学習用のロジック（logits と cross-entropy loss）を返し、
#   generate() では逐次生成（autoregressive decoding）を行う。
#
#   AMP（自動混合精度）と相性が良い素直な構造で、
#   trainer.py から呼び出される GPT モデルの中心的役割を担う。


import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # x: (B, T, C)
        B, T, C = x.size()

        q = self.q_proj(x)  # (B, T, C)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # (B, T, C) -> (B, n_heads, T, d_head)
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        # scaled dot-product attention
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)  # (B, n_heads, T, T)

        if mask is not None:
            att = att.masked_fill(mask == 0, float("-inf"))

        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        y = att @ v  # (B, n_heads, T, d_head)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # (B, T, C)

        y = self.out_proj(y)
        y = self.resid_dropout(y)
        return y


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, n_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)

    def forward(self, x, mask=None):
        # Pre-LN
        x = x + self.attn(self.ln1(x), mask=mask)
        x = x + self.ff(self.ln2(x))
        return x


class GPTModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 768,
        n_layers: int = 12,
        n_heads: int = 12,
        d_ff: int = 3072,
        max_seq_len: int = 1024,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        # 語彙の世界観
        self.token_emb = nn.Embedding(vocab_size, d_model)
        # 文の順番の感覚（Learned Positional Embedding）
        self.pos_emb = nn.Embedding(max_seq_len, d_model)

        self.drop = nn.Dropout(dropout)

        # 思考の層
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(d_model, n_heads, d_ff, dropout)
                for _ in range(n_layers)
            ]
        )

        # 精神安定剤
        self.ln_f = nn.LayerNorm(d_model)

        # 発話の口（LM Head）
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Embedding と LM Head の重み共有
        self.lm_head.weight = self.token_emb.weight

        # causal mask（未来を見ない）
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(max_seq_len, max_seq_len)).unsqueeze(0).unsqueeze(0),
            persistent=False,
        )

    def forward(self, input_ids, labels=None):
        """
        input_ids: (B, T)
        labels: (B, T) or None
        """
        # --- モデルのデバイスを基準に統一 ---
        device = next(self.parameters()).device

        # 入力をモデルのデバイスに移す（安全策）
        input_ids = input_ids.to(device)
        if labels is not None:
            labels = labels.to(device)

        B, T = input_ids.size()

        if T > self.max_seq_len:
            raise ValueError(f"Sequence length {T} > max_seq_len {self.max_seq_len}")

        # 位置ID（device を明示、dtype を long に）
        pos = torch.arange(0, T, device=device, dtype=torch.long).unsqueeze(0)  # (1, T)

        # 埋め込み（入力は既に device 上）
        tok_emb = self.token_emb(input_ids)  # (B, T, C)
        pos_emb = self.pos_emb(pos)          # (1, T, C)
        x = tok_emb + pos_emb
        x = self.drop(x)

        # マスク（B, 1, T, T）を必ずモデルの device に移す
        mask = self.causal_mask[:, :, :T, :T].to(device)

        # Transformer Blocks
        for block in self.blocks:
            x = block(x, mask=mask)

        x = self.ln_f(x)  # (B, T, C)

        logits = self.lm_head(x)  # (B, T, vocab_size)

        loss = None
        if labels is not None:
            # 1トークン先を予測する言語モデル損失
            loss = F.cross_entropy(
                logits.view(-1, self.vocab_size),
                labels.view(-1),
                ignore_index=-100,
            )

        return logits, loss


    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens: int = 50, temperature: float = 1.0, top_k: int = None):
        """
        シンプルな逐次生成
        input_ids: (B, T)
        """
        # 保存しておく（呼び出し前のモードを復元するため）
        was_training = self.training
        self.eval()

        # モデルのパラメータが乗っているデバイスを基準にする
        device = next(self.parameters()).device
        input_ids = input_ids.to(device).long()

        for _ in range(max_new_tokens):
            B, T = input_ids.size()
            if T > self.max_seq_len:
                input_cond = input_ids[:, -self.max_seq_len :]
            else:
                input_cond = input_ids

            # forward は内部で device を揃える実装になっている前提
            logits, _ = self(input_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-6)  # (B, vocab)

            # top_k が指定されている場合はバッチごとに閾値を計算してマスク
            if top_k is not None and top_k > 0:
                # topk の最小値を閾値として使う（バッチごと）
                kth_vals = torch.topk(logits, top_k, dim=-1)[0][:, -1].unsqueeze(-1)  # (B, 1)
                mask = logits < kth_vals
                logits = logits.masked_fill(mask, -float("inf"))

            probs = F.softmax(logits, dim=-1)
            # 安全のため NaN/inf をチェック（まれに temperature が極端な値で発生）
            probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)

            next_token = torch.multinomial(probs, num_samples=1)  # (B, 1)
            input_ids = torch.cat([input_ids, next_token], dim=1)

        # 呼び出し前のモードを復元
        if was_training:
            self.train()

        return input_ids

