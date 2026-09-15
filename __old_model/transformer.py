import torch
import torch.nn as nn

from config.model_config import MODEL_CONFIG
from model.block import TransformerBlock
from utils.mask import causal_mask


class TransformerLM(nn.Module):
    def __init__(self, config=MODEL_CONFIG):
        super().__init__()

        self.vocab_size = config["vocab_size"]
        self.d_model = config["d_model"]
        self.n_layer = config["n_layer"]
        self.n_head = config["n_head"]
        self.d_ff = config["d_ff"]
        self.max_seq_len = config["max_seq_len"]
        # rope_theta は現状未使用だが残す
        self.rope_theta = config.get("rope_theta", 10000)

        # トークン埋め込み
        self.token_emb = nn.Embedding(self.vocab_size, self.d_model)

        # ブロックを積む
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model=self.d_model,
                n_head=self.n_head,
                d_ff=self.d_ff,
                rope_theta=self.rope_theta,
            )
            for _ in range(self.n_layer)
        ])

        # 最後の LayerNorm
        self.ln_f = nn.LayerNorm(self.d_model)

        # LM ヘッド（語彙への写像）
        self.lm_head = nn.Linear(self.d_model, self.vocab_size, bias=False)

    def forward(self, input_ids):
        """
        input_ids: (B, T) の整数テンソル
        """
        B, T = input_ids.size()
        assert T <= self.max_seq_len, "sequence length exceeds max_seq_len"

        # 埋め込み
        x = self.token_emb(input_ids)  # (B, T, C)

        # causal mask を作成
        mask = causal_mask(T).to(x.device)  # (1, 1, T, T)

        # ブロックを順に適用
        for block in self.blocks:
            x = block(x, mask=mask)

        # 最終 LayerNorm
        x = self.ln_f(x)

        # 語彙ごとのロジット
        logits = self.lm_head(x)  # (B, T, vocab_size)

        return logits
