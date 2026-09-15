# File: /infer/generate.py
# Description:
#   Text generation module for GPT-style language models using GPTModel.
#   Handles model loading, tokenizer initialization, and inference logic
#   including greedy decoding and top-k sampling.
#
# 説明:
#   GPTModel を用いた推論処理を担当する生成モジュール。
#   モデルおよびトークナイザの読み込み、推論処理、
#   greedy / top-k サンプリングによるテキスト生成を実行する。
#
# Notes:
#   - Trainer と同じ GPTModel を使用する。
#   - checkpoint は {"model": ..., "optimizer": ..., ...} 形式を想定。
#   - 推論時は model(x) のみ（y は渡さない）。

import torch
from tokenizer.spm_wrapper import Tokenizer
from src.model import GPTModel
from config.train_config import TRAIN_CONFIG


class Generator:
    def __init__(self, model_path, tokenizer_path, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # tokenizer
        self.tokenizer = Tokenizer(tokenizer_path)

        # model
        self.model = GPTModel(
            vocab_size=self.tokenizer.vocab_size,
            d_model=TRAIN_CONFIG["d_model"],
            n_layers=TRAIN_CONFIG["n_layers"],
            n_heads=TRAIN_CONFIG["n_heads"],
            d_ff=TRAIN_CONFIG["d_ff"],
            max_seq_len=TRAIN_CONFIG["max_seq_len"],
        ).to(self.device)

        # checkpoint load
        state = torch.load(model_path, map_location=self.device)

        # checkpoint の形式を自動判定
        if isinstance(state, dict) and "model" in state:
            # finetune の辞書形式
            self.model.load_state_dict(state["model"])
        else:
            # base の state_dict 形式
            self.model.load_state_dict(state)

        self.model.eval()

        self.max_seq_len = TRAIN_CONFIG["max_seq_len"]

    @torch.no_grad()
    def generate(self, prompt, max_new_tokens=50, temperature=1.0, top_k=50):
        ids = self.tokenizer.encode(prompt)
        x = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(self.device)

        for _ in range(max_new_tokens):
            if x.size(1) >= self.max_seq_len:
                break

            # GPTModel forward (y=None)
            logits, _ = self.model(x, None)

            logits = logits[:, -1, :] / temperature

            # top-k sampling
            if top_k is not None:
                values, _ = torch.topk(logits, top_k)
                min_value = values[:, -1].unsqueeze(-1)
                logits = torch.where(logits < min_value, torch.full_like(logits, -1e10), logits)

            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            x = torch.cat([x, next_token], dim=1)

        out = x[0].tolist()

        # 入力プロンプトのトークン数
        prompt_len = len(ids)

        # 生成部分だけをデコード
        generated = out[prompt_len:]

        return self.tokenizer.decode(generated)

