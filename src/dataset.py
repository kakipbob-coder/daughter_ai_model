# File: src/dataset.py
# Description:
#   Dataset class for next-token prediction training.
#   Loads a text file where each line is treated as one training sample.
#   Applies optional line limiting (limit) for lightweight validation datasets.
#
# 説明:
#   GPT 系モデル向けの next-token prediction 用 Dataset。
#   1 行を 1 サンプルとして読み込み、tokenizer で ID 化し、
#   max_seq_len に合わせて切り詰め・padding を行う。
#   limit を指定した場合は、先頭 N 行のみを使用する（validation の軽量化用）。
#
#   主な用途:
#     - train_dataset: limit=None（全データ使用）
#     - val_dataset:   limit=N（軽量化した検証用データ）
#
#   Trainer 側で train/val の API を統一するため、
#   limit 引数は train 側にも渡すが、train では None を指定する。

import torch
from torch.utils.data import Dataset


class TextDataset(Dataset):
    """
    単純な next-token prediction 用 Dataset
    1 行 1 サンプルのテキストを想定
    """

    def __init__(self, file_path, tokenizer, max_seq_len=512, limit=None):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

        with open(file_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]

        # ★ limit が指定されていれば先頭 N 行だけ使う
        if limit is not None:
            lines = lines[:limit]

        self.lines = lines  # ★ self.lines に保存する

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        text = self.lines[idx]

        ids = self.tokenizer.encode(text)

        # 1. 長すぎる場合は切る
        ids = ids[: self.max_seq_len]

        # 2. 短い場合は pad する
        if len(ids) < self.max_seq_len:
            pad_id = self.tokenizer.pad_id  # SentencePiece の pad_id
            ids = ids + [pad_id] * (self.max_seq_len - len(ids))

        # 3. x と y を作る（固定長になる）
        x = torch.tensor(ids[:-1], dtype=torch.long)
        y = torch.tensor(ids[1:], dtype=torch.long)

        return x, y

