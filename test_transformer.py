import torch
from model.transformer import TransformerLM

# モデルを初期化
model = TransformerLM()

# ダミー入力（バッチ1、長さ10）
dummy_ids = torch.randint(0, model.vocab_size, (1, 10))

# 前向き計算
logits = model(dummy_ids)

print("Input shape:", dummy_ids.shape)
print("Output shape:", logits.shape)
print("Output sample:", logits[0, 0, :5])  # 先頭5語彙だけ表示
