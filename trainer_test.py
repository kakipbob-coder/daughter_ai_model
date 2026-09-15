import torch
from src.model import GPTModel

# 最小テスト用の設定
vocab_size = 32000
d_model = 128
n_layers = 2
n_heads = 4
d_ff = 512
max_seq_len = 32

# モデル初期化
model = GPTModel(
    vocab_size=vocab_size,
    d_model=d_model,
    n_layers=n_layers,
    n_heads=n_heads,
    d_ff=d_ff,
    max_seq_len=max_seq_len,
)

# ダミー入力（2バッチ × 16トークン）
input_ids = torch.randint(0, vocab_size, (2, 16))
labels = input_ids.clone()

# forward テスト
logits, loss = model(input_ids, labels)

print("logits shape:", logits.shape)
print("loss:", loss.item())
