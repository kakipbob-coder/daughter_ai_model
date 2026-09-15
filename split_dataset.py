import random

input_path = "data/train.txt"
train_out = "data/train_split.txt"
val_out = "data/val.txt"
ratio = 0.95

with open(input_path, "r", encoding="utf-8") as f:
    lines = [line for line in f if line.strip()]

random.shuffle(lines)

split_idx = int(len(lines) * ratio)
train_lines = lines[:split_idx]
val_lines = lines[split_idx:]

with open(train_out, "w", encoding="utf-8") as f:
    f.writelines(train_lines)

with open(val_out, "w", encoding="utf-8") as f:
    f.writelines(val_lines)

print("Train:", len(train_lines))
print("Val:", len(val_lines))

