TRAIN_CONFIG = {
    "seed": 42,
    "epochs": 3,
    "batch_size": 12,
    "learning_rate": 1.5e-4,
    "max_seq_len": 512,

    # model
    "d_model": 768,
    "n_layers": 12,
    "n_heads": 12,
    "d_ff": 3072,

    # tokenizer
    "tokenizer_path": "tokenizer/tokenizer.model",

    # data
    "train_data_path": "data/train_split.txt",
    "val_data_path": "data/val.txt",

    # checkpoint
    "save_dir": "checkpoints/",

    # ★ validation データの軽量化
    "val_limit": 50000,   # ← ここを追加（必要に応じて調整）
}
