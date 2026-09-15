TRAIN_CONFIG = {
    # --- モード管理 ---
    # "base"     → 新規学習（ランダム初期化）
    # "finetune" → 追加学習（会話データのみ）
    "mode": "finetune",
    "version_update": "minor",   # "minor" または "major"

    # --- モデルロード ---
    # base モード → 無視される
    # finetune モード → ここをロードする
    "base_model_path": "checkpoints/latest_base.pt",

    # --- 中間保存 管理（モードごとに分離） ---
    "save": {
        "base": {
            "save_dir": "checkpoints/",
            "epoch": "base_checkpoint_epoch_{epoch}.pt",
            "step":  "base_step_checkpoint_e{epoch}_s{step}.pt",
            "latest": "base_latest_step.pt"
        },
        "finetune": {
            "save_dir": "checkpoints/",
            "epoch": "ft_checkpoint_v{version}_epoch_{epoch}.pt",
            "step":  "ft_step_checkpoint_v{version}based_e{epoch}_s{step}.pt",
            "latest": "ft_latest_step.pt"
        }
    },

    # --- 学習設定 ---
    "seed": 42,
    "train_cfg": {
        "base": {
            "epochs": 3,
            "train_data_path": "data/train_split.txt",
            "val_data_path":   "data/val.txt",
        },
        "finetune": {
            "epochs": 1,
            "train_data_path": "data/mix_train.txt",
            "val_data_path":   "data/mix_val.txt",
        }
    },

    "batch_size": 12,
    "learning_rate": 1.5e-4,
    "max_seq_len": 512,

    # --- モデル構造 ---
    "d_model": 768,
    "n_layers": 12,
    "n_heads": 12,
    "d_ff": 3072,

    # tokenizer
    "tokenizer_path": "tokenizer/tokenizer.model",

    # checkpoint
#    "save_dir": "checkpoints/",

    # ★ validation データの軽量化
    "val_limit": 50000,   # ← ここを追加（必要に応じて調整）

}
