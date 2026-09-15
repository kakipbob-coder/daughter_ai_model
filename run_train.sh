#!/bin/bash

# --- GPU 電力制限（省エネモード） ---
# 200W に制限（必要なら値を変更）
echo "[INFO] Setting GPU power limit to 200W"
rocm-smi --setpoweroverdrive 200

# --- GUI を落として TTY に移行（必要なときだけ手動で実行） ---
# sudo systemctl isolate multi-user.target

# --- ROCm / PyTorch の安定動作用 環境変数 ---
export HSA_OVERRIDE_GFX_VERSION=11.0.0
export HIP_VISIBLE_DEVICES=0
export AMD_SERIALIZE_KERNEL=3

# --- 学習実行 ---
python ./main_train.py 2>&1 | tee train_run.log

# --- GPU 電力制限をデフォルトに戻す（任意） ---
echo "[INFO] Resetting GPU power limit to default"
rocm-smi --resetpoweroverdrive

# --- GUI を戻す（必要なときだけ手動で実行） ---
# sudo systemctl isolate graphical.target

