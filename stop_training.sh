#!/bin/bash

# ====== 安全停止スクリプト ======
# Ctrl+C のような「緊急停止」ではなく、
# SIGTERM による「安全停止」を行う。
# latest_step.pt が確実に残るため、再開が安全。

TARGET="main_train.py"
LOCK="checkpoints/.ckpt_lock"

echo "=== 安全停止処理を開始します ==="

# --- PID を配列として取得 ---
pids=($(pgrep -f "$TARGET"))

if [[ ${#pids[@]} -eq 0 ]]; then
    echo "学習プロセス ($TARGET) が見つかりません。すでに停止している可能性があります。"
    exit 1
fi

echo "学習プロセスを安全に停止します (PID=${pids[@]})..."

# --- checkpoint 保存中なら待機 ---
if [[ -f "$LOCK" ]]; then
    echo -e "\033[33m[WAIT] Checkpoint 保存中… 完了を待機します\033[0m"
    while [[ -f "$LOCK" ]]; do
        sleep 1
    done
fi

# --- SIGTERM による安全停止 ---
touch stop.flag
for pid in "${pids[@]}"; do
    kill -15 "$pid"
done

# --- 終了待ち ---
echo "プロセス終了を待機中..."
while pgrep -f "$TARGET" > /dev/null; do
    sleep 1
done

echo "プロセスが正常に停止しました。"

# --- latest_step.pt の確認 ---
if [[ -f checkpoints/latest_step.pt ]]; then
    echo "最新ステップチェックポイントが存在します → checkpoints/latest_step.pt"
    echo "次回起動時に自動で再開されます。"
else
    echo "⚠ 注意: latest_step.pt が見つかりません。"
    echo "再開できない可能性があります。"
fi

echo "=== 安全停止処理が完了しました ==="

