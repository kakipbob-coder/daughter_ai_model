#!/bin/bash

# ====== Settings ======
TRAIN_PROCESS="main_train.py"   # ★ Python 学習プロセス名を変数化
STATUS_INTERVAL=600             # 10分
ROTATE_INTERVAL=600             # 10分

# ====== Colors ======
RED="\033[31m"
GREEN="\033[32m"
YELLOW="\033[33m"
BLUE="\033[34m"
MAGENTA="\033[35m"
CYAN="\033[36m"
RESET="\033[0m"

# ====== Timestamp ======
get_now() {
    date +"%Y-%m-%d %H:%M:%S"
}

# ====== GPU Status Logging ======
log_gpu_state() {
    local now=$(get_now)
    local json="$(rocm-smi --showuse --showmemuse --json)"
    echo "[$now] $json" >> gpu_state.log
}

# ====== Latest checkpoint ======
find_latest_checkpoint() {
    ls -1t checkpoints/*.pt 2>/dev/null | head -n 1
}

# ====== Checkpoint Rotation (Latest 5 only) ======
rotate_checkpoints() {
    local files=( $(ls -1t checkpoints/*.pt 2>/dev/null) )
    local count=${#files[@]}

    if (( count > 5 )); then
        for ((i=5; i<count; i++)); do
            rm -f "${files[$i]}"
        done
    fi
}

# ====== Discord Webhook ======
DISCORD_WEBHOOK_URL="https://discord.com/api/webhooks/1500071478933061822/np7YoC4I8a5NUlnKN8IwP4SoIv41fUad8KOKJpFeaqKu43LwUINnidnb0AmqcvCeC5hS"

send_discord() {
    local message="$1"
    local now=$(get_now)
    curl -H "Content-Type: application/json" \
        -X POST \
        -d "{\"content\": \"[$now] ${message}\"}" \
        "$DISCORD_WEBHOOK_URL"
}

# ====== LINE Notify ======
TOKEN="$LINE_CHANNEL_TOKEN"
USER="$LINE_USER_ID"

send_line() {
    local message="$1"
    local now=$(get_now)
    curl -X POST \
        -H "Content-Type: application/json" \
        -H "Authorization: Bearer ${TOKEN}" \
        -d "{
            \"to\": \"${USER}\",
            \"messages\": [{
                \"type\": \"text\",
                \"text\": \"[$now] ${message}\"
            }]
        }" \
        https://api.line.me/v2/bot/message/push
}

# ====== GPU Info ======
get_gpu_info() {
    rocm-smi --showuse --showmemuse --json
}

# ====== GPU Usage / VRAM / ROCm 落ち ======
check_gpu() {
    local json="$(get_gpu_info)"

    if [[ -z "$json" ]]; then
        echo -e "${RED}[ERROR] GPU が認識されていません（ROCm 落ちの可能性）${RESET}"
        send_discord "⚠ GPU が認識されていません（ROCm 落ちの可能性）"
        send_line    "⚠ GPU が認識されていません（ROCm 落ちの可能性）"
        return
    fi

    local use=$(echo "$json" | jq -r '.card0["GPU use (%)"]')
    local vram=$(echo "$json" | jq -r '.card0["GPU Memory Allocated (VRAM%)"]')

    if (( use > 90 )); then
        echo -e "${YELLOW}[WARN] GPU 使用率が高いです: ${use}%${RESET}"
        send_discord "⚠ GPU 使用率が高いです: ${use}%"
    fi

    if (( vram > 80 )); then
        echo -e "${YELLOW}[WARN] VRAM 使用量が高いです: ${vram}%${RESET}"
        send_discord "⚠ VRAM 使用量が高いです: ${vram}%"
    fi
}

# ====== Python Process ======
check_python() {
    if pgrep -f "$TRAIN_PROCESS" > /dev/null; then
        :
    else
        # ★ 手動停止検知
        if [[ -f stop.flag ]]; then
            echo -e "${MAGENTA}[STOP] 手動停止を検知しました${RESET}"
            send_discord "🛑 手動停止を検知しました"
            rm stop.flag
            return
        fi
        
        echo -e "${RED}[ERROR] Python 学習プロセス($TRAIN_PROCESS)が停止しました${RESET}"
        send_discord "⚠ Python 学習プロセス($TRAIN_PROCESS)が停止しました"
        send_line    "⚠ Python 学習プロセス($TRAIN_PROCESS)が停止しました"
    fi
}

# ====== GPU Hang Detection ======
check_gpu_hang() {
    local json="$(get_gpu_info)"

    if [[ -z "$json" ]]; then
        echo -e "${RED}[GPU Hang] ROCm が GPU を認識していません${RESET}"
        send_discord "🛑 GPU Hang 検出：ROCm が GPU を認識していません"
        send_line    "🛑 GPU Hang 検出：ROCm が GPU を認識していません"
        return
    fi

    local gpu=$(echo "$json" | jq -r '.card0["GPU use (%)"]')
    local vram=$(echo "$json" | jq -r '.card0["GPU Memory Allocated (VRAM%)"]')

    if (( gpu == 0 && vram > 20 )); then
        echo -e "${RED}[GPU Hang] GPU 使用率 0% / VRAM ${vram}%${RESET}"
        send_discord "🛑 GPU Hang の可能性：GPU 使用率 0% / VRAM ${vram}%"
        send_line    "🛑 GPU Hang の可能性：GPU 使用率 0% / VRAM ${vram}%"

        cp gpu_state.log gpu_state_before_hang.log

        rotate_checkpoints

        latest=$(find_latest_checkpoint)

        if [[ -n "$latest" ]]; then
            echo -e "${MAGENTA}[RESTART]最新 checkpoint から再開します: ${latest}${RESET}"
            send_discord "🔄 最新 checkpoint から再開します: ${latest}"

            #jq --arg path "$latest" '.resume_from = $path' config.json > config_tmp.json
            #mv config_tmp.json config.json
            
            # ★ checkpoint 保存中なら再起動を保留
            if [[ -f checkpoints/.ckpt_lock ]]; then
                echo -e "${YELLOW}[INFO] Checkpoint 保存中のため再起動を保留${RESET}"
                return
            fi

            pkill -f "$TRAIN_PROCESS"

            bash run_train.sh &
            exit 0
        else
            echo -e "${RED}[ERROR] checkpoint が見つからず再開できません${RESET}"
            send_discord "⚠ checkpoint が見つからず再開できません"
        fi
    fi
}

# ====== 学習進捗通知 ======
check_status() {
    if [[ -f train_status.txt ]]; then
        local status=$(cat train_status.txt)
        echo -e "${CYAN}[STATUS] ${status}${RESET}"
        send_discord "📘 学習進行状況: ${status}"
    fi
}

# ====== 10分ごとに確実に通知 ======
check_status_interval() {
    local now=$(date +%s)
    if (( now - last_status_time >= STATUS_INTERVAL )); then
        check_status
        last_status_time=$now
    fi
}

# ====== 10分ごとに checkpoint ローテーション ======
rotate_interval() {
    local now=$(date +%s)
    if (( now - last_rotate_time >= ROTATE_INTERVAL )); then
        rotate_checkpoints
        last_rotate_time=$now
    fi
}

# ====== 学習完了通知 ======
check_training_done() {
    if [[ -f training_done.flag ]]; then
        echo -e "${GREEN}[SUCCESS] 学習が完了しました${RESET}"
        send_discord "🎉 学習が完了しました"
        send_line    "🎉 学習が完了しました"
        rm training_done.flag
    fi
}

# ====== エポック完了通知 ======
check_epoch_done() {
    if [[ -f epoch_done.flag ]]; then
        local epoch=$(cat epoch_done.flag)
        echo -e "${GREEN}[STATUS] エポック ${epoch} が完了しました${RESET}"
        send_discord "📗 エポック ${epoch} が完了しました"
        send_line    "📗 エポック ${epoch} が完了しました"
        rm epoch_done.flag
    fi
}

# ====== Ctrl+C（SIGINT）で終了通知 ======
trap '
    echo -e "${GREEN}[INFO] GPU Monitor Bot が終了しました（Ctrl+C）${RESET}"
    send_discord "GPU Monitor Bot が終了しました（Ctrl+C）"
    send_line    "GPU Monitor Bot が終了しました（Ctrl+C）"
    exit 0
' INT

# ====== 初期化フェーズ ======
# ★ 初回通知を抑止するため、現在時刻で初期化
last_status_time=$(date +%s)
last_rotate_time=$(date +%s)

# ====== 起動通知 ======
echo -e "${GREEN}[INFO] GPU Monitor Bot を開始しました${RESET}"
send_discord "GPU Monitor Bot を開始しました"
send_line    "GPU Monitor Bot を開始しました"


# ====== Main Loop ======
while true; do
    check_gpu
    check_python
    check_gpu_hang

    rotate_interval
    check_training_done
    check_epoch_done

    log_gpu_state

    check_status_interval

    sleep 60
done

