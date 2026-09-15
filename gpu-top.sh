#!/bin/bash
width=46

bar() {
  local val=$1
  local max=100
  local width=20
  local filled=$(( val * width / max ))
  local empty=$(( width - filled ))
  printf "%s%s" "$(printf '█%.0s' $(seq 1 $filled))" "$(printf '░%.0s' $(seq 1 $empty))"
}

get_gpu_info() {
  rocm-smi --showuse --showmemuse --showtemp --showpower --json \
    | jq -r '.card0 | {
        edge: ."Temperature (Sensor edge) (C)",
        junction: ."Temperature (Sensor junction) (C)",
        memtemp: ."Temperature (Sensor memory) (C)",
        power: ."Average Graphics Package Power (W)",
        gpu: ."GPU use (%)",
        vram: ."GPU Memory Allocated (VRAM%)"
      }'
}

print_line() {
  local text="$1"
  printf "│ %-*s │\n" "$width" "$text"
}

draw_tui() {
  clear
  info=$(get_gpu_info)

  gpu=$(echo "$info" | jq -r '.gpu')
  vram=$(echo "$info" | jq -r '.vram')
  edge=$(echo "$info" | jq -r '.edge')
  junction=$(echo "$info" | jq -r '.junction')
  memtemp=$(echo "$info" | jq -r '.memtemp')
  power=$(echo "$info" | jq -r '.power')

  echo "+$(printf -- '-%.0s' $(seq 1 $((width+2))))+"
  print_line "GPU0 (RX 7800 XT)"
  print_line "Temp: Edge ${edge}C / Junction ${junction}C"
  print_line "VRAM: ${vram}%  $(bar $vram)"
  print_line "GPU : ${gpu}%  $(bar $gpu)"
  print_line "Power: ${power}W"
  echo "+$(printf -- '-%.0s' $(seq 1 $((width+2))))+"
}

while true; do
  draw_tui
  sleep 1
done

