#!/bin/bash
export HSA_OVERRIDE_GFX_VERSION=11.0.0
export HIP_VISIBLE_DEVICES=0
export AMD_SERIALIZE_KERNEL=3

python -m infer.chat_loop \
  --model checkpoints/ft_latest_step.pt \
  --tokenizer tokenizer/tokenizer.model \
  --max_new_tokens 120 \
  --temperature 0.8 \
  --top_k 50

