#!/bin/sh

m="/dd2/lmstudio/Models/JonathanColetti/Qwen3.8-27B-Uncensored-GGUF/Qwen3.8-27B-Uncensored-noMTP-Q8_0.gguf"
d="/dd2/lmstudio/Models/JonathanColetti/Qwen3.8-27B-Uncensored-GGUF/Qwen3.8-27B-Uncensored-draft-Q8_0.gguf"

ctx=256000

CUDA_VISIBLE_DEVICES=0,1 \
/dd2/andrei/bench/build/bin/llama-server \
    --model $m \
    --alias qwen3.8-27b,local-metrics \
    --host 0.0.0.0 \
    --port  8091 \
    --split-mode tensor \
    --tensor-split 1,1 \
    --fit off \
    --n-gpu-layers 999 \
    --ctx-size $ctx \
    --batch-size 2048 \
    --ubatch-size 512 \
    --flash-attn on \
    --cache-type-k q8_0 \
    --cache-type-v q8_0 \
    --cache-prompt \
    --jinja \
    --reasoning-budget 4096 \
    --parallel 1 \
    --cont-batching \
    --metrics \
    --spec-type draft-mtp --spec-draft-n-max 5 --model-draft $d 
