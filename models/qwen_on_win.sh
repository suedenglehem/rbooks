#!/bin/sh

m="./Qwen3.8-27B-Uncensored-Q6_K.gguf"

ctx=142768

./build/bin/llama-server.exe  \
    --model $m \
    --alias qwen3.8-27b,local-metrics \
    --host 0.0.0.0 \
    --port  8080 \
    --split-mode tensor \
    --tensor-split 1.6,1.6 \
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
    --spec-type draft-mtp --spec-draft-n-max 5 
