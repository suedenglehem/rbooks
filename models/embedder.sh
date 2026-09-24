#!/bin/sh

m="/dd2/lmstudio/Models/xtra/bce-embedding-base_v1-Q8_0.gguf"

port=8081

case $1 in
  -p|--port) shift; port=$1 ;;
esac

CUDA_VISIBLE_DEVICES=2 \
/dd2/andrei/bench/build/bin/llama-server -m ./bge-m3-q8_0.gguf --port $port --embedding -c 8192 --batch-size 8192 --ubatch-size 8192 --n-gpu-layers 99
