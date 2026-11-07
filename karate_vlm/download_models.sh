#!/bin/bash
V=/d/Users/karate-training/vlm; cd $V
get(){ # url dest
  for i in 1 2 3; do curl -L --fail -C - -s -o "$2" "$1" && { echo "OK $(basename "$2") $(du -m "$2" | cut -f1)MB"; return; }; sleep 5; done; echo "FAIL $2"; }
get https://github.com/ggml-org/llama.cpp/releases/download/b11538/llama-b11538-bin-win-cuda-12.4-x64.zip llama/llama.zip
get https://github.com/ggml-org/llama.cpp/releases/download/b11538/cudart-llama-bin-win-cuda-12.4-x64.zip llama/cudart.zip
(cd llama && unzip -oq llama.zip && unzip -oq cudart.zip && rm -f llama.zip cudart.zip && echo "OK llama.cpp unzipped")
HF=https://huggingface.co
get $HF/Qwen/Qwen3-VL-4B-Instruct-GGUF/resolve/main/Qwen3VL-4B-Instruct-Q4_K_M.gguf models/qwen3vl-4b.gguf
get $HF/Qwen/Qwen3-VL-4B-Instruct-GGUF/resolve/main/mmproj-Qwen3VL-4B-Instruct-Q8_0.gguf models/qwen3vl-4b-mmproj.gguf
get $HF/unsloth/Qwen3.5-4B-GGUF/resolve/main/Qwen3.5-4B-Q4_K_M.gguf models/qwen35-4b.gguf
get $HF/unsloth/Qwen3.5-4B-GGUF/resolve/main/mmproj-F16.gguf models/qwen35-4b-mmproj.gguf
get $HF/ggml-org/gemma-3-4b-it-GGUF/resolve/main/gemma-3-4b-it-Q4_K_M.gguf models/gemma3-4b.gguf
get $HF/ggml-org/gemma-3-4b-it-GGUF/resolve/main/mmproj-model-f16.gguf models/gemma3-4b-mmproj.gguf
echo ALL DONE
