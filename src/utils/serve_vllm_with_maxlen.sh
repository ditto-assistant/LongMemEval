#!/usr/bin/env bash

export PYTHONIOENCODING=utf-8;
HOME_DIR=`realpath ..`

GPU=$1
MODEL=$2
MAXLEN=$3
PORT=${4:-8001}
TP_SIZE=${5:-1}
HOST=${VLLM_HOST:-127.0.0.1}

positive_integer() {
    case "$1" in ''|*[!0-9]*) return 1 ;; esac
    [ "${#1}" -le 9 ] && [ "$1" -gt 0 ]
}
if ! positive_integer "$PORT" || [ "${#PORT}" -gt 5 ] || [ "$PORT" -gt 65535 ] || ! positive_integer "$TP_SIZE" || ! positive_integer "$MAXLEN"; then
    echo "PORT must be 1-65535, and TP_SIZE and MAXLEN must be positive integers." >&2
    exit 2
fi

case "$HOST" in
    localhost) HOST=127.0.0.1 ;;
    127.0.0.1|::1) ;;
    *) if [ -z "${VLLM_API_KEY:-}" ]; then
        echo "A non-loopback VLLM_HOST requires VLLM_API_KEY and network access controls." >&2
        exit 2
    fi ;;
esac

export CUDA_VISIBLE_DEVICES=${GPU}

declare -A MODEL_ZOO
MODEL_ZOO["mistral-7b-instruct-v0.1"]="mistralai/Mistral-7B-Instruct-v0.1"
MODEL_ZOO["mistral-7b-instruct-v0.2"]="mistralai/Mistral-7B-Instruct-v0.2"
MODEL_ZOO["mistral-7b-instruct-v0.3"]="mistralai/Mistral-7B-Instruct-v0.3"
MODEL_ZOO["mistral-8x7b-instruct-v0.1"]="mistralai/Mixtral-8x7B-Instruct-v0.1"
MODEL_ZOO["mistral-8x22b-instruct-v0.1"]="mistralai/Mixtral-8x22B-Instruct-v0.1"
MODEL_ZOO["llama-3-8b-instruct"]="meta-llama/Meta-Llama-3-8B-Instruct"
MODEL_ZOO["llama-3-70b-instruct"]="meta-llama/Meta-Llama-3-70B-Instruct"
MODEL_ZOO["llama-3.1-8b-instruct"]="meta-llama/Meta-Llama-3.1-8B-Instruct"
MODEL_ZOO["llama-3.1-70b-instruct"]="meta-llama/Meta-Llama-3.1-70B-Instruct"
MODEL_ZOO["film-7b"]="In2Training/FILM-7B"
MODEL_ZOO["phi-3-medium-128k-instruct"]="microsoft/Phi-3-medium-128k-instruct"
MODEL_ZOO["phi-3.5-mini-instruct"]="microsoft/Phi-3.5-mini-instruct"
MODEL_ZOO["phi-4"]="microsoft/phi-4"
model_name=${MODEL_ZOO["$MODEL"]}

VLLM_API_KEY="${VLLM_API_KEY:-}" python -m vllm.entrypoints.openai.api_server \
       --host "${HOST}" \
       --model "${model_name}" \
       --max-model-len "${MAXLEN}" \
       --tensor-parallel-size "${TP_SIZE}" \
       --download-dir "${HOME_DIR}/model_cache/" \
       --port "${PORT}"
