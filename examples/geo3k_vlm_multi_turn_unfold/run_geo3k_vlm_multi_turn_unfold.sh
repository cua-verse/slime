#!/bin/bash

# Multi-turn unfold VLM RL training on geo3k dataset
# Think-stripping: each turn's context hides previous turns' <think> blocks.
# SGLang IndexError fix: uses "text" payload instead of "input_ids".
# Usage:
#   SLIME_SCRIPT_TRAIN_BACKEND=megatron ./run_geo3k_vlm_multi_turn_unfold.sh
#   SLIME_SCRIPT_MODEL_NAME=Qwen3-VL-8B-Thinking ./run_geo3k_vlm_multi_turn_unfold.sh

# Configuration
TRAIN_BACKEND=${SLIME_SCRIPT_TRAIN_BACKEND:-"megatron"}
MODEL_NAME=${SLIME_SCRIPT_MODEL_NAME:-"Qwen3-VL-4B-Thinking"}
NUM_GPUS=${SLIME_SCRIPT_NUM_GPUS:-4}
DATASET_NAME="VeraIsHere/geo3k_imgurl_processed"
DATA_ROOT="/root/datasets/geo3k_imgurl_processed"

# Validate MODEL_NAME
VALID_MODELS="
  Qwen3-VL-2B-Instruct
  Qwen3-VL-4B-Instruct
  Qwen3-VL-8B-Instruct
  Qwen3-VL-2B-Thinking
  Qwen3-VL-4B-Thinking
  Qwen3-VL-8B-Thinking
"
if ! echo "$VALID_MODELS" | grep -qw "$MODEL_NAME"; then
   echo "Error: MODEL_NAME must be one of: $VALID_MODELS"
   exit 1
fi

# External Ray flag
if [ -z "$SLIME_SCRIPT_EXTERNAL_RAY" ] || [ "$SLIME_SCRIPT_EXTERNAL_RAY" = "0" ]; then
   USE_EXTERNAL_RAY=0
else
   USE_EXTERNAL_RAY=1
fi

# Cleanup
pkill -9 sglang
sleep 3
if [ "$USE_EXTERNAL_RAY" = "0" ]; then
   ray stop --force
   pkill -9 ray
fi
pkill -9 slime
sleep 3
if [ "$USE_EXTERNAL_RAY" = "0" ]; then
   pkill -9 ray
fi
pkill -9 slime
pkill -9 redis

DUMP_DIR="/tmp/slime_debug/$(date +%Y%m%d_%H%M%S)"

set -ex

export PYTHONBUFFERED=16

# Detect NVLink
NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
   HAS_NVLINK=1
else
   HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

# Download model and dataset
mkdir -p /root/models /root/datasets
if [ ! -d "/root/models/${MODEL_NAME}" ]; then
   hf download Qwen/${MODEL_NAME} --local-dir /root/models/${MODEL_NAME}
fi
if [ ! -d "${DATA_ROOT}" ]; then
   hf download --repo-type dataset ${DATASET_NAME} --local-dir ${DATA_ROOT}
fi

# Common args
CKPT_ARGS=(
   --hf-checkpoint /root/models/${MODEL_NAME}
)

ROLLOUT_ARGS=(
   --prompt-data ${DATA_ROOT}/train.parquet
   --input-key problem
   --label-key answer
   --apply-chat-template
   --rollout-shuffle
   --rm-type math
   --num-rollout 3000
   --rollout-batch-size 64
   --n-samples-per-prompt 8
   --rollout-max-response-len 4096
   --rollout-temperature 1
   --global-batch-size 512
   --custom-generate-function-path examples.geo3k_vlm_multi_turn_unfold.rollout.generate
   --custom-convert-samples-to-train-data-path examples.geo3k_vlm_multi_turn_unfold.rollout.convert_samples_to_train_data
   --custom-config-path examples/geo3k_vlm_multi_turn_unfold/geo3k_vlm_multi_turn_unfold_config.yaml
)

MULTIMODAL_KEYS='{"image": "images"}'

GRPO_ARGS=(
   --advantage-estimator grpo
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --kl-coef 0.00
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 2
   --sglang-mem-fraction-static 0.5
   --sglang-cuda-graph-bs 1 2 4 8 $(seq 16 8 128)
)

# Wandb args (only if WANDB_API_KEY is set)
if [ -n "$WANDB_API_KEY" ]; then
   WANDB_ARGS=(
      --use-wandb
      --wandb-project slime-dev
      --wandb-group geo3k_vlm_multi_turn_unfold
      --wandb-key ${WANDB_API_KEY}
      --disable-wandb-random-suffix
   )
else
   WANDB_ARGS=()
fi

MISC_ARGS=(
   --actor-num-nodes 1
   --actor-num-gpus-per-node ${NUM_GPUS}
   --rollout-num-gpus ${NUM_GPUS}
   --colocate
)

# Backend-specific args
if [ "$TRAIN_BACKEND" = "fsdp" ]; then
   BACKEND_ARGS=(
      --train-backend fsdp
      --gradient-checkpointing
      --sglang-attention-backend fa3
      --attn-implementation flash_attention_3
      --update-weight-buffer-size 536870912
   )
   MODEL_ARGS=()
else
   # megatron backend
   BACKEND_ARGS=(
      --train-backend megatron
      --load /root/models/${MODEL_NAME}
      --tensor-model-parallel-size 2
      --sequence-parallel
      --pipeline-model-parallel-size 1
      --context-parallel-size 1
      --expert-model-parallel-size 1
      --expert-tensor-parallel-size 1
      --recompute-granularity full
      --recompute-method uniform
      --recompute-num-layers 1
      --use-dynamic-batch-size
      --max-tokens-per-gpu 2048
      --attention-dropout 0.0
      --hidden-dropout 0.0
      --accumulate-allreduce-grads-in-fp32
      --attention-softmax-in-fp32
      --attention-backend flash
      --megatron-to-hf-mode bridge
   )

   # get MODEL_ARGS from scripts/models for megatron backend
   SLIME_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." &>/dev/null && pwd)"
   MODEL_ARGS_FILE=$(echo "$MODEL_NAME" | sed 's/-Instruct//g; s/-Thinking//g; s/Qwen3-VL-/qwen3-/g; s/-2B/-1.7B/g')
   MODEL_ARGS_ROTARY_BASE=5000000 source "${SLIME_DIR}/scripts/models/${MODEL_ARGS_FILE}.sh"
fi

# Start Ray if not using external Ray
if [ "$USE_EXTERNAL_RAY" = "0" ]; then
   export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
   export no_proxy="127.0.0.1,${MASTER_ADDR}"
   ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus ${NUM_GPUS} --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265
fi

# Build runtime env
RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\"
  }
}"

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train.py \
   --multimodal-keys "${MULTIMODAL_KEYS}" \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${BACKEND_ARGS[@]} \
   ${MISC_ARGS[@]} \
   --dump-details ${DUMP_DIR}
