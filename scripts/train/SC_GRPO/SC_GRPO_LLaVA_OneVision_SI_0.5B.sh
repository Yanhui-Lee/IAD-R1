#!/bin/bash
export PYTHONPATH=$(pwd)/train/stage_rl
export GLOO_SOCKET_IFNAME=eno8303
export NCCL_SOCKET_IFNAME=eno8303
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_IB_CUDA_SUPPORT=1
export NCCL_IB_GID_INDEX=3
export OMP_NUM_THREADS=4
export NCCL_IB_HCA=mlx5_0,mlx5_1

# Wandb
export WANDB_MODE=disabled
export WANDB_BASE_URL=https://api.wandb.ai
export WANDB_PROJECT=IAD-R1
export WANDB_API_KEY="Your wandb api key"
export WANDB_RUN_NAME=LLaVA_OneVision_SI_0.5B_Expert_AD_SC_GRPO-$(date +%Y-%m-%d-%H-%M-%S)
wandb login $WANDB_API_KEY

# Dataset
export DATASET_NAME=/mnt/nfs/lyh/Project/IAD-R1/data/SC-GRPO/Expert_AD_Stage_2.json
export IMAGE_PATH=/mnt/nfs/lyh/Project/IAD-R1/Expert-AD

# PA-SFT model path of LLaVA_OneVision_SI_0.5B
export MODEL_NAME_OR_PATH=/mnt/nfs/lyh/Project/IAD-R1/final_model/PA_SFT/LLaVA_OneVision_SI_0.5B_Expert_AD_PA_SFT
export OUTPUT_DIR=/mnt/nfs/lyh/Project/IAD-R1/final_model/SC-GRPO/LLaVA_OneVision_SI_0.5B_Expert_AD_SC_GRPO

if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir "$OUTPUT_DIR"
fi

# Debug
export DEBUG_MODE="True"
export LOG_PATH=${OUTPUT_DIR}/reward.log

# Perform GPU
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Run
torchrun --nproc_per_node=3 --nnodes=1 --master_port=12345 \
  train/stage_rl/grpo_ad.py \
  --deepspeed scripts/train/zero3.json \
  --output_dir ${OUTPUT_DIR} \
  --model_name_or_path ${MODEL_NAME_OR_PATH} \
  --dataset_name ${DATASET_NAME} \
  --image_path ${IMAGE_PATH} \
  --use_vllm_for_gen true \
  --use_system_prompt false \
  --max_prompt_length 8192 \
  --max_completion_length 512 \
  --num_generations 4 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 2 \
  --logging_steps 1 \
  --bf16 \
  --report_to wandb \
  --gradient_checkpointing true \
  --attn_implementation flash_attention_2 \
  --max_pixels 480000 \
  --save_steps 100 \
  --num_train_epochs 1 \
  --run_name ${WANDB_RUN_NAME} \
  --single_img 1 \
  2>&1 | tee ${OUTPUT_DIR}/train.log