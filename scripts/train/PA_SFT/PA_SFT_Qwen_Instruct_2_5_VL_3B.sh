export PYTHONPATH=$(pwd)/train/stage_sft

export GLOO_SOCKET_IFNAME=bond0
export NCCL_SOCKET_IFNAME=eno8303
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_IB_CUDA_SUPPORT=1
export NCCL_IB_GID_INDEX=3
export OMP_NUM_THREADS=4
export NCCL_IB_HCA=mlx5_0,mlx5_1

export WANDB_MODE=offline
export ACCELERATE_CPU_AFFINITY=1
# The image paths of your dataset
export IMAGE_DIR=/mnt/nfs/lyh/Project/IAD-R1/Expert-AD
# Pretrain model path of Qwen_Intruct_2_5_VL_3B
export PRETRAIN_MODEL_PATH=/mnt/nfs/lyh/Project/IAD-R1/final_model/Pretrain/Qwen_Intruct_2_5_VL_3B
export OUTPUT_PATH=/mnt/nfs/lyh/Project/IAD-R1/final_model/PA_SFT/Qwen_Intruct_2_5_VL_3B_Expert_AD_PA_SFT
export DATASET=Expert_AD_Stage_1

if [ ! -d "$OUTPUT_PATH" ]; then
  mkdir "$OUTPUT_PATH"
fi

torchrun --nproc_per_node=4 --nnodes=1 --master_port=12345 \
  train/stage_sft/train.py \
  --deepspeed scripts/train/zero3.json \
  --stage sft \
  --do_train \
  --model_name_or_path $PRETRAIN_MODEL_PATH \
  --dataset $DATASET \
  --image_dir $IMAGE_DIR \
  --template qwen2_vl \
  --finetuning_type full \
  --output_dir $OUTPUT_PATH \
  --overwrite_cache \
  --overwrite_output_dir \
  --warmup_steps 100 \
  --weight_decay 0.1 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 2 \
  --ddp_timeout 90000 \
  --learning_rate 1e-5 \
  --lr_scheduler_type cosine \
  --logging_steps 5 \
  --cutoff_len 4096 \
  --save_steps 365 \
  --plot_loss \
  --num_train_epochs 1 \
  --bf16 \
  2>&1 | tee ${OUTPUT_DIR}/train.log