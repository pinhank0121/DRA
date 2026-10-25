#!/bin/bash
CONFIG_PATH=$1

GPUS_PER_NODE=${GPUS_PER_NODE:-8}
NNODES=${WORLD_SIZE:-1}
NODE_RANK=${RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-1235}
WORLD_SIZE=$(($GPUS_PER_NODE * $NNODES))
PRECISION=${PRECISION:-bf16}

# accelerate launch \
#     --main_process_ip $MASTER_ADDR \
#     --main_process_port $MASTER_PORT \
#     --machine_rank $NODE_RANK \
#     --num_processes  $(($GPUS_PER_NODE*$NNODES)) \
#     --num_machines $NNODES \
#     --mixed_precision $PRECISION \
#     extract_features.py \
#     --config $CONFIG_PATH \
#     --data_path data/imagenet/val \
#     --data_split imagenet_val \
#     --output_path data/imagenet_vae_latents_bf16 \
#     --image_size 256 \
#     --batch_size 20 \
#     --seed 42 \
#     --num_workers 8 \
#     --shard_size 100 \
#     --save_dtype bf16

# accelerate launch \
#     --main_process_ip $MASTER_ADDR \
#     --main_process_port $MASTER_PORT \
#     --machine_rank $NODE_RANK \
#     --num_processes  $(($GPUS_PER_NODE*$NNODES)) \
#     --num_machines $NNODES \
#     --mixed_precision $PRECISION \
#     extract_features.py \
#     --config $CONFIG_PATH \
#     --data_path data/imagenet/train \
#     --data_split imagenet_train \
#     --output_path data/imagenet_vae_latents_bf16 \
#     --image_size 256 \
#     --batch_size 20 \
#     --seed 42 \
#     --num_workers 8 \
#     --shard_size 100 \
#     --save_dtype bf16

accelerate launch \
  --main_process_ip $MASTER_ADDR \
  --main_process_port $MASTER_PORT \
  --machine_rank $NODE_RANK \
  --num_processes $(($GPUS_PER_NODE * $NNODES)) \
  --num_machines $NNODES \
  --mixed_precision $PRECISION \
  extract_features.py \
  --config $CONFIG_PATH \
  --data_path data/imagenet_ldit4M \
  --data_split imagenet_ldit4M \
  --output_path data/imagenet_vae_latents_bf16 \
  --image_size 256 \
  --batch_size 20 \
  --seed 42 \
  --num_workers 8 \
  --shard_size 100 \
  --save_dtype bf16
