#!/bin/bash

set -e

# Mirror mv_recon style: edit variables below then just `bash run.sh`.

workdir='.'
model_name='vggt'
model_weights="/root/autodl-tmp/models/model_tracker_fixed_e20.pt"
dataset_root="/root/autodl-tmp/data/neural_rgbd_data"
size=224
device="cuda:0"
delta=1

output_dir="${workdir}/eval_results/pose/${model_name}/fp16"
echo "Output dir: $output_dir"
mkdir -p "$output_dir"

python -u launch.py \
  --weights "$model_weights" \
  --dataset_root "$dataset_root" \
  --output_dir "$output_dir" \
  --model_name "$model_name" \
  --size "$size" \
  --device "$device" \
  --delta "$delta"
