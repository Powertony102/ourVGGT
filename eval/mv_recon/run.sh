#!/bin/bash

set -e

workdir='.'
model_name='vggt'
model_weights="/root/autodl-tmp/models/model_tracker_fixed_e20.pt"

output_dir="${workdir}/eval_results/mv_recon/${model_name}/fastvggt/fp16"
echo "$output_dir"
accelerate launch --num_processes 8 --main_process_port 29501 launch.py \
    --weights "$model_weights" \
    --output_dir "$output_dir" \
    --model_name "$model_name" \
