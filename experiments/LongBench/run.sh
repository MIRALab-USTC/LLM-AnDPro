#!/bin/bash

# Set PyTorch library path
export LD_LIBRARY_PATH=/root/miniconda3/envs/andpro/lib/python3.10/site-packages/torch/lib:$LD_LIBRARY_PATH

# ============================================
# Path Configuration (MODIFY THESE PATHS)
# ============================================
# Model path - Update to your model location
MODEL_PATH="/root/autodl-tmp/Mistral-7B-Instruct-v0.2"

# Dataset path - Update to your dataset location
DATASET_PATH="/root/autodl-tmp/LongBench"

# GPU device ID
GPU_ID=0

# ============================================
# Experiment Configuration
# ============================================
budget_scopes=(128)
chunk_size_scopes=(4)
Mode=(andpro)

for budget in ${budget_scopes[@]}; do
    for chunk_size in ${chunk_size_scopes[@]}; do
        for mode in ${Mode[@]}; do
            echo "budget: "$budget", chunk_size: "$chunk_size", mode: "$mode
        
            CUDA_VISIBLE_DEVICES=$GPU_ID python pred.py -m $MODEL_PATH \
                --max_length 64000 \
                --dataset $DATASET_PATH \
                --mode  $mode \
                --budget $budget \
                --chunk_size $chunk_size \
                --out_name "$mode"_Mistral-7B_"$budget"_chunk_"$chunk_size" \
                --compress_args_path c"$budget"_w32_k7_maxpool.json \
                --dataset_name_list "qasper"
        done
    done
done

python eval.py