#!/bin/bash

export CUDA_VISIBLE_DEVICES=2

### C3VD

# Define the base path for your sequences
base_path="../dataset/C3VD"

# List of sequences
sequences=(
    "cecum_t1_b"
    "cecum_t2_b"
    "cecum_t3_a"
    "sigmoid_t1_a"
    "sigmoid_t2_a"
    "sigmoid_t3_a"
    "trans_t1_b"
    "trans_t2_c"
    "trans_t4_a"
    "trans_t4_b"
)

# Loop through each sequence and run the commands
for seq in "${sequences[@]}"; do
    sequence_path="$base_path/$seq"
    model_path="output_test-0327-2/$seq"
    echo "Processing sequence: $sequence_path with model path: $model_path"

    # --- Train ---
    python train.py --model_path "$model_path" --eval -s "$sequence_path" \
        --iteration 40000 \
        --disable_viewer \
        --use_specular_branch \
        --spec_schedule staged \
        --spec_start_iter 3000 \
        --staged_gauss_lr_scale 0.1 \
        --base_mask_alpha 0.9 \
        --base_mask_clamp_min 0.1 \
        --lambda_sparse 0.0 \
        --lambda_prior 0.05 \
        --spec_prior_gamma 1.0 \
        --spec_vis_interval 2000

    # --- Render ---
    python render.py --model_path "$model_path" --eval -s "$sequence_path" \
        --iteration 40000 \
        --use_specular_branch

done

# compute mean metric values based on logs
python metrics.py "output_test"
