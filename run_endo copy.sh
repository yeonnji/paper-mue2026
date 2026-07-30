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

# Loop through each sequence and run the commands362
for seq in "${sequences[@]}"; do
    sequence_path="$base_path/$seq"
    model_path="output_test/$seq"  # Use sequence name as part of the model path
    echo "Processing sequence: $sequence_path with model path: $model_path"

    python train.py --model_path "$model_path" --eval  -s "$sequence_path" --iteration 40000 --use_specular_branch --spec_latent_dim 8 --spec_schedule joint --spec_vis_interval 1000
    python render.py --model_path "$model_path" --eval -s "$sequence_path" --iteration 40000 --use_specular_branch
    
done

# compute mean metric values based on logs
python metrics.py "output_test"