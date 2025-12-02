#!/bin/bash

# Array of batch sizes (excluding 1024 which is already done)
BATCH_SIZES=(256 512 2048 4096)

# Directory for this homework
HOMEWORK_DIR="homework_batch_comparison"

cd ~/R_2.6.0/modelzoo/src/cerebras/modelzoo/models/nlp/llama

# Activate virtual environment
source ~/R_2.6.0/venv_cerebras_pt/bin/activate

echo "Note: Skipping batch_size=1024 (already completed)"
echo ""

# Run experiments for each batch size
for BS in "${BATCH_SIZES[@]}"; do
    echo "=========================================="
    echo "Running experiment with batch_size=${BS}"
    echo "=========================================="
    
    # Set model directory
    export MODEL_DIR="${HOMEWORK_DIR}/model_dir_bs${BS}"
    
    # Remove old model directory if exists
    if [ -d "$MODEL_DIR" ]; then 
        rm -Rf $MODEL_DIR
    fi
    
    # Run training
    cszoo fit ${HOMEWORK_DIR}/params_llama2_7b_bs${BS}.yaml \
        --job_labels name=llama2_7b_bs${BS} \
        --model_dir $MODEL_DIR \
        |& tee ${HOMEWORK_DIR}/log_bs${BS}.txt
    
    echo ""
    echo "Completed batch_size=${BS}"
    echo "Log saved to: ${HOMEWORK_DIR}/log_bs${BS}.txt"
    echo ""
    
    # Wait a bit between jobs
    sleep 10
done

echo "All experiments completed!"
echo "Batch sizes tested: 256, 512, 2048, 4096 (plus previously completed 1024)"
