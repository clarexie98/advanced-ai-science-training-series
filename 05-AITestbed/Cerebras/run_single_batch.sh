#!/bin/bash

# Check if batch size argument provided
if [ $# -eq 0 ]; then
    echo "Usage: ./run_single_batch.sh <batch_size>"
    echo "Example: ./run_single_batch.sh 256"
    echo ""
    echo "Recommended batch sizes to test: 256, 512, 2048, 4096"
    echo "Note: Batch size 1024 already completed"
    exit 1
fi

BS=$1
HOMEWORK_DIR="homework_batch_comparison"

# Warn if trying to run 1024 again
if [ "$BS" -eq 1024 ]; then
    echo "WARNING: Batch size 1024 was already completed in your first run!"
    echo "Do you want to run it again? (y/n)"
    read -r response
    if [[ ! "$response" =~ ^[Yy]$ ]]; then
        echo "Skipping batch_size=1024"
        exit 0
    fi
fi

cd ~/R_2.6.0/modelzoo/src/cerebras/modelzoo/models/nlp/llama

# Activate virtual environment
source ~/R_2.6.0/venv_cerebras_pt/bin/activate

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
