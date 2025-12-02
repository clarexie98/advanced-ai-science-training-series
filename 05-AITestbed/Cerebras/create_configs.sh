#!/bin/bash

# Base config
BASE_CONFIG="../configs/params_llama2_7b.yaml"

# Batch sizes to test (excluding 1024 which is already done)
BATCH_SIZES=(256 512 2048 4096)

for BS in "${BATCH_SIZES[@]}"; do
    # Copy base config
    cp $BASE_CONFIG params_llama2_7b_bs${BS}.yaml
    
    # Replace batch_size using sed
    sed -i "s/batch_size: [0-9]*/batch_size: ${BS}/" params_llama2_7b_bs${BS}.yaml
    
    echo "Created config for batch_size=${BS}"
done

echo "All config files created!"
echo "Note: Batch size 1024 already completed, configs created for: 256, 512, 2048, 4096"
