#!/bin/bash

# Setup environment for SambaNova homework
echo "Setting up environment for SambaNova homework..."

# Activate conda environment
source ../../04-Inference-Workflows/Agentic-workflows/0_activate_env.sh

# Install datasets package
echo ""
echo "Installing HuggingFace datasets package..."
pip install --user datasets

echo ""
echo "✓ Setup complete! Now run:"
echo "  python compare_metis_sophia.py"
