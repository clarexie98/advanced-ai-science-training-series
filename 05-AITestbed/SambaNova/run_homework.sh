#!/bin/bash
# Quick Start Script for SambaNova Homework
# This script helps you run the Metis vs Sophia comparison

set -e

echo "=========================================="
echo "SambaNova Homework: Metis vs Sophia"
echo "=========================================="
echo ""

# Check if we're in the right directory
if [ ! -f "compare_metis_sophia.py" ]; then
    echo "Error: Please run this script from the SambaNova directory"
    echo "cd /home/clarexie/2025/advanced-ai-science-training-series/05-AITestbed/SambaNova"
    exit 1
fi

# Check if inference_auth_token.py exists
if [ ! -f "../inference_auth_token.py" ]; then
    echo "Error: inference_auth_token.py not found in parent directory"
    echo "Please ensure the file exists at: ../inference_auth_token.py"
    exit 1
fi

echo "✓ Found compare_metis_sophia.py"
echo "✓ Found inference_auth_token.py"
echo ""

# Check token expiration
echo "Checking authentication token..."
cd ..
token_check=$(python inference_auth_token.py get_time_until_token_expiration --units hours 2>&1)

if echo "$token_check" | grep -q "expired\|error"; then
    echo "⚠️  Token expired or invalid. Re-authenticating..."
    python inference_auth_token.py authenticate --force
else
    echo "✓ Token is valid"
fi
cd SambaNova

echo ""
echo "=========================================="
echo "Starting Benchmark"
echo "=========================================="
echo ""
echo "This will test 5 prompts on both Metis and Sophia"
echo "Estimated time: 2-3 minutes"
echo ""
read -p "Press Enter to continue or Ctrl+C to cancel..."

# Run the benchmark
python compare_metis_sophia.py

echo ""
echo "=========================================="
echo "Benchmark Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Review the console output above"
echo "2. Check the JSON results file created"
echo "3. Fill in Homework5_Analysis.md with your findings"
echo "4. Run with more prompts by editing compare_metis_sophia.py line 269"
echo ""
echo "To commit results:"
echo "  cd /home/clarexie/2025/advanced-ai-science-training-series"
echo "  git add 05-AITestbed/"
echo "  git commit -m 'Add Homework 5: Metis vs Sophia comparison'"
echo "  git push origin main"
echo ""
