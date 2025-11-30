#!/bin/bash -x

NNODES=`wc -l < $PBS_NODEFILE`
NRANKS_PER_NODE=4
let NRANKS=$((NNODES*NRANKS_PER_NODE))

module use /soft/modulefiles
module load conda/2025-09-25
conda activate

export DISABLE_PYMODULE_LOG=1
export CPU_AFFINITY="verbose,list:0,1:8,9:16,17:24,25"

TRACE_DIR=./traces/io_comparison/run_$(date +"%Y%m%d_%H%M%S")

mpiexec -n ${NRANKS} -ppn ${NRANKS_PER_NODE} -l --line-buffer --cpu-bind ${CPU_AFFINITY} \
    python pytorch_ddp_io_comparison.py --epochs 5 --batch-size 32 --trace-dir ${TRACE_DIR}

echo "Results saved to: ${TRACE_DIR}"