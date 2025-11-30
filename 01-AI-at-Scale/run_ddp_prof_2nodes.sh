#!/bin/bash -x

tstamp() {
     date +"%Y-%m-%d-%H%M%S"
}

# Get actual number of nodes from PBS
NNODES=`wc -l < $PBS_NODEFILE`
NRANKS_PER_NODE=1  # 1 rank per node

let NRANKS=${NNODES}*${NRANKS_PER_NODE}

EPOCHS=10

TRACE_DIR_ROOT=./traces/pytorch_2p8
TRACE_DIR=${TRACE_DIR_ROOT}/cuda_pt_2p8_${NNODES}nodes_E${EPOCHS}_R${NRANKS_PER_NODE}_$(tstamp)

module use /soft/modulefiles
module load conda/2025-09-25
conda activate  # Don't deactivate first

export DISABLE_PYMODULE_LOG=1

# CPU affinity for single GPU (GPU 0) - only cores 0,1
export CPU_AFFINITY="verbose,list:0,1"

mpiexec -n ${NRANKS} -ppn ${NRANKS_PER_NODE} -l --line-buffer --cpu-bind ${CPU_AFFINITY} \
    python pytorch_2p8_ddp_prof.py --epochs ${EPOCHS} --trace-dir ${TRACE_DIR}

echo "Profiling complete. Trace files saved to: ${TRACE_DIR}"
echo ""
echo "To download to your local machine, run:"
echo "rsync -avhHSP YOUR_USERNAME@polaris.alcf.anl.gov:~/2025/advanced-ai-science-training-series/01-AI-at-Scale/${TRACE_DIR}/*.json /path/on/your/laptop/"