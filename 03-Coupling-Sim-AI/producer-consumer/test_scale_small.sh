#!/bin/bash -l
#PBS -A ALCFAITP
#PBS -l select=1
#PBS -N pc_scale_small
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:eagle
#PBS -k doe
#PBS -j oe
#PBS -l place=scatter
#PBS -q debug-scaling

cd $PBS_O_WORKDIR

module use /soft/modulefiles
module load conda/2025-09-25
conda activate /eagle/ALCFAITP/03-Coupling-Sim-AI/_ai4s_simAI
export TMPDIR=/tmp
export PATH=$PATH:/opt/pbs/bin

# Small scale test: 32 simulations, 256x256 grid (smaller than baseline)
NUM_SIMS=32
GRID_SIZE=256

echo "=========================================="
echo "Scaling Test: 1 node, 32 sims, 256x256 grid"
echo "=========================================="

echo "Running with Parsl writing to the file system"
python 6_parsl_fs_producer_consumer.py --grid_size $GRID_SIZE --num_sims $NUM_SIMS
echo

echo "Running with DragonHPC + DDict"
dragon 8_dragon_producer_consumer.py --grid_size $GRID_SIZE --num_sims $NUM_SIMS
echo

echo "Running with Parsl transferring data through futures"
python 5_parsl_fut_producer_consumer.py --grid_size $GRID_SIZE --num_sims $NUM_SIMS
echo

echo "Test complete!"
