#!/bin/bash -l
#PBS -A ALCFAITP
#PBS -l select=1
#PBS -N ml_baseline
#PBS -l walltime=0:30:00
#PBS -l filesystems=home:eagle
#PBS -k doe
#PBS -j oe
#PBS -l place=scatter
#PBS -q debug

cd $PBS_O_WORKDIR

module use /soft/modulefiles
module load conda/2025-09-25
conda activate /eagle/ALCFAITP/03-Coupling-Sim-AI/_ai4s_simAI
export TMPDIR=/tmp

echo "=========================================="
echo "Experiment 1: Baseline Configuration"
echo "initial_training_count=8, max_training_count=24, batch_size=4"
echo "=========================================="

# Create a temporary modified script
cat > 3_ml_in_the_loop_exp1.py << 'EOF'
from asyncio import new_event_loop
from parsl_config import polaris_config
from chemfunctions import compute_vertical, train_model, run_model
from matplotlib import pyplot as plt
import parsl
from parsl.app.app import python_app
from time import monotonic
from random import sample
import pandas as pd
import numpy as np
from concurrent.futures import as_completed
from pathlib import Path
import random
import sys

seed = 42
np.random.seed(seed)
random.seed(seed)

# Experiment 1: Baseline
initial_training_count = 8
max_training_count = 24
batch_size = 4

if initial_training_count >= max_training_count:
    print("Must do at least 1 active trianing iteration.")
    sys.exit(1)

compute_vertical_app = python_app(compute_vertical)
train_model_app = python_app(train_model)
inference_app = python_app(run_model)

@python_app
def combine_inferences(inputs=[]):
    import pandas as pd
    return pd.concat(inputs, ignore_index=True)

search_space = pd.read_csv('./data/QM9-search.tsv', sep=r'\s+')
search_space_size = len(search_space)

if __name__ == "__main__":
    train_data = []
    with parsl.load(polaris_config):
        start_time = monotonic()
        print(f"Will collect a maximum of {max_training_count} training samples for training.")
        print(f"Will run {batch_size} new simulations in each loop iteration to refine the model.\n")
        
        print(f"Creating initial training data composed of {initial_training_count}/{search_space_size} random molecules")
        train_data = []
        init_mols = search_space.sample(initial_training_count)['smiles']
        sim_futures = [compute_vertical_app(mol) for mol in init_mols]
        print(f'Submitted {len(sim_futures)} simulations for initial training ...')
        already_ran = set()
        
        while len(sim_futures) > 0:
            future = next(as_completed(sim_futures))
            sim_futures.remove(future)
            smiles = future.task_record['args'][0]
            already_ran.add(smiles)
            
            if future.exception() is not None:
                smiles = search_space.sample(1).iloc[0]['smiles']
                new_future = compute_vertical_app(smiles)
                sim_futures.append(new_future)
            else:
                train_data.append({
                    'smiles': smiles,
                    'ie': future.result(),
                    'batch': 0,
                    'time': monotonic() - start_time
                })
        
        print("Training data collected!\n")
        print("Starting active learning loop\n")
        
        train_data_df = pd.DataFrame(train_data)
        iter_no = 0
        
        while len(train_data) < max_training_count:
            iter_no += 1
            iter_start = monotonic()
            
            print(f'Iteration {iter_no}:')
            print(f'\tTraining on {len(train_data)}/{search_space_size} random molecules')
            
            train_future = train_model_app(train_data_df)
            
            chunks = np.array_split(search_space, 256)
            inference_futures = [inference_app(train_future, chunk) for chunk in chunks]
            infer_future = combine_inferences(inputs=inference_futures)
            predictions = infer_future.result()
            
            best_list = predictions.nlargest(batch_size * 2, 'ie')
            
            new_sim_inputs = []
            for smiles in best_list['smiles']:
                if smiles not in already_ran:
                    new_sim_inputs.append(smiles)
                if len(new_sim_inputs) >= batch_size:
                    break
            
            best_molecule = best_list.iloc[0]['smiles']
            best_pred_ie = best_list.iloc[0]['ie']
            print(f'\tBest predicted molecule: {best_molecule} with ionization energy {best_pred_ie:.2f} Ha')
            
            new_sim_futures = [compute_vertical_app(s) for s in new_sim_inputs]
            
            mre_score = 0
            for future, smiles in zip(as_completed(new_sim_futures), new_sim_inputs):
                already_ran.add(smiles)
                
                true_ie = future.result()
                pred_ie = predictions.query(f'smiles=="{smiles}"')['ie'].iloc[0]
                mre_score += abs((true_ie - pred_ie) / true_ie)
                
                train_data.append({
                    'smiles': smiles,
                    'ie': true_ie,
                    'batch': iter_no,
                    'time': monotonic() - start_time
                })
            
            train_data_df = pd.DataFrame(train_data)
            print(f'\tPerformed {len(new_sim_inputs)} new simulations')
            print(f'\tEstimate of KNN Model Mean Relative Error (MRE): {mre_score / len(new_sim_inputs):.2f} %')
            print(f'\tFinished loop iteration in {monotonic() - iter_start:.2f}s\n')
        
        total_time = monotonic() - start_time
        print(f'Training completed in {total_time:.2f} seconds\n')
        
        print("Plotting results...")
        train_data = pd.DataFrame(train_data)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        batch_colors = {0: 'blue'}
        for i in range(1, iter_no + 1):
            batch_colors[i] = plt.cm.viridis(i / iter_no)
        
        for batch in train_data['batch'].unique():
            batch_data = train_data[train_data['batch'] == batch]
            label = 'Initial Training' if batch == 0 else f'Iteration {batch}'
            ax1.scatter(batch_data.index, batch_data['ie'], 
                       label=label, color=batch_colors[batch], s=50, alpha=0.7)
        
        ax1.set_xlabel('Simulation Number')
        ax1.set_ylabel('Ionization Energy (Ha)')
        ax1.set_title('Ionization Energy by Simulation Number')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        for batch in train_data['batch'].unique():
            batch_data = train_data[train_data['batch'] == batch]
            label = 'Initial Training' if batch == 0 else f'Iteration {batch}'
            ax2.scatter(batch_data['time'], batch_data['ie'],
                       label=label, color=batch_colors[batch], s=50, alpha=0.7)
        
        ax2.set_xlabel('Time (seconds)')
        ax2.set_ylabel('Ionization Energy (Ha)')
        ax2.set_title('Ionization Energy Over Time')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('parsl_ml_in_the_loop_exp1.png', dpi=150, bbox_inches='tight')
        print(f"Plot saved as 'parsl_ml_in_the_loop_exp1.png'")
        print("All done!")
EOF

python 3_ml_in_the_loop_exp1.py

# Save the output plot with experiment name
cp parsl_ml_in_the_loop_exp1.png parsl_ml_in_the_loop.png

echo "Experiment 1 complete!"
