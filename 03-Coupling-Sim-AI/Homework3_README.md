# Homework 3: Coupling Simulation and AI

1. Tune the parameters of the ML-in-the-loop active learning workflow in order to find molecules with the largest ionization energy in the shortest possible time. Use the PBS submit script [4_submit_multinode.sh](./ml-in-the-loop/4_submit_multinode.sh) to run the full workflow on 1 or multiple nodes of Polaris. Note that all that should be needed for this exercise is to change the values of the `initial_training_count`, `max_training_count` and `batch_size` variables at the top of the `3_ml_in_the_loop.py` script. Submit the plot that is produced by the script as well as your code to showcase your results and how you obtained them.


### Experiments Conducted

I tested the following parameter combinations:

| Experiment | initial_training_count | max_training_count | batch_size | Best IE (Ha) | Total Time (s) | Iterations | Result |
|------------|------------------------|--------------------|-----------|--------------:|---------------:|-----------:|--------|
| Exp 1: Baseline | 8 | 24 | 4 | **15.33** | 28.83 | 4 | ✓ Found high-IE molecule |
| Exp 2: Large batch | 12 | 32 | 8 | 14.36 | 23.10 | 3 | ✗ Got stuck in local optimum |
| Exp 3: Balanced | 12 | 32 | 4 | **15.36** | 29.80 | 6 | ✓ Best IE found |
| Exp 4: Balanced (repeat) | 12 | 32 | 4 | **15.36** | 30.02 | 6 | ✓ Consistent results |
| Exp 5: Less initial | 4 | 24 | 4 | 15.13 | 29.06 | 5 | ~ Ok but not optimal |
| Exp 6: Small batch | 8 | 32 | 2 | 14.40 | 53.62 | 14 | ✗ Too many iterations, slow |
| Exp 7: Moderate | 10 | 30 | 4 | 15.28 | 32.65 | 5 | ✓ Ok but not optimal |

**Best Configuration Found:**
- `initial_training_count`: **12**
- `max_training_count`: **32**  
- `batch_size`: **4**
- **Best ionization energy found:** **15.36 Ha** (molecule: NCC(F)(F)F)
- **Total time:** **29.80-30.02 seconds**

![ML-in-the-loop results](ml-in-the-loop/parsl_ml_in_the_loop.png)

**Thoughts:**

Optimal configuration seems to balances pre-training and iterations, i.e balance between exploration quality and computational efficiency. Too few iterations hurt performance. 

---



2. Experiment with the Parsl and DragonHPC implementations of the producer-consumer workflow by scaling up the problem size, both in terms of size of the data being produced/transferred and in terms of the number of nodes (although 2-4 nodes will be sufficient). Collect data from these experiments to fill in the table under the `Data Transfer Performance (Homework)` section at the bottom of the [example README](./producer-consumer/README.md) and write a short paragraph about your observations. See the notes under the `Data Transfer Performance (Homework)` section for more detailed information and some hints. 


**Scaling experiments:**
- Vary number of simulations (`--num_sims`): 32, 64, 128
- Vary grid size (`--grid_size`): 256, 512, 1024
- Test on 1-2 nodes

### Results

| Implementation   | Number of Nodes | Training Data Size (GB) | Simulation Run / IO Time (sec) | Training Run / IO Time (sec) |
|------------------|-----------------|-------------------------|-------------------------------|------------------------------|
| Parsl + futures | 1 | 0.62 | 14.38 / NA | 26.59 / NA |
| Parsl + file system | 1 | 0.62 | 11.22 / 0.094 | 14.90 / 0.422 |
| DragonHPC + DDict | 1 | 0.62 | 7.01 / 0.233 | 17.92 / 1.194 |
| Parsl + futures | 1 | 0.16 | 11.41 / NA | 14.29 / NA |
| Parsl + file system | 1 | 0.16 | 12.37 / 0.032 | 81.64 / 0.212 |
| DragonHPC + DDict | 1 | 0.16 | 6.54 / 0.014 | 37.73 / 0.680 |
| Parsl + futures | 1 | 0.16 | 11.49 / NA | 13.67 / NA |
| Parsl + file system | 1 | 0.16 | 11.09 / 0.033 | 83.77 / 0.217 |
| DragonHPC + DDict | 1 | 0.16 | 6.52 / 0.017 | 35.06 / 0.628 |
| Parsl + futures | 1 | 5.00 | 33.49 / NA | 170.83 / NA |
| Parsl + file system | 1 | 5.00 | 18.93 / 0.376 | 103.93 / 1.678 |
| DragonHPC + DDict | 1 | 5.00 | 13.37 / 0.418 | 47.17 / 3.150 |
| Parsl + futures | 2 | 1.25 | 14.20 / NA | 82.31 / NA |
| Parsl + file system | 2 | 1.25 | 15.56 / 0.119 | 115.30 / 2.246 |
| DragonHPC + DDict | 2 | 1.25 | 6.78 / 0.142 | 39.87 / 2.127 |

**Observations**

The experimental results reveal performance differences between the three data transfer approaches. For small data (0.16 GB), Parsl with futures performs best for training (14.29s) despite the serialization overhead, while DragonHPC excels at simulations (6.54s). However, as data size increases to 5 GB, DragonHPC + DDict proved to perform best with 47.17s training time compared to 103.93s for file system and 170.83s for futures, demonstrating that in-memory RDMA transfers scale far better than serialization or file I/O. On multi-node runs (2 nodes, 1.25 GB), DragonHPC + DDict still perform best (39.87s training), showing that RDMA-enabled distributed memory provides efficient inter-node data sharing. This matches expectations that in-memory approaches outperform disk-based solutions for large-scale data transfers.


