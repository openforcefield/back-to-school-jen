#!/bin/bash

#SBATCH --job-name=a_ff  ## job name
#SBATCH --account DMOBLEY_LAB_GPU
#SBATCH -p gpu              ## use free partition
#SBATCH -t 4-00:00:00
#SBATCH --nodes=1            ## use 1 node, don't ask for multiple
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=16gb                  # request 16Gb of memory
#SBATCH --constraint=fastscratch
#SBATCH --export ALL
#SBATCH -o stdout_multi.txt
#SBATCH -e stderr_multi.txt

date
hn=$(hostname)
echo "Running job on host $hn"
ncpus=$SLURM_CPUS_ON_NODE
echo "$ncpus allocated CPUs"

nvcc --version
nvidia-smi

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

source ~/.bashrc
mamba activate bts
echo "$(which python)"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'Number of GPUs: {torch.cuda.device_count()}')"

python ../fit_data.py --data-dir "../../3_split_train_test/full_split_uci/data-train" \
                      --filename-forcefield "../../5_setup_train_ff_topologies/2025_12_08_uci_off_b/smee_force_field.pkl" \
                      --filename-topo-dict "../../5_setup_train_ff_topologies/2025_12_08_uci_off_b/smee_topology_dict.pkl" \
                      --offxml "../../4_make_offxmls/b_no_bond_with_ring_in_atom/openff-2.2.1-ring-no-bond.offxml" \
                      --n-epochs 300 \
		      --to-cuda true \
                      --learning-rate 0.1 2>&1 | tee log.txt
