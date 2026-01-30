#!/bin/bash

#SBATCH --job-name=EquCheck  ## job name
#SBATCH -p standard              ## use free partition
#SBATCH -t 1-00:00:00
#SBATCH --nodes=1            ## use 1 node, don't ask for multiple
#SBATCH --ntasks 1
#SBATCH --cpus-per-task=32
#SBATCH --account DMOBLEY_LAB
#SBATCH --mem-per-cpu=4G     ## ask for 1Gb memory per CPU
#SBATCH --constraint=fastscratch
#SBATCH -o stdout.txt
#SBATCH -e stderr.txt

export OMP_NUM_THREADS=1
N_PROCESSES=${SLURM_CPUS_PER_TASK:-1}

date
hn=`hostname`
echo "Running job on host $hn"
echo "JobID: ${SLURM_JOB_ID}"
echo "Running with ${N_PROCESSES} processes"

source ~/.bashrc
mamba activate bts

# Run analysis
python -u ../../6a_compare_equil/check_equil_values.py \
    --offxml "../../4_make_offxmls/b_no_bond_with_ring_in_atom/openff-2.2.1-ring-no-bond.offxml" \
    --dataset-paths "../../1_data/uci/raw-spice" \
                    "../../1_data/uci_qca_sage/qca_sage_data" \
    --dataset-labels "SPICE2" "Sage 2.3.0 Train" \
    --output-dir "equilibrium_value_analysis" \
    --n-processes ${N_PROCESSES} 2>&1 | tee check_equil_log.txt

date
