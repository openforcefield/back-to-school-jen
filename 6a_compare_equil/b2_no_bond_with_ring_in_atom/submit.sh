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
python -u ../check_equil_values.py \
    --offxml "../../6_run_fit/b2_no_bond_with_ring_in_atom/final-force-field.offxml" \
    --dataset-paths "../../3_split_train_test/full_split_uci/data-train" \
                    "../../3_split_train_test/full_split_uci/data-test" \
                    "../../1_data/uci_qca_sage/qca_sage_data" \
    --dataset-labels "Training" "Validation" "Sage 2.3.0 Train" \
    --output-dir "equilibrium_value_analysis" \
    --n-processes ${N_PROCESSES} 2>&1 | tee check_equil_log.txt

echo "Analysis complete!"
date
