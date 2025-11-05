#!/bin/bash

#SBATCH --job-name=b_ff  ## job name
#SBATCH -p standard              ## use free partition
#SBATCH -t 3-00:00:00
#SBATCH --nodes=1            ## use 1 node, don't ask for multiple
#SBATCH --ntasks 1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4GB
#SBATCH --constraint=fastscratch
#SBATCH --account DMOBLEY_LAB
#SBATCH -o stdout_multi.txt
#SBATCH -e stderr_multi.txt

date
hn=`hostname`
echo "Running job on host $hn"
ncpus=$SLURM_CPUS_ON_NODE
echo "$ncpus allocated CPUs"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

source ~/.bashrc
micromamba activate bts

# Fit filtered training data
python ../make_offxml.py --data-dir "../../3_split_train_test/full_split_uci/data-train" \
                         --specificity-json "specificity.json" \
			 -n $ncpus \
                         --filename-offxml-in "../../forcefields/openff_unconstrained-2.3.0-rc2.offxml" \
                         --filename-offxml-out "openff-2.2.1-ring-no-bond.offxml" \
                         --filename-test-train-smiles "../../3_split_train_test/full_split_uci/smiles_test_train.json" \
                         -vvv \
                         --datasets "OpenFF Industry Benchmark Season 1 v1.2" \
                         --datasets-type optimization 2>&1 | tee log.txt
echo "\nRun complete"
date
