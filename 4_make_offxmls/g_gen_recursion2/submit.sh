#!/bin/bash

#SBATCH --job-name=ff-g  ## job name
#SBATCH -p standard              ## use free partition
#SBATCH -t 1-00:00:00
#SBATCH --nodes=1            ## use 1 node, don't ask for multiple
#SBATCH --ntasks 1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4G
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
mamba activate bts

# Fit filtered training data
python ../make_offxml.py --data-dir "../../3_split_train_test/full_split_uci/data-train" \
                         --specificity-json "specificity.json" \
			 -n $ncpus \
                         --filename-offxml-in "../../forcefields/openff_unconstrained-2.3.0-rc2.offxml" \
                         --filename-offxml-out "openff-2.3.0-bondanglegen-recursion2.offxml" \
                         --filename-test-train-smiles "../../3_split_train_test/full_split_uci/smiles_test_train.json" \
                         -vvv 2>&1 | tee log.txt
