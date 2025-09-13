#!/bin/bash

#SBATCH --job-name=MakeInterchanges  ## job name
#SBATCH -p standard              ## use free partition
#SBATCH -t 1-00:00:00
#SBATCH --nodes=1            ## use 1 node, don't ask for multiple
#SBATCH --ntasks 1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=5G
#SBATCH --constraint=fastscratch
#SBATCH -o stdout_multi.txt
#SBATCH -e stderr_multi.txt

date
hn=`hostname`
echo "Running job on host $hn"
ncpus=$SLURM_CPUS_ON_NODE
echo "$ncpus allocated CPUs"

source ~/.bashrc
micromamba activate descent-workflow

# Setup and save topologies for fitting
python -u ../setup_train_ff_topo.py --data-dir "../../3_split_train_test/full_split_uci/data-train" \
                                    --offxml "../../forcefields/openff_unconstrained-2.3.0-rc1.offxml" \
                                    --file-format "pkl" \
                                    --device "cpu" \
			            --n-cpus $SLURM_CPUS_ON_NODE \
                                    --precision "single" 2>&1 | tee log_multi.txt
