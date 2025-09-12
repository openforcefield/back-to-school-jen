#!/bin/bash

#SBATCH --job-name=MakeInterchanges  ## job name
#SBATCH -p free              ## use free partition
#SBATCH -t 1-00:00:00
#SBATCH --nodes=1            ## use 1 node, don't ask for multiple
#SBATCH --ntasks 1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=4G     ## ask for 1Gb memory per CPU
#SBATCH --constraint=fastscratch
#SBATCH -o stdout.txt
#SBATCH -e stderr.txt

date
hn=`hostname`
echo "Running job on host $hn"
ncpus = $SLURM_CPUS_ON_NODE

source ~/.bashrc
micromamba activate descent-workflow

# Setup and save topologies for fitting
python -u ../setup_train_ff_topo.py --data-dir "../../3_split_train_test/full_split_uci/data-train" \
                              --offxml "/dfs9/dmobley-lab/jclark9/descent/openff-2.2.1.offxml" \
                              --file-format "pkl" \
                              --device "cuda" \
			      --n-cpus $SLURM_CPUS_ON_NODE \
                              --precision "single" 2>&1 | tee log.txt
