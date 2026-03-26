#!/bin/bash

#SBATCH --job-name=MakeInterchanges  ## job name
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

date
hn=`hostname`
echo "Running job on host $hn"

source ~/.bashrc
mamba activate bts

# Setup and save topologies for fitting
python -u ../setup_train_ff_topo.py --data-dir "../../2_filtered_results/uci/raw-spice-95.0thpercentile" \
                                    --offxml "/dfs9/dmobley-lab/jclark9/descent/back-to-school-jen/4_make_offxmls/e_broad_spec_recursion1/openff-2.3.0-bondtypes-recursion.offxml" \
                                    --file-format "pkl" \
                                    --device "cpu" \
			            --n-cpus $SLURM_CPUS_ON_NODE \
                                    --precision "single" 2>&1 | tee log.txt
