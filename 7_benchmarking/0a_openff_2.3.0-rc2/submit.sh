#!/bin/bash

#SBATCH --job-name=Benchmark  ## job name
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

python ../benchmarking.py --offxml "../../forcefields/openff_unconstrained-2.3.0-rc2.offxml" \
                          --input-dataset "$HOME/bin/yammbs-dataset-submission/datasets/OpenFF-Industry-Benchmark-Season-1-v1.2/cache.json" \
                          --force-field-name "OpenFF-2.3.0-rc2" \
                          --output-dir benchmark_results \
                          --n-processes $SLURM_CPUS_ON_NODE \
                          -vvv 2>&1 | tee log.txt
