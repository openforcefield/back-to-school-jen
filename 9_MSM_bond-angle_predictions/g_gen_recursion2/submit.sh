#!/bin/bash

#SBATCH --job-name=MSM-g  ## job name
#SBATCH -p standard              ## use free partition
#SBATCH -t 7-00:00:00
#SBATCH --nodes=1            ## use 1 node, don't ask for multiple
#SBATCH --ntasks 1
#SBATCH --cpus-per-task=48
#SBATCH --account DMOBLEY_LAB
#SBATCH --mem-per-cpu=4G     ## ask for 1Gb memory per CPU
#SBATCH --constraint="intel&fastscratch"
#SBATCH -o stdout.txt
#SBATCH -e stderr.txt

date
hn=`hostname`
echo "Running job on host $hn"

source ~/.bashrc
mamba activate bts

python ../generate-msm-forcefield.py --input-forcefield ../../4_make_offxmls/g_gen_recursion2/openff-2.3.0-bondanglegen-recursion2.offxml \
                                     --output-forcefield openff-ff-g-msm.offxml \
                                     --output-msm msm-out.json \
                                     --msm-data-directory ../msm-data \
                                     --n-workers $SLURM_CPUS_ON_NODE 2>&1 | tee log.txt
