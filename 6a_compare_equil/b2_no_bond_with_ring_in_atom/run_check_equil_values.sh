#!/bin/bash

export OMP_NUM_THREADS=1
N_PROCESSES=${SLURM_CPUS_PER_TASK:-1}

echo "Running with ${N_PROCESSES} processes"

# Run analysis
python -u ../check_equil_values.py \
    --offxml "../../6_run_fit/b2_no_bond_with_ring_in_atom/final-force-field.offxml" \
    --train-data "../../3_split_train_test/full_split_uci/data-train" \
    --val-data "../../3_split_train_test/full_split_uci/data-test" \
    --output-dir "equilibrium_value_analysis" \
    --n-processes ${N_PROCESSES} 2>&1 | tee check_equil_log.txt
