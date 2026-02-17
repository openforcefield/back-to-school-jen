#!/bin/bash

export OMP_NUM_THREADS=1
N_PROCESSES=${SLURM_CPUS_PER_TASK:-1}

echo "Running with ${N_PROCESSES} processes"

# Run analysis
python -u ../check_equil_values.py \
    --offxml "../../6_run_fit/c_bond_generalized/final-force-field.offxml" \
    --dataset-paths "../../3_split_train_test/full_split_uci/data-train" \
                    "../../3_split_train_test/full_split_uci/data-test" \
                    "../../1_data/uci_qca_sage/qca_sage_data" \
    --dataset-labels "Training" "Validation" "Sage 2.3.0 Train" \
    --output-dir "equilibrium_value_analysis" \
    --n-processes ${N_PROCESSES} 2>&1 | tee check_equil_log.txt
