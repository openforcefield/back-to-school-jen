#!/bin/bash

python ../compare_checkpoints.py \
    --checkpoint-first ../b3_no_bond_with_ring_in_atom/my-smee-fit/final-force-field.pt \
    --checkpoint-last ../b2_no_bond_with_ring_in_atom/my-smee-fit/final-force-field.pt \
    --output-dir checkpoint_analysis/ 2>&1 | tee compare_checkpoints_log.txt
