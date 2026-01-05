#!/bin/bash

python ../compare_checkpoints.py \
    --checkpoint-first ../b_no_bond_with_ring_in_atom/final-force-field.offxml \
    --checkpoint-last ../b2_no_bond_with_ring_in_atom/final-force-field.offxml \
    --output-dir checkpoint_analysis/ 2>&1 | tee compare_checkpoints_log.txt
