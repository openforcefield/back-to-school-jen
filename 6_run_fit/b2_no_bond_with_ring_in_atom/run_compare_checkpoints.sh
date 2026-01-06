#!/bin/bash

python ../compare_checkpoints.py \
    --checkpoint-dir my-smee-fit/ \
    --top-n 10 \
    --plot \
    --output-dir checkpoint_analysis/ 2>&1 | tee compare_checkpoints_log.txt
