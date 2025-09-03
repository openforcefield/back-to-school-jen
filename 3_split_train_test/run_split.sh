#!/bin/bash

# Split data
python split_train_test_data.py --data-dir "../2_filtered_results/raw-spice-95thpercentile" \
                                --max-n-pts 100 \
                                --seed 42 2>&1 | tee log.txt
