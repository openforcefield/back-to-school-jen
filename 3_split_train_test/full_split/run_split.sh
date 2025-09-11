#!/bin/bash

# Split data
python ../split_train_test_data.py --data-dir "../../2_filtered_results/local/raw-spice-95.0thpercentile" 2>&1 | tee log.txt
