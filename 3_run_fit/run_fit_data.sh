#!/bin/bash

# Fit filtered data
mkdir -p data
python fit_data.py --data-dir "../2_filtered_results/raw-spice-95thpercentile" \
                   --offxml "/Users/jenniferclark/bin/sage-2.2.1/openff-2.2.1.offxml" \
                   --n-epochs 2 \
                   --learning-rate 0.1 2>&1 | tee log.txt
