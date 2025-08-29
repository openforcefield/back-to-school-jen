#!/bin/bash

# Fit filtered data
mkdir -p data
python fit_data.py --data-dir "../2_filtered_results/raw-spice_filtered-95thpercentile" 2>&1 | tee log.txt
