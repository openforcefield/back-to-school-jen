#!/bin/bash

# Fit filtered data
mkdir -p data
python tasks/fit_data.py --data-dir "data/raw-spice_filtered-95thpercentile"
