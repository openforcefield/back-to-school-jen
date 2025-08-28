#!/bin/bash

# Download and Process SPICE2 Dataset from Zenodo
mkdir -p data
python tasks/filter_data.py --data-dir "data/raw-spice"

# From output Finlay's cutoff of everything in the 95th percentile
# corresponds to a Z-score of 1, with RMS Force of ~32 kcal/mol
