#!/bin/bash

# Download and Process SPICE2 Dataset from Zenodo
python filter_data.py --data-dir "../1_data/raw-spice"

# From output Finlay's cutoff of everything in the 95th percentile
# corresponds to a Z-score of 1, with RMS Force of ~32 kcal/mol
