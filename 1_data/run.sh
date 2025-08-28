#!/bin/bash

# Download and Process SPICE2 Dataset from Zenodo
mkdir -p data
python get_data_spice2.py --data-dir "."
#python get_data_spice2.py --data-dir "/dfs9/dmobley-lab/openff-bts"

# Download and Process QCArchive Datasets
## Singlpoint
#python get_data_qca.py --datasets "OpenFF CX3-CX4 singlepoints v4.0" \
#                       --dataset_type singlepoint \
#                       --data_file ./data/singlepoint

## Optimization
#python get_data_qca.py --datasets "OpenFF Cresset Additional Coverage Optimizations v4.0" \
#                       --dataset_type optimization \
#                       --data_file ./data/optimization

## Torsiondrive
#python get_data_qca.py --datasets "OpenFF Additional Generated Guanidine Amidine Derivative and O-Linker TorsionDrives 4.0" \
#                       --dataset_type torsiondrive \
#                       --data_file ./data/torsion_data
