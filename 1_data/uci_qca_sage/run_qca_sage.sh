#!/bin/bash

######## Download and Process QCA Optimization Dataset ########
python ../get_data_qca.py --datasets "OpenFF SMIRNOFF Sage 2.2.0" \
                          --dataset_type optimization \
                          --data_file "./qca_sage_data" 2>&1 | tee log.txt
