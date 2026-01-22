#!/bin/bash

# Pull record ids used to train Sage 2.3.0

######## Download and Process QCA Optimization Dataset ########
python ../get_data_qca.py --input_data "record_ids.txt" \
                          --dataset_type optimization \
                          --input_type record \
                          --data_file "./qca_sage_data" 2>&1 | tee log.txt
