#!/bin/bash

# Fit filtered training data
python make_offxml.py --data-dir "../3_split_train_test/data-train" \
                      --filename-offxml-in "/Users/jenniferclark/bin/sage-2.2.1/openff-2.2.1.offxml" \
                      --filename-offxml-out "openff-2.2.1-new.offxml" \
                      --filename-test-train-smiles "../3_split_train_test/smiles_test_train.json" 2>&1 | tee log.txt
#                      --datasets "OpenFF Additional Generated ChEMBL Optimizations v4.0" "OpenFF Additional Generated ChEMBL Optimizations v4.0" \\
#                      --datasets-type singlepoint
