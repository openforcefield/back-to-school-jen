#!/bin/bash

# Fit filtered training data
python a_make_offxml.py --data-dir "../../3_split_train_test/full_split/data-train" \
                      --filename-offxml-in "/Users/jenniferclark/bin/sage-2.2.1/openff-2.2.1.offxml" \
                      --filename-offxml-out "openff-2.2.1-broad-spec.offxml" \
                      --filename-test-train-smiles "../../3_split_train_test/full_split/smiles_test_train.json" \
                      -vv \
                      --datasets "OpenFF Industry Benchmark Season 1 v1.2" \
                      --datasets-type optimization 2>&1 | tee log.txt
