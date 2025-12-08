#!/bin/bash

# Fit filtered training data
python ../make_offxml.py --data-dir "../../3_split_train_test/test_split/data-train" \
                        --specificity-json "specificity.json" \
                        -n 2 \
                        --filename-offxml-in "../../forcefields/openff_unconstrained-2.3.0-rc2.offxml" \
                        --filename-offxml-out "openff-2.2.1-broad-spec.offxml" \
                        --filename-test-train-smiles "../../3_split_train_test/test_split/smiles_test_train.json" \
                        -vv \
                        --datasets "OpenFF Industry Benchmark Season 1 v1.2" \
                        --datasets-type optimization 2>&1 | tee log.txt
