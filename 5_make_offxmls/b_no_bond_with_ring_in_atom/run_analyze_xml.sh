#!/bin/bash

# Fit filtered training data
python ../make_offxml.py --data-dir "../../3_split_train_test/test_split/data-train" \
                         --specificity-json "specificity.json" \
                         --filename-offxml-in "../../forcefields/openff_unconstrained-2.2.1.offxml" \
                         --filename-offxml-out "openff-2.2.1-ring-no-bond.offxml" \
                         --filename-test-train-smiles "../../3_split_train_test/test_split/smiles_test_train.json" \
                         -vv \
                         --n-workers 3 \
                         --datasets "OpenFF Industry Benchmark Season 1 v1.2" \
                         --datasets-type optimization 2>&1 | tee log.txt
