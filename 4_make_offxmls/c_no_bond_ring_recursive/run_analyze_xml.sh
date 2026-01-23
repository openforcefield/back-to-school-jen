#!/bin/bash

# Fit filtered training data
python ../make_offxml.py --data-dir "../../1_data/uci/raw-spice" \
                         --specificity-json "specificity.json" \
                         --filename-offxml-in "../../forcefields/openff_unconstrained-2.3.0-rc2.offxml" \
                         --filename-offxml-out "openff-2.3.0-ring-no-bond.offxml" \
                         --filename-test-train-smiles "../../3_split_train_test/test_split/smiles_test_train.json" \
                         -vv \
                         --n-workers 3 \
                         --cutoff-pop 1 \
                         --datasets "OpenFF Industry Benchmark Season 1 v1.2" \
                         --datasets-type optimization 2>&1 | tee log.txt
