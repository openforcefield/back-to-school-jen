#!/bin/bash

# Fit filtered training data
python ../make_offxml.py --data-dir "../../3_split_train_test/full_split_uci/data-train" \
                         --specificity-json "specificity.json" \
                         --filename-offxml-in "../../forcefields/openff_unconstrained-2.3.0-rc2.offxml" \
                         --filename-offxml-out "openff-2.3.0-gen_bond_atom_rec2.offxml" \
                         --filename-test-train-smiles "../../3_split_train_test/full_split_uci/smiles_test_train.json" \
                         -vvv \
                         --cutoff-population 1 \
                         --datasets "OpenFF Industry Benchmark Season 1 v1.2" \
                         --datasets-type optimization 2>&1 | tee log.txt
echo "\nRun complete"
date
