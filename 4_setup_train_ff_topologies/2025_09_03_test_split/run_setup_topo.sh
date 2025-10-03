#!/bin/bash

# Setup and save topologies for fitting
python ../setup_train_ff_topo.py --data-dir "../../3_split_train_test/test_split/data-train" \
                                 --offxml "../../forcefields/openff_unconstrained-2.2.1.offxml" \
                                 --file-format "pkl" \
                                 --device "cpu" \
                                 --n-cpus 3 \
                                 --precision "single" 2>&1 | tee log.txt
