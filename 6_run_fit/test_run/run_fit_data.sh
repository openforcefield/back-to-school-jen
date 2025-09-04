#!/bin/bash

# Fit filtered training data
python fit_data.py --data-dir "../../3_split_train_test/data-train" \
                   --filename-forcefield "../../4_setup_train_ff_topologies/smee_force_field.pkl" \
                   --filename-topo-dict "../../4_setup_train_ff_topologies/smee_topology_dict.pkl" \
                   --offxml "/Users/jenniferclark/bin/sage-2.2.1/openff-2.2.1.offxml" \
                   --n-epochs 2 \
                   --learning-rate 0.1 2>&1 | tee log.txt
