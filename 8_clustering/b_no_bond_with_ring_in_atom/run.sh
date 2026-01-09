#!/bin/bash

python ../clustering.py --offxml "../../6_run_fit/b_no_bond_with_ring_in_atom/final-force-field.offxml" \
                        --output-dir ./clusters 2>&1 | tee log.txt
