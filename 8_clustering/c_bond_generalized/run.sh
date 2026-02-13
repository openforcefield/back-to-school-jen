#!/bin/bash

python ../clustering.py --offxml "../../6_run_fit/c_bond_generalized/final-force-field.offxml" \
                        --output-dir ./clusters 2>&1 | tee log.txt
