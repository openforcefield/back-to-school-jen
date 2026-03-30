#!/bin/bash

python ../clustering.py --offxml "../../6_run_fit/d_gen_bond_recursive/final-force-field.offxml" \
                        --output-dir ./clusters 2>&1 | tee log.txt
