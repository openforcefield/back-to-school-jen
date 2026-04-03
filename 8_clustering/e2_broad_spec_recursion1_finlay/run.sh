#!/bin/bash

python ../clustering.py --offxml "../../7_benchmarking/e2_broad_spec_recursion1_finlay/spice2_linearised_harmonics_minibatch_sage_bondtypes_recursion.offxml" \
                        --output-dir ./clusters 2>&1 | tee log.txt
