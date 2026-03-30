#!/bin/bash

python ../clustering.py --offxml "../../6_run_fit/e_broad_spec_recursion1/final-force-field.offxml" \
                        --output-dir ./clusters 2>&1 | tee log.txt
