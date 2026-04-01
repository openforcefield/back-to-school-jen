#!/bin/bash

python ../clustering.py --offxml "../../6_run_fit/0b_openff_2.3-rc2/final-force-field.offxml" \
                        --output-dir ./clusters 2>&1 | tee log.txt
