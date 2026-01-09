#!/bin/bash

python ../benchmarking.py --offxml "../../6_run_fit/0_openff_2.2.1/openff-2.2.1-smee-spice.offxml" \
                      --input-dataset "~/bin/yammbs-dataset-submission/datasets/OpenFF-Industry-Benchmark-Season-1-v1.2/cache.json" \
                      --force-field-name openff-2.2.1-smee-spice.json \
                      --output-dir benchmark_results \
                      -vvv 2>&1 | tee log.txt
