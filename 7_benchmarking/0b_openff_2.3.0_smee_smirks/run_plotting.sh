#!/bin/bash

python ../plot_benchmark.py \
    --metrics benchmark_results/metrics.json \
    --database benchmark.sqlite \
    --offxml "../../6_run_fit/0_openff_2.2.1/openff-2.2.1-smee-spice.offxml" \
    --output-dir benchmark_results \
    --n-processes 4 \
    -v 2>&1 | tee plotting_log.txt
