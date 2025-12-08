#!/bin/bash

python -u ../setup_train_ff_topo.py --data-dir "../../3_split_train_test/full_split/data-train" \
                                    --offxml "/Users/jenniferclark/bin/sage-2.2.1/openff-2.2.1.offxml" \
                                    --file-format "pkl" \
                                    --device "cuda" \
			                        --n-cpus 1 \
                                    --precision "single" 2>&1 | tee log.txt
