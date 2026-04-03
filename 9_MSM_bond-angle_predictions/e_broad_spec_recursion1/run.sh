#! /bin/bash

python ../generate-msm-forcefield.py --input-forcefield ../../4_make_offxmls/e_broad_spec_recursion1/openff-2.3.0-bondtypes-recursion.offxml \
                                  --output-forcefield openff-ff-e-msm.offxml \
                                  --output-msm msm-out.json \
                                  --msm-data-directory ../msm-data
