#!/bin/bash

python ../compare_ff_params.py \
    --handler "both" \
    --offxml "../../9_MSM_bond-angle_predictions/g_gen_recursion2/openff-ff-g-msm.offxml" \
             "../../forcefields/openff_unconstrained-2.3.0-rc2.offxml" \
    --labels "Gen Rec2 MSM" "Sage 2.3.0" \
    --alpha 0.6 1.0  \
    --marker-size 7 20 \
    --output-dir ./plots 2>&1 | tee log.txt
