#!/bin/bash

python ../compare_ff_params.py \
    --offxml "../../6_run_fit/e_broad_spec_recursion1/final-force-field.offxml" \
             "../../7_benchmarking/e2_broad_spec_recursion1_finlay/spice2_linearised_harmonics_minibatch_sage_bondtypes_recursion.offxml" \
             "../../9_MSM_bond-angle_predictions/e_broad_spec_recursion1/openff-ff-e-msm.offxml" \
             "../../forcefields/openff_unconstrained-2.3.0-rc2.offxml" \
    --labels "ff-e bonds only" "ff-e all valance" "ff-e MSM" "Sage 2.3.0" \
    --alpha 0.6 0.6 0.8 1.0  \
    --marker-size 7 7 7 20 \
    --output-dir ./plots_ff_e_with_sage 2>&1 | tee log_ff-e_sage_compare.txt

#python ../compare_ff_params.py \
#    --offxml "../../6_run_fit/e_broad_spec_recursion1/final-force-field.offxml" \
#             "../../7_benchmarking/e2_broad_spec_recursion1_finlay/spice2_linearised_harmonics_minibatch_sage_bondtypes_recursion.offxml" \
#             "openff-ff-e-msm.offxml" \
#    --labels "ff-e bonds only" "ff-e all valance" "ff-e MSM" \
#    --connect-lines \
#    --marker-size 7 \
#    --alpha 0.6 \
#    --output-dir ./plots_ff_e 2>&1 | tee log_ff-e_compare.txt
