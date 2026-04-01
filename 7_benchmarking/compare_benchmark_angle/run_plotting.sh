#!/bin/bash

python ../plot_benchmark.py \
    --metrics \
        ../0a_openff_2.3.0-rc2/benchmark_results/metrics.json \
        ../0b_openff_2.3-rc2/benchmark_results/metrics.json \
        ../0b_openff_2.3.0_smee_smirks/benchmark_results/metrics.json \
	    ../c_bond_generalized/benchmark_results/metrics.json \
        ../b3_no_bond_with_ring_in_atom/benchmark_results/metrics.json \
    --labels \
        "Sage 2.3.0" \
        "Sage 2.3.0 SMEE-SPICE Bonds" \
        "Sage 2.3.0 SMEE-SPICE Bonds/Angles" \
        "ff-c: Gen Bonds" \
        "ff-b: Gen Bonds / Angles" \
    --param-json \
        ../0a_openff_2.3.0-rc2/benchmark_results/metrics_by_parameter_type.json \
        ../0b_openff_2.3-rc2/benchmark_results/metrics_by_parameter_type.json \
        ../0b_openff_2.3.0_smee_smirks/benchmark_results/metrics_by_parameter_type.json \
        ../c_bond_generalized/benchmark_results/metrics_by_parameter_type.json \
        ../b3_no_bond_with_ring_in_atom/benchmark_results/metrics_by_parameter_type.json \
    --offxml \
        ../../forcefields/openff_unconstrained-2.3.0-rc2.offxml \
        ../../6_run_fit/0b_openff_2.3-rc2/final-force-field.offxml \
        ../../6_run_fit/0_openff_2.3-rc2/final-force-field.offxml \
        ../../6_run_fit/c_bond_generalized/final-force-field.offxml \
        ../../6_run_fit/b3_no_bond_with_ring_in_atom/final-force-field.offxml \
    --output-dir . \
    -vv 2>&1 | tee plotting_log.txt

python ../plot_benchmark.py \
    --metrics \
        ../0a_openff_2.3.0-rc2/benchmark_results/metrics.json \
        ../0b_openff_2.3-rc2/benchmark_results/metrics.json \
        ../0b_openff_2.3.0_smee_smirks/benchmark_results/metrics.json \
	    ../c_bond_generalized/benchmark_results/metrics.json \
        ../b3_no_bond_with_ring_in_atom/benchmark_results/metrics.json \
    --labels \
        "Sage 2.3.0" \
        "Sage 2.3.0 SMEE-SPICE Bonds" \
        "Sage 2.3.0 SMEE-SPICE Bonds/Angles" \
        "ff-c: Gen Bonds" \
        "ff-b: Gen Bonds / Angles" \
    --param-json \
        ../0a_openff_2.3.0-rc2/benchmark_results/metrics_by_parameter_type.json \
        ../0b_openff_2.3-rc2/benchmark_results/metrics_by_parameter_type.json \
        ../0b_openff_2.3.0_smee_smirks/benchmark_results/metrics_by_parameter_type.json \
        ../c_bond_generalized/benchmark_results/metrics_by_parameter_type.json \
        ../b3_no_bond_with_ring_in_atom/benchmark_results/metrics_by_parameter_type.json \
    --offxml \
        ../../forcefields/openff_unconstrained-2.3.0-rc2.offxml \
        ../../6_run_fit/0b_openff_2.3-rc2/final-force-field.offxml \
        ../../6_run_fit/0_openff_2.3-rc2/final-force-field.offxml \
        ../../6_run_fit/c_bond_generalized/final-force-field.offxml \
        ../../6_run_fit/b3_no_bond_with_ring_in_atom/final-force-field.offxml \
    --output-dir . \
    --plot-difference \
    -vv 2>&1 | tee plotting_log_diff.txt
