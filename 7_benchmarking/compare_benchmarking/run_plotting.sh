#!/bin/bash

python ../plot_benchmark.py \
    --metrics \
        ../0a_openff_2.3.0-rc2/benchmark_results/metrics.json \
        ../0b_openff_2.3-rc2/benchmark_results/metrics.json \
        ../a_broad_specification/benchmark_results/metrics.json \
	    ../c_bond_generalized/benchmark_results/metrics.json \
	    ../d_gen_bond_recursive/benchmark_results/metrics.json \
	    ../e_broad_spec_recursion1/benchmark_results/metrics.json \
    --labels \
        "Sage 2.3.0" \
        "Sage 2.3.0 SMEE-SPICE" \
        "ff-a: Specific Bonds" \
        "ff-c: Gen Bonds" \
        "ff-d: Gen Bonds Recursive" \
        "ff-e: Specific Bonds Recursive" \
    --param-json \
        ../0a_openff_2.3.0-rc2/benchmark_results/metrics_by_parameter_type.json \
        ../0b_openff_2.3-rc2/benchmark_results/metrics_by_parameter_type.json \
        ../a_broad_specification/benchmark_results/metrics_by_parameter_type.json \
        ../c_bond_generalized/benchmark_results/metrics_by_parameter_type.json \
        ../d_gen_bond_recursive/benchmark_results/metrics_by_parameter_type.json \
        ../e_broad_spec_recursion1/benchmark_results/metrics_by_parameter_type.json \
    --offxml \
        ../../forcefields/openff_unconstrained-2.3.0-rc2.offxml \
        ../../6_run_fit/0b_openff_2.3-rc2/final-force-field.offxml \
	    ../../6_run_fit/a_broad_specification/final-force-field.offxml \
        ../../6_run_fit/c_bond_generalized/final-force-field.offxml \
        ../../6_run_fit/d_gen_bond_recursive/final-force-field.offxml \
        ../../6_run_fit/e_broad_spec_recursion1/final-force-field.offxml \
    --output-dir . \
    -vv 2>&1 | tee plotting_log.txt

python ../plot_benchmark.py \
    --metrics \
        ../0a_openff_2.3.0-rc2/benchmark_results/metrics.json \
        ../0b_openff_2.3-rc2/benchmark_results/metrics.json \
        ../a_broad_specification/benchmark_results/metrics.json \
	    ../c_bond_generalized/benchmark_results/metrics.json \
	    ../d_gen_bond_recursive/benchmark_results/metrics.json \
	    ../e_broad_spec_recursion1/benchmark_results/metrics.json \
    --labels \
        "Sage 2.3.0" \
        "Sage 2.3.0 SMEE-SPICE" \
        "ff-a: Specific Bonds" \
        "ff-c: Gen Bonds" \
        "ff-d: Gen Bonds Recursive" \
        "ff-e: Specific Bonds Recursive" \
    --param-json \
        ../0a_openff_2.3.0-rc2/benchmark_results/metrics_by_parameter_type.json \
        ../0b_openff_2.3-rc2/benchmark_results/metrics_by_parameter_type.json \
        ../a_broad_specification/benchmark_results/metrics_by_parameter_type.json \
        ../c_bond_generalized/benchmark_results/metrics_by_parameter_type.json \
        ../d_gen_bond_recursive/benchmark_results/metrics_by_parameter_type.json \
        ../e_broad_spec_recursion1/benchmark_results/metrics_by_parameter_type.json \
    --offxml \
        ../../forcefields/openff_unconstrained-2.3.0-rc2.offxml \
        ../../6_run_fit/0b_openff_2.3-rc2/final-force-field.offxml \
        ../a_broad_specification/benchmark_results/metrics.json \
        ../../6_run_fit/c_bond_generalized/final-force-field.offxml \
        ../../6_run_fit/d_gen_bond_recursive/final-force-field.offxml \
        ../../6_run_fit/e_broad_spec_recursion1/final-force-field.offxml \
    --output-dir . \
    --plot-difference \
    -vv 2>&1 | tee plotting_log_diff.txt
