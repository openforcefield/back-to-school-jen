#!/bin/bash
# Generate bond parameter plots for selected force fields from 6_run_fit.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FF_BASE="$SCRIPT_DIR/../6_run_fit"

for ff in \
    "$FF_BASE/a_broad_specification/final-force-field.offxml" \
    "$FF_BASE/b3_no_bond_with_ring_in_atom/final-force-field.offxml" \
    "$FF_BASE/b3_no_bond_with_ring_in_atom/final-force-field.offxml" \
    "$FF_BASE/d_gen_bond_recursive/final-force-field.offxml" \
    "$FF_BASE/c_bond_generalized/final-force-field.offxml"; do

    name="$(basename "$(dirname "$ff")")"
    python "$SCRIPT_DIR/get_k_values.py" \
        --force-field "$ff" \
        --output "outputs/${name}_k_values.png"

done
