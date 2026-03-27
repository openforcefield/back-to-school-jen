# 6_run_fit: Force Field Fitting Runs

Bond and angle parameter refit using SMEE and the UCI SPICE dataset (`3_split_train_test/full_split_uci`).
All fits use Adam optimizer with `amsgrad=True`. See `fit_data.py` for full training details.

Unless noted otherwise, bond parameter limits are `k ≥ 0`, `length ≥ 0`; angle limits are `k ≥ 0`, `angle ∈ [0, π]`.

---

## Summary Table

| Directory | Base OFFXML | SMIRKS | Fit | LR | Epochs | Scaling | Val Set | Ref | Minibatch |
|---|---|---|---|---|---|---|---|---|---|
| `0_openff_2.3-rc2` | `openff_unconstrained-2.3.0-rc2` | Stock OpenFF-2.3.0-rc2 | Bonds + Angles | 0.015 | 300 | mean | ✗ | mean | ✗ |
| `a_broad_specification` | `a_broad_specification/openff-2.3.0-ring-no-bond` | Broad element+ring (atom primitive) | Bonds | 0.005 | 40 | mean | ✓ | mean | ✓ |
| `b_no_bond_with_ring_in_atom` | `b_no_bond_with_ring_in_atom/openff-2.2.1-ring-no-bond` | Element connectivity + ring, no bond type | Bonds + Angles | 0.03 | 300 | mean | ✗ | mean | ✗ |
| `b2_no_bond_with_ring_in_atom` | `b_no_bond_with_ring_in_atom/openff-2.2.1-ring-no-bond` | Same as b | Bonds + Angles | 0.02 | 300 | fixed (`1e-2`/`1.0`) | ✗ | mean | ✗ |
| `b3_no_bond_with_ring_in_atom` | `b_no_bond_with_ring_in_atom/openff-2.2.1-ring-no-bond` | Same as b | Bonds + Angles | 0.005 | 400 | mean | ✗ | mean | ✗ |
| `c_bond_generalized` | `c_no_bond_ring_recursive/openff-2.3.0-fit-bond_ring-no-bond-type` | Recursive ring SMIRKS, no bond type | Bonds | 0.005 | 40 | mean | ✓ | mean | ✓ |
| `c2_bond_generalized_min` | `c_no_bond_ring_recursive/openff-2.3.0-fit-bond_ring-no-bond-type` | Same as c | Bonds | 0.005 | 40 | mean | ✓ | **min** | ✓ |
| `d_gen_bond_recursive` | — | — | — | — | — | — | — | — | ✓ |
| `d2_gen_bond_recursive` | `d_gen_bond_recursive/openff-2.3.0-gen_bond_atom_rec1` | Generalized bonds, recursive ring SMIRKS | Bonds | 0.005 | 500 | mean | ✓ | mean | ✓ |

> `d_gen_bond_recursive` is incomplete; only the output `final-force-field.offxml` is preserved (no run scripts).

**Note:** `d2_gen_bond_recursive` differs from all others: bond `k` lower bound is **90** (not 0), and uses minibatching.

---

## OFFXML Sources (`4_make_offxmls/`)
| Label | OFFXML file |
|---|---|
| a | `a_broad_specification/openff-2.3.0-ring-no-bond.offxml` |
| b / b2 / b3 | `b_no_bond_with_ring_in_atom/openff-2.2.1-ring-no-bond.offxml` |
| c / c2 | `c_no_bond_ring_recursive/openff-2.3.0-fit-bond_ring-no-bond-type.offxml` |
| d / d2 | `d_gen_bond_recursive/openff-2.3.0-gen_bond_atom_rec1.offxml` |

## Topology Sources (`5_setup_train_ff_topologies/`)
| Label | Topology directory |
|---|---|
| 0 | `2025_09_11_uci_full_split` |
| a | `2026_02_20_uci_off_a` |
| b / b2 / b3 | `2025_12_08_uci_off_b` |
| c / c2 | `2026_02_03_uci_c` |
| d / d2 | `uci_off_d` |

---

## Shared Scripts
- `fit_data.py` — main training script
- `compare_checkpoints.py` — compare parameter evolution across checkpoints
- `convert_pt_to_offxml.py` — convert `.pt` checkpoint to `.offxml`
