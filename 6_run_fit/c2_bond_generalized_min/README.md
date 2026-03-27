# FF-C2: Generalized Bonds with Recursive Ring SMIRKS (descent-reference=min)

Bond SMIRKS use recursive ring detection (`r:1` notation) to encode ring membership without specifying bond type.
Source OFFXML: `4_make_offxmls/c_no_bond_ring_recursive/openff-2.3.0-fit-bond_ring-no-bond-type.offxml`.
Topology: `5_setup_train_ff_topologies/2026_02_03_uci_c`.

**Bonds only** (Angles not fit). Validation set used during training (`full_split_uci/data-test`).
**Minibatching**

Scaling is data-derived (mean-based) as defined in `fit_data.py`:
```python
PARAMETERS = {
    "Bonds": descent.train.ParameterConfig(
        cols=["k", "length"],
        scales={"k": 1/bond_k_mean, "length": 1/bond_x0_mean},
        limits={"k": [0.0, None], "length": [0.0, None]},
    ),
}
```

Scaling values for Bonds: {'k': 0.001889332660956697, 'length': 0.6332648961569184}

The process of taking the mean from the initial dataset is done in `fit_data.py`.

Learning Rate: 0.005
Number of Epochs: 40

Previous Step:
Identical to `c_bond_generalized` except uses `--descent-reference min` instead of the default `mean`.
The reference structure for computing the regression target is the per-molecule minimum-energy conformer
rather than the mean over conformers.

Conclusion:


Next Step:
