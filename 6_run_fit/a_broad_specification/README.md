# FF-A: Broad Bond Specification Fit

Bond SMIRKS use broad element-connectivity and ring-membership (atom primitive) but no bond type specificity.
Source OFFXML: `4_make_offxmls/a_broad_specification/openff-2.3.0-ring-no-bond.offxml`.
Topology: `5_setup_train_ff_topologies/2026_02_20_uci_off_a`.

**Bonds only** (Angles not fit).
**Mini-batching Used**

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

Scaling values for Bonds: {'k': 0.0017385467335095824, 'length': 0.6426217500674781}
Scaling values for Angles: {'k': 0.0063567565117824765, 'angle': 0.4871127755735693}

Learning Rate: 0.005
Number of Epochs: 40

Previous Step:
None

Conclusion:


Next Step:
