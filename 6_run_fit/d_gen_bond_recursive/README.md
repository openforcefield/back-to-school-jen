# FF-D: Generalized Bond Recursive

Bond SMIRKS use recursive ring detection with generalized atom environments.
Source OFFXML: `4_make_offxmls/d_gen_bond_recursive/openff-2.3.0-gen_bond_atom_rec1.offxml`.
Topology: `5_setup_train_ff_topologies/uci_off_d`.

**Bonds only** (Angles not fit). Validation set used during training (`full_split_uci/data-test`).
**Minibatching**

Scaling is data-derived (mean-based) as defined in `fit_data.py`:
```python
PARAMETERS = {
    "Bonds": descent.train.ParameterConfig(
        cols=["k", "length"],
        scales={"k": 1/bond_k_mean, "length": 1/bond_x0_mean},
        limits={"k": [90, None], "length": [0.0, None]},  # k lower bound 90, not 0
    ),
}
```
The process of taking the mean from the initial dataset is done in `fit_data.py`.

Learning Rate: 0.005
Number of Epochs: 220 (ended early)

Previous Step:

Conclusion:


Next Step:
