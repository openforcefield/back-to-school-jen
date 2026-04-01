# FF-E: Broad Specification with Bond Types and Recursion

In this force field bond SMIRKS use a broad element-connectivity specification with
explicit bond types and recursive atom neighbor detection.
Source OFFXML: `4_make_offxmls/e_broad_spec_recursion1/openff-2.3.0-bondtypes-recursion.offxml`.
Topology: `5_setup_train_ff_topologies/uci_off_e`.
Validation set used during training (`full_split_uci/data-test`).

This fit was run with the parameters as is, and scaling defined (loosely) as:
```python
PARAMETERS = {
    "Bonds": descent.train.ParameterConfig(
        cols=["k", "length"],
        scales={"k": 1/bond_k_mean, "length": 1/bond_x0_mean},
        limits={"k": [90, None], "length": [0.0, None]},
    ),
}
```
The process of taking the mean from the initial dataset is done in fit_data.py

Learning Rate: 0.005
Number of Epochs: 100

Previous Step:
`a_broad_specification` — broad element+ring SMIRKS without explicit bond types or recursion.

Conclusion:


Next Step:
