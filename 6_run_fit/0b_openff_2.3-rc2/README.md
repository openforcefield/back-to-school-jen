# FF-0b: Baseline Refit of OpenFF-2.3.0-rc2 (Bonds only, lower bound k=90)

This is a bond refit of `openff_unconstrained-2.3.0-rc2.offxml` using SMEE and the UCI SPICE dataset.
Uses the stock OpenFF-2.3.0-rc2 SMIRKS without modification.
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
Number of Epochs: 50

Previous Step:
`0_openff_2.3-rc2` — Bonds + Angles fit with no lower bound constraint on k.

Conclusion:


Next Step:
