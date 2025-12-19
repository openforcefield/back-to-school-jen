# FF-SMEE-SPICE, Fit OpenFF-2.2.1 with SPICE data using SMEE

Force field is kept identical to OpenFF-2.2.1

This fit was run with the parameters as is, and scaling defined (loosely) as:
```python
PARAMETERS = {
    "Bonds": descent.train.ParameterConfig(
        cols=["k", "length"],
        scales={"k": 1e-2, "length": 1.0},
        limits={"k": [0.0, None], "length": [0.0, None]},
    ),
    "Angles": descent.train.ParameterConfig(
        cols=["k", "angle"],
        scales={"k": 1e-2, "angle": 1.0},
        limits={"k": [0.0, None], "angle": [0.0, math.pi]},
    ),
}
```

Learning Rate: 0.001
Number of Epochs: 1000

Previous Step:
None

Conclusion:
Poor fitting performance for bond and angle spring constants.

Next Step:
None. This was run in error
