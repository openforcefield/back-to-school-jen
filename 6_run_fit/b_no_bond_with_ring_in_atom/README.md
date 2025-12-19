# FF-B, Generalized Bonds with Ring Definitions: Attempt 3

In this force field bond and angle SMIRKs are set to be specific in the degree of element connectivity, but the bond type is generalized.
Information of whether the bond is in a ring or not is provided.

This fit was run with the parameters as is, and scaling defined (loosely) as:
```python
PARAMETERS = {
    "Bonds": descent.train.ParameterConfig(
        cols=["k", "length"],
        scales={"k": bond_k_mean, "length": bond_x0_mean},
        limits={"k": [0.0, None], "length": [0.0, None]},
    ),
    "Angles": descent.train.ParameterConfig(
        cols=["k", "angle"],
        scales={"k": angle_k_mean, "angle": angle_x0_mean},
        limits={"k": [0.0, None], "angle": [0.0, math.pi]},
    ),
}
```
The process of taking the mean from the initial dataset is done in fit_data.py

Learning Rate: 0.1
Number of Epochs: 100 (submitted as 300 and stopped early)

Previous Step:
FF-B2, Generalized Bonds with Ring Definitions: Attempt 2
Showed descent progress in all parameters, but it's expected that fitting of Bond spring constant k could be improved.

Conclusion:
Poor fitting performance for angles and seemingly descent performance for bond reference length.

Next Step:
FF-B3, Generalized Bonds with Ring Definitions: Attempt 4
Run with linearized potentials
