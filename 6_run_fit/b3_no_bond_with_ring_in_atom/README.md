# FF-B, Generalized Bonds with Ring Definitions: Attempt 3

In this force field bond and angle SMIRKs are set to be specific in the degree of element connectivity, but the bond type is generalized.
Information of whether the bond is in a ring or not is provided.

This fit was run with the parameters as is, and scaling defined (loosely) as:
```python
PARAMETERS = {
    "Bonds": descent.train.ParameterConfig(
        cols=["k", "length"],
        scales={"k": 1/bond_k_mean, "length": 1/bond_x0_mean},
        limits={"k": [0.0, None], "length": [0.0, None]},
    ),
    "Angles": descent.train.ParameterConfig(
        cols=["k", "angle"],
        scales={"k": 1/angle_k_mean, "angle": 1/angle_x0_mean},
        limits={"k": [0.0, None], "angle": [0.0, math.pi]},
    ),
}
```
The process of taking the mean from the initial dataset is done in fit_data.py

Mean values for Bonds: {'k': 0.0018241198088180758, 'length': 0.638552889282609}
Mean values for Angles: {'k': 0.008619126829830522, 'angle': 0.49716413002732057} # in radians

Learning Rate: 0.005
Number of Epochs: 400

Previous Step:
FF-B, Generalized Bonds with Ring Definitions: Attempt 3
Showed descent progress in all parameters, but equilibrium bond lengths fell to half of what was expected, presumably from the high learning rate.

Conclusion:


Next Step:
