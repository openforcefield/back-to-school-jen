# FF-B, Generalized Bonds with Ring Definitions: Attempt 3

This is a bond and angle refit of OpenFF-2.3.0-rc2 using SMEE and the SPICE dataset.

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

Scaling values for Bonds: {'k': 0.0014741405255011971, 'length': 0.6393008625225789}
Scaling values for Angles: {'k': 0.0063567565117824765, 'angle': 0.4871127755735693} # in radians

Learning Rate: 0.015
Number of Epochs: 300

Previous Step:
None

Conclusion:
The loss plateaus after 50 epochs. The bond parameters and angle harmonic constant plateau over the 300 epochs. The equilibrium angles seem to have just finished plateauing but more epochs could improve.

The learning rate of 0.015 has a similar amount of oscillation to other fits with 0.02, but not as heavy as 0.03.

Next Step:
Benchmark and compare to SMIRKS specific fits.
