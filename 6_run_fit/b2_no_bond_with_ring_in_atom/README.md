# FF-B2, Generalized Bonds with Ring Definitions: Attempt 2

In this force field bond and angle SMIRKs are set to be specific in the degree of element connectivity, but the bond type is generalized.
Information of whether the bond is in a ring or not is provided.

This fit was run with the parameters as is, and scaling defined as:
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

Learning Rate: 0.02
Number of Epochs: 300

Previous Step:
Now overwritten in directory "b-*". Fit with a learning rate of 0.001 was unsatisfactory in its progress, similar to `0_openff2.2.1` which showed no benchmark improvement to the original `openff_2.2.1`.

Conclusion:
Good progress is observed for all parameters where angle parameters are leveling off but bond spring energies continue to change steadily suggesting that better scaling would improve their performance.

Next Step:
FF-B, Generalized Bonds with Ring Definitions: Attempt 3
