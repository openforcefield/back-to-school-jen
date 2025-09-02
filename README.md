# Research Question: Do we achieve better performance with hyper-specified bond and angle parameters (and later clustering?)
[Lily's Slides](https://docs.google.com/presentation/d/1c4_XFG18kQBQNQ86gaeqEUIFioH0Ld7TtG-Xfzu3j3Y/edit?slide=id.g2f3a95763d8_0_883#slide=id.g2f3a95763d8_0_883)

## In scope:
- 1 fit with Sage 2.2.1 valence
	- https://github.com/openforcefield/openff-forcefields
	- `forcefield = ForceField("openff_unconstrained-2.1.0.offxml")`
- 1 fit with parameters hyper-specified, w/o linearised harmonics
- 1 fit with parameters hyper-specified, with linearised harmonics
- Benchmarking QM

## Nice to have:
- Incorporate neighboring atoms
- Fit with torsions over-specified
- Clustering final parameters
- Other benchmarks

## Not in scope:
- Generating new data
- vdW fits

## Getting Started
1) In home dir on UCI HPC3, clone this repo somewhere in `/dfs9/dmobley-lab/user_id/` with:
```bash
   git clone https://github.com/openforcefield/back-to-school-jen.git
```
2) Install with:
```bash
	srun -c 2 -p free --pty /bin/bash -i
	cd back-to-school-jen; micromamba create -f environment.yaml
```
3) From the `1_data` directory get data and reformat:
```bash
    python get_data_spice2.py --data-dir "." 2>&1 | tee log.txt
```
    or
```bash
    chmod 774 run_local.sh
    ./run_local.sh
```
