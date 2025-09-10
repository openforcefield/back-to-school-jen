from datasets import load_from_disk
import numpy as np
import matplotlib.pyplot as plt
from openff.units import unit

HARTREE_TO_KCAL: float = (1 * unit.hartree * unit.avogadro_constant).m_as(
    unit.kilocalories_per_mole
)
BOHR_TO_ANGSTROM: float = (1.0 * unit.bohr).m_as(unit.angstrom)

ds = load_from_disk("raw-spice")
forces = []
for row in ds:
    n_conformers = len(row["energy"])
    n_atoms = len(row["forces"]) // (n_conformers * 3)
    for tmp in np.reshape(row["forces"], (n_conformers, n_atoms, 3)):
        forces.append(
            np.sum(np.linalg.norm(tmp, -1))
        )  # * HARTREE_TO_KCAL / BOHR_TO_ANGSTROM)
_, axs = plt.subplots(1, 1, figsize=(6, 4))
axs.hist(forces, bins=50, alpha=0.6)
axs.set_xlabel("Force in Molecules")
axs.set_ylabel("Count")
plt.show()
