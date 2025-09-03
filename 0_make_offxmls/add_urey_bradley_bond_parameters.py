"""Add Urey-Bradley terms to a supplied OpenFF force field."""

from copy import deepcopy
from pathlib import Path

import typer
from openff.toolkit import ForceField
from openff.toolkit.typing.engines.smirnoff.parameters import AngleType
from openff.units import unit as off_unit
from smirnoff_plugins.handlers.valence import UreyBradleyHandler


def angle_smirks_to_urey_bradley_smirks(smirks: str) -> str:
    """
    Convert an angle SMIRKS to a Urey-Bradley SMIRKS by removing the ':2' label and converting the
    ':3' label to a '2' label.
    """
    # Remove the ':2' label
    smirks = smirks.replace(":2", "")
    # Convert the ':3' label to a '2' label
    smirks = smirks.replace(":3", ":2")
    return smirks


def get_urey_bradley_bond_parameter(
    angle: AngleType, k: float = 0.01, length: float = 2.0
) -> UreyBradleyHandler.UreyBradleyType:
    """
    Get the Urey-Bradley bond parameter for a given angle parameter.

    Parameters
    ----------
    angle : AngleType
        The angle parameter to convert to a Urey-Bradley bond parameter.
    k : float
        The force constant for the Urey-Bradley term, in kcal/mol/Angstrom^2.
        Defaults to 0.01 kcal/mol/Angstrom^2.
    length : float
        The length for the Urey-Bradley term, in Angstroms.
        Defaults to 2.0 Angstroms.

    Returns
    -------
    UreyBradleyHandler.UreyBradleyType
        A Urey-Bradley bond parameter with the same SMIRKS as the angle parameter,
        a force constant of `k` kcal/mol/Angstrom^2, and a length of 2.0 Angstroms.
    """
    return UreyBradleyHandler.UreyBradleyType(
        smirks=angle_smirks_to_urey_bradley_smirks(angle.smirks),
        k=k * off_unit.kilocalories_per_mole / off_unit.angstrom**2,
        length=length * off_unit.angstrom,
        id=f"{angle.id}_urey_bradley",
    )


def add_urey_bradley_terms(forcefield: ForceField) -> ForceField:
    """
    Add Urey-Bradley terms to the force field, initialising the force constants
    to 0 and distances to 2 Angstroms.
    """
    new_ff = deepcopy(forcefield)
    angle_handler = new_ff.get_parameter_handler("Angles")
    ub_handler = new_ff.get_parameter_handler("UreyBradleys")
    for parameter in angle_handler.parameters:
        ub_bond = get_urey_bradley_bond_parameter(parameter)
        ub_handler.parameters.append(ub_bond)

    return new_ff


def main(input_ff_path: Path, output_ff_path: Path):
    """
    Add Urey-Bradley terms to a supplied OpenFF force field.
    """
    forcefield = ForceField(str(input_ff_path), load_plugins=True)
    new_forcefield = add_urey_bradley_terms(forcefield)
    new_forcefield.to_file(str(output_ff_path))


if __name__ == "__main__":
    typer.run(main)
