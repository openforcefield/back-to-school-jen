"""
SMIRKS Pattern Generation and Force Field Parameter Processing.

This module provides utilities for generating SMIRKS patterns at different specificity
levels and integrating molecular mechanics components into OpenFF force fields. It
handles the conversion from molecular structures to parameterizable SMIRKS patterns
and manages force field parameter creation and assignment.

Key Functions
-------------
Atom SMIRKS Generation:
    - get_atom_smirks_standard: Standard specificity with atomic number and degree
    - get_atom_smirks_terminal_wildcard: Wildcard terminals for broader coverage
    - get_atom_smirks_terminal_h_no_h: Hydrogen/non-hydrogen distinction for terminals
    - get_atom_smirks_all_bonded: Detailed neighboring atom specifications

Bond SMIRKS Generation:
    - get_bond_smirks_standard: Explicit bond type specification
    - get_bond_smirks_non_central_bonds_generalised: Generalized non-central bonds
    - get_bond_smirks_all_bonds_generalised: Wildcard for all bonds

Force Field Integration:
    - add_types_to_ff: Add component parameters to force field handlers

Examples
--------
Standard atom SMIRKS generation:

>>> from rdkit import Chem
>>> mol = Chem.MolFromSmiles("CCO")
>>> atom_smirks = get_atom_smirks_standard(0, 0, mol, (0, 2))
>>> print(atom_smirks)
[#6X4:1]

Terminal wildcard for broader matching:

>>> wildcard_smirks = get_atom_smirks_terminal_wildcard(0, 0, mol, (0, 2))
>>> print(wildcard_smirks)
[*:1]

Adding components to force field:

>>> from openff.toolkit import ForceField
>>> component_types = {0: {"[#6:1]-[#6:2]": [bond1, bond2]}}
>>> ff = ForceField("openff-2.0.0.offxml")
>>> updated_ff = add_types_to_ff(ff, component_types, Bond)
"""

from copy import deepcopy

from rdkit import Chem
from tqdm import tqdm

from openff.toolkit import ForceField
from openff.toolkit.typing.engines.smirnoff.parameters import ParameterType

from .molecular_classes import MMComponent


def get_atom_descriptors(at_idx: int, mol: Chem.Mol) -> dict[str, str]:
    """
    Generate SMIRKS-ready descriptors for an atom.

    Extracts atomic properties needed for SMIRKS pattern generation, including
    atomic number, degree (number of bonds), and formal charge.

    Parameters
    ----------
    at_idx : int
        Index of the atom in the molecule.
    mol : rdkit.Chem.Mol
        RDKit molecule object containing the atom.

    Returns
    -------
    dict[str, str]
        Dictionary containing SMIRKS-formatted descriptors:
        - 'atomic_num': Atomic number as "#N" (e.g., "#6" for carbon)
        - 'degree': Degree as "XN" (e.g., "X4" for tetrahedral)
        - 'charge': Formal charge as "+N" or "-N" (e.g., "+1", "-2")

    Examples
    --------
    >>> from rdkit import Chem
    >>> mol = Chem.MolFromSmiles("CCO")
    >>> descriptors = get_atom_descriptors(0, mol)
    >>> print(descriptors)
    {'atomic_num': '#6', 'degree': 'X4', 'charge': '+0'}
    """
    descriptors = {
        "atomic_num": f"#{mol.GetAtomWithIdx(at_idx).GetAtomicNum()}",
        "degree": f"X{mol.GetAtomWithIdx(at_idx).GetDegree()}",
        "charge": mol.GetAtomWithIdx(at_idx).GetFormalCharge(),
    }
    descriptors["charge"] = (
        f"+{descriptors['charge']}"
        if descriptors["charge"] >= 0
        else str(descriptors["charge"])
    )

    return descriptors


def get_bond_type(atom_idxs: tuple[int, int], mol: Chem.Mol) -> str:
    """
    Get SMIRKS bond type symbol between two atoms.

    Converts RDKit bond types to their corresponding SMIRKS representations
    used in chemical environment patterns.

    Parameters
    ----------
    atom_idxs : tuple[int, int]
        Indices of the two atoms forming the bond.
    mol : rdkit.Chem.Mol
        RDKit molecule object containing the bond.

    Returns
    -------
    str
        SMIRKS bond type symbol:
        - '-' for single bonds
        - '=' for double bonds
        - '#' for triple bonds
        - ':' for aromatic bonds
        - '~' for any other bond type

    Raises
    ------
    ValueError
        If no bond exists between the specified atoms.

    Examples
    --------
    >>> from rdkit import Chem
    >>> mol = Chem.MolFromSmiles("C=C")
    >>> bond_type = get_bond_type((0, 1), mol)
    >>> print(bond_type)
    =
    """
    bond = mol.GetBondBetweenAtoms(*atom_idxs)
    if bond is None:
        raise ValueError(f"No bond found between atoms {atom_idxs} in the molecule.")

    bt = bond.GetBondType()
    if bt == Chem.BondType.SINGLE:
        return "-"
    elif bt == Chem.BondType.DOUBLE:
        return "="
    elif bt == Chem.BondType.TRIPLE:
        return "#"
    elif bt == Chem.BondType.AROMATIC:
        return ":"
    else:
        return "~"


def get_atom_smirks_standard(
    at_idx: int, at_id: int, mol: Chem.Mol, terminal_idxs: tuple[int, int]
) -> str:
    """
    Generate standard specificity SMIRKS representation for an atom.

    Creates a SMIRKS atom pattern using atomic number and degree, providing
    the standard level of chemical specificity for force field parameterization.

    Parameters
    ----------
    at_idx : int
        Index of the atom in the molecule.
    at_id : int
        Position identifier in the component (0-based, converted to 1-based for SMIRKS).
    mol : rdkit.Chem.Mol
        RDKit molecule object containing the atom.
    terminal_idxs : tuple[int, int]
        Indices of terminal atoms in the component (not used in standard specificity).

    Returns
    -------
    str
        SMIRKS atom pattern in format "[#N X M:ID]" where:
        - N is atomic number
        - M is degree (number of bonds)
        - ID is 1-based position identifier

    Examples
    --------
    >>> from rdkit import Chem
    >>> mol = Chem.MolFromSmiles("CCO")
    >>> atom_smirks = get_atom_smirks_standard(0, 0, mol, (0, 2))
    >>> print(atom_smirks)
    [#6X4:1]
    """
    ds = get_atom_descriptors(at_idx, mol)
    return f"[{ds['atomic_num']}{ds['degree']}:{at_id + 1}]"


def get_atom_smirks_all_bonded(
    at_idx: int,
    at_id: int,
    mol: Chem.Mol,
    central_only: bool,
    terminal_idxs: tuple[int, int],
) -> str:
    """
    Generate highly specific SMIRKS representation including all bonded atoms.

    Creates detailed SMIRKS patterns that specify all neighboring atoms and their
    bond types, providing maximum chemical specificity for central atoms while
    keeping terminal atoms at standard specificity.

    Parameters
    ----------
    at_idx : int
        Index of the atom in the molecule.
    at_id : int
        Position identifier in the component (0-based).
    mol : rdkit.Chem.Mol
        RDKit molecule object containing the atom.
    central_only : bool
        If True, only apply detailed specification to central atoms (positions 1, 2).
        Terminal atoms (positions 0, 3) remain at standard specificity.
    terminal_idxs : tuple[int, int]
        Indices of terminal atoms in the component.

    Returns
    -------
    str
        SMIRKS atom pattern with bonded atom specifications:
        - Standard format for terminal atoms: "[#N X M:ID]"
        - Extended format for central atoms: "[#N X M:ID](bond_type[#N X M])..."

    Examples
    --------
    >>> from rdkit import Chem
    >>> mol = Chem.MolFromSmiles("CCCO")
    >>> # Central carbon with all bonded atoms specified
    >>> central_smirks = get_atom_smirks_all_bonded(1, 1, mol, True, (0, 3))
    >>> print(central_smirks)
    [#6X4:2](-[#6X4])(-[#6X4])(-[#1X1])(-[#1X1])
    """
    ds = get_atom_descriptors(at_idx, mol)

    # If terminal, don't make more specific
    if central_only and at_id in [0, 3]:  # Terminal atoms which we shouldn't specialize
        return f"[{ds['atomic_num']}{ds['degree']}:{at_id + 1}]"

    # The atom is central. Get all the bonded atoms and their bond types
    bonded_atoms = mol.GetAtomWithIdx(at_idx).GetNeighbors()
    bonded_atom_smirks = []
    for bonded_atom in bonded_atoms:
        bonded_idx = bonded_atom.GetIdx()
        bonded_ds = get_atom_descriptors(bonded_idx, mol)
        bond_type = get_bond_type((at_idx, bonded_idx), mol)
        bonded_atom_smirks.append(
            f"({bond_type}[{bonded_ds['atomic_num']}{bonded_ds['degree']}])"
        )

    return (
        f"[{ds['atomic_num']}{ds['degree']}:{at_id + 1}]{''.join(bonded_atom_smirks)}"
    )


def get_atom_smirks_terminal_h_no_h(
    at_idx: int, at_id: int, mol: Chem.Mol, terminal_idxs: tuple[int, int]
) -> str:
    """
    Generate SMIRKS with hydrogen/non-hydrogen distinction for terminals.

    Creates SMIRKS patterns that distinguish between hydrogen and non-hydrogen
    atoms for terminal positions while maintaining standard specificity for
    central atoms. This provides broader coverage than atomic number specificity.

    Parameters
    ----------
    at_idx : int
        Index of the atom in the molecule.
    at_id : int
        Position identifier in the component (0-based).
    mol : rdkit.Chem.Mol
        RDKit molecule object containing the atom.
    terminal_idxs : tuple[int, int]
        Indices of terminal atoms in the component.

    Returns
    -------
    str
        SMIRKS atom pattern:
        - Terminal hydrogen: "[#1:ID]"
        - Terminal non-hydrogen: "[!#1:ID]"
        - Central atoms: "[#N X M:ID]" (standard specificity)

    Examples
    --------
    >>> from rdkit import Chem
    >>> mol = Chem.MolFromSmiles("CCO")
    >>> # Terminal carbon (non-hydrogen)
    >>> terminal_smirks = get_atom_smirks_terminal_h_no_h(0, 0, mol, (0, 2))
    >>> print(terminal_smirks)
    [!#1:1]
    """
    ds = get_atom_descriptors(at_idx, mol)

    # If not terminal, remain specific
    if at_id not in terminal_idxs:  # Not terminal, remain specific
        return f"[{ds['atomic_num']}{ds['degree']}:{at_id + 1}]"

    # If terminal, check if hydrogen or not
    ds["atomic_num"] = ds["atomic_num"] if ds["atomic_num"] == "#1" else "!#1"

    return f"[{ds['atomic_num']}:{at_id + 1}]"


def get_atom_smirks_terminal_wildcard(
    at_idx: int, at_id: int, mol: Chem.Mol, terminal_indices: tuple[int, int]
) -> str:
    """
    Generate SMIRKS with wildcard terminals for maximum coverage.

    Creates SMIRKS patterns using wildcards for terminal atoms to achieve
    the broadest possible chemical coverage while maintaining specificity
    for central atoms where it matters most for parameterization.

    Parameters
    ----------
    at_idx : int
        Index of the atom in the molecule.
    at_id : int
        Position identifier in the component (0-based).
    mol : rdkit.Chem.Mol
        RDKit molecule object containing the atom.
    terminal_indices : tuple[int, int]
        Indices of terminal atoms in the component.

    Returns
    -------
    str
        SMIRKS atom pattern:
        - Terminal atoms: "[*:ID]" (wildcard matches any atom)
        - Central atoms: "[#N X M:ID]" (standard specificity)

    Examples
    --------
    >>> from rdkit import Chem
    >>> mol = Chem.MolFromSmiles("CCO")
    >>> # Terminal with wildcard
    >>> wildcard_smirks = get_atom_smirks_terminal_wildcard(0, 0, mol, (0, 2))
    >>> print(wildcard_smirks)
    [*:1]
    """
    ds = get_atom_descriptors(at_idx, mol)

    # If not terminal, remain specific
    if at_id not in terminal_indices:  # Not terminal, remain specific
        return f"[{ds['atomic_num']}{ds['degree']}:{at_id + 1}]"

    # If terminal, use wildcard
    return f"[*:{at_id + 1}]"


def get_bond_smirks_standard(
    atom_idxs: tuple[int, int], central_bond: bool, mol: Chem.Mol
) -> str:
    """
    Generate standard specificity SMIRKS representation for a bond.

    Creates explicit bond type specifications for all bonds in the component,
    providing the standard level of chemical specificity.

    Parameters
    ----------
    atom_idxs : tuple[int, int]
        Indices of the two atoms forming the bond.
    central_bond : bool
        Whether this is the central bond in the component (not used in standard).
    mol : rdkit.Chem.Mol
        RDKit molecule object containing the bond.

    Returns
    -------
    str
        SMIRKS bond type symbol ('-', '=', '#', ':', or '~').

    Examples
    --------
    >>> from rdkit import Chem
    >>> mol = Chem.MolFromSmiles("C=C")
    >>> bond_smirks = get_bond_smirks_standard((0, 1), False, mol)
    >>> print(bond_smirks)
    =
    """
    return get_bond_type(atom_idxs, mol)


def get_bond_smirks_non_central_bonds_generalised(
    atom_idxs: tuple[int, int], central_bond: bool, mol: Chem.Mol
) -> str:
    """
    Generate SMIRKS with explicit central bonds and generalized non-central bonds.

    Provides specificity for central bonds while using wildcards for non-central
    bonds to achieve broader chemical coverage with focused parameterization.

    Parameters
    ----------
    atom_idxs : tuple[int, int]
        Indices of the two atoms forming the bond.
    central_bond : bool
        Whether this is the central bond in the component.
    mol : rdkit.Chem.Mol
        RDKit molecule object containing the bond.

    Returns
    -------
    str
        SMIRKS bond type:
        - Explicit type for central bonds ('-', '=', '#', ':', '~')
        - Wildcard '~' for non-central bonds

    Examples
    --------
    >>> from rdkit import Chem
    >>> mol = Chem.MolFromSmiles("C-C=C-C")
    >>> # Central bond (explicit)
    >>> central = get_bond_smirks_non_central_bonds_generalised((1, 2), True, mol)
    >>> print(central)
    =
    >>> # Non-central bond (wildcard)
    >>> non_central = get_bond_smirks_non_central_bonds_generalised((0, 1), False, mol)
    >>> print(non_central)
    ~
    """
    if not central_bond:
        return "~"
    else:
        return get_bond_type(atom_idxs, mol)


def get_bond_smirks_all_bonds_generalised(
    atom_idxs: tuple[int, int], central_bond: bool, mol: Chem.Mol
) -> str:
    """
    Generate fully generalized SMIRKS with wildcards for all bonds.

    Uses wildcard bond types for maximum chemical coverage, useful when
    bond-specific parameterization is not required.

    Parameters
    ----------
    atom_idxs : tuple[int, int]
        Indices of the two atoms forming the bond.
    central_bond : bool
        Whether this is the central bond (not used in this generalization).
    mol : rdkit.Chem.Mol
        RDKit molecule object containing the bond.

    Returns
    -------
    str
        Always returns '~' (wildcard bond type).

    Examples
    --------
    >>> from rdkit import Chem
    >>> mol = Chem.MolFromSmiles("C=C")
    >>> bond_smirks = get_bond_smirks_all_bonds_generalised((0, 1), True, mol)
    >>> print(bond_smirks)
    ~
    """
    return "~"


def add_types_to_ff(
    ff: ForceField,
    component_types: dict[int, dict[str, list[MMComponent]]],
    component_class: type[MMComponent],
    extra_parameters: list[ParameterType] | None = None,
) -> ForceField:
    """
    Add molecular mechanics component parameters to a force field.

    Integrates component-specific parameters into an OpenFF force field by creating
    a new parameter handler and populating it with parameters derived from the
    provided component types and their associated molecules.

    Parameters
    ----------
    ff : openff.toolkit.ForceField
        Base force field to extend with new parameters.
    component_types : dict[int, dict[str, list[MMComponent]]]
        Hierarchical organization of components:
        - First level: specificity level number
        - Second level: SMIRKS pattern string
        - Third level: list of components matching that pattern
    component_class : type[MMComponent]
        Type of component (Bond, Angle, ProperTorsion, ImproperTorsion).
    extra_parameters : list[ParameterType], optional
        Additional parameters to append at the end of the handler.

    Returns
    -------
    openff.toolkit.ForceField
        New force field with integrated component parameters. The original
        force field is deep-copied to avoid modification.

    Notes
    -----
    Parameters are added in order of increasing specificity (lowest specificity
    first), and within each specificity level, they are sorted by population
    (most common patterns first). This ensures proper parameter precedence
    in the SMIRNOFF force field hierarchy.

    Examples
    --------
    Adding bond parameters to a force field:

    >>> from openff.toolkit import ForceField
    >>> base_ff = ForceField("openff-2.0.0.offxml")
    >>> bond_types = {
    ...     0: {"[#6:1]-[#6:2]": [bond1, bond2, bond3]},
    ...     1: {"[#6X4:1]-[#6X4:2]": [bond1, bond2]}
    ... }
    >>> new_ff = add_types_to_ff(base_ff, bond_types, Bond)

    Adding extra parameters:

    >>> extra_params = [custom_bond_param]
    >>> new_ff = add_types_to_ff(base_ff, bond_types, Bond, extra_params)
    """
    ff_copy = deepcopy(ff)
    handler = component_class.handler_class(version=component_class.handler_version)

    # Write the lowest specificity level first
    for specificity_num, components_by_type in sorted(
        component_types.items(), key=lambda item: item[0]
    ):
        for i, (smirks, components) in tqdm(
            enumerate(
                sorted(
                    components_by_type.items(),
                    key=lambda item: len(item[1]),
                    reverse=True,
                )
            ),
            total=len(components_by_type),
            desc=f"Adding parameters for specificity {specificity_num}",
        ):
            parameter = component_class.get_parameter(
                smirks, specificity_num, components, i, ff
            )
            handler.parameters.append(parameter)

    # Add any extra parameters at the end
    if extra_parameters:
        for parameter in extra_parameters:
            handler.parameters.append(parameter)

    tag_name = component_class.handler_class._TAGNAME
    if tag_name is not None:
        ff_copy.deregister_parameter_handler(tag_name)
    ff_copy.register_parameter_handler(handler)

    return ff_copy
