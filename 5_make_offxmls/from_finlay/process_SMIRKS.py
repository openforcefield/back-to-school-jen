"""
SMIRKS Pattern Generation with Factory Pattern.

This module provides a factory-based interface for generating SMIRKS patterns
at different specificity levels for molecular mechanics force field parameters.

Specificity Levels
------------------
Specificity levels control how detailed and chemical-environment-specific the
generated SMIRKS patterns are:

- **Low specificity**: Generic patterns like [#6:1] (any carbon) that match
  many molecular environments but provide less chemical detail.
- **High specificity**: Detailed patterns like [#6X4;!r3;!r4;!r5;!r6;!r7;!r8:1]
  (~[#1X1])(~[#8X2]) that match fewer environments but capture more chemistry.

Lower-numbered specificity levels are applied first in force fields, with
higher levels providing more specific overrides. This creates a hierarchical
parameter assignment where general patterns are refined by specific ones.

Key Classes
-----------
SMIRKSFactory : Main factory class for generating SMIRKS patterns
AtomSMIRKSConfig : Configuration for atom-specific SMIRKS generation
BondSMIRKSConfig : Configuration for bond-specific SMIRKS generation
TerminalBehavior : Enum for terminal atom handling options
BondedAtomBehavior : Enum for bonded atom inclusion options
BondSpecificity : Enum for bond specificity levels

Key Functions
-------------
get_atom_descriptors : Extract atomic properties for SMIRKS patterns
get_bond_descriptors : Extract bond properties for SMIRKS patterns
add_types_to_ff : Add component parameters to OpenFF force fields

Examples
--------
Basic factory usage:
>>> factory = SMIRKSFactory()
>>> specificity_level = factory.create_specificity_level("Standard")

Ring-aware patterns:
>>> factory = SMIRKSFactory(atom_include_ring_info=True)
>>> mol = Chem.MolFromSmiles("CCO")
>>> atom_smirks = factory.get_atom_smirks(0, 0, mol, (0, 2))
>>> print(atom_smirks)  # [#6X4;!r3;!r4;!r5;!r6;!r7;!r8:1]

Using SpecificityLevel objects:
>>> level = factory.create_specificity_level("RingAware")
>>> # Generate atom SMIRKS for carbon (index 0, position 0, terminals 0,2)
>>> atom_pattern = level.get_atom_smirks(0, 0, mol, (0, 2))
>>> print(atom_pattern)  # '[#6X4;!r3;!r4;!r5;!r6;!r7;!r8:1]'
>>> # Generate bond SMIRKS for C-C bond (atoms 0,1, not central)
>>> bond_pattern = level.get_bond_smirks((0, 1), False, mol)
>>> print(bond_pattern)  # '-;!@'

Integration in force field workflows:
>>> levels = {0: level}  # Dict used by component processing pipelines
>>> # The level methods are called to generate SMIRKS for each component
>>> # in get_mm_components_by_specificity_by_type() function
"""

from copy import deepcopy
from enum import Enum
from dataclasses import dataclass

from rdkit import Chem
from tqdm import tqdm

from openff.toolkit import ForceField
from openff.toolkit.typing.engines.smirnoff.parameters import ParameterType

from .molecular_classes import MMComponent, SpecificityLevel


class TerminalBehavior(Enum):
    """Behavior for terminal atoms in SMIRKS patterns."""

    WILDCARD = "wildcard"  # [*:ID]
    H_NO_H = "h_no_h"  # [#1:ID] or [!#1:ID]
    STANDARD = "standard"  # [#NXM:ID]


class BondedAtomBehavior(Enum):
    """How to include bonded atom information."""

    NONE = "none"  # No bonded atoms
    CENTRAL_EXPLICIT_ATOMS = "central_explicit_atoms"  # Wildcard non-central
    CENTRAL_EXPLICIT_ATOMS_BONDS = (
        "central_explicit_atom_and_bonds"  # Wildcard non-central
    )
    EXPLICIT_ATOMS = "explicit_atoms"  # All atoms
    EXPLICIT_ATOMS_BONDS = (
        "explicit_atoms_and_bonds"  # All atoms with explicit bond types
    )


class BondSpecificity(Enum):
    """Bond specificity levels."""

    STANDARD = "standard"  # Explicit bond types
    NON_CENTRAL_WILDCARD = "non_central_wildcard"  # Wildcard non-central
    WILDCARD = "wildcard"  # All wildcard


@dataclass
class AtomSMIRKSConfig:
    """Configuration for atom SMIRKS pattern generation."""

    include_ring_info: bool = False
    bonded_atom_behavior: BondedAtomBehavior = BondedAtomBehavior.NONE
    terminal_behavior: TerminalBehavior = TerminalBehavior.STANDARD

    def __post_init__(self):
        """Validate configuration consistency."""
        if not isinstance(self.bonded_atom_behavior, BondedAtomBehavior):
            raise ValueError(
                f"Invalid bonded_atom_behavior: {self.bonded_atom_behavior}"
            )
        if not isinstance(self.terminal_behavior, TerminalBehavior):
            raise ValueError(f"Invalid terminal_behavior: {self.terminal_behavior}")


@dataclass
class BondSMIRKSConfig:
    """Configuration for bond SMIRKS pattern generation."""

    include_ring_info: bool = False
    specificity: BondSpecificity = BondSpecificity.STANDARD

    def __post_init__(self):
        """Validate configuration consistency."""
        if not isinstance(self.specificity, BondSpecificity):
            raise ValueError(f"Invalid specificity: {self.specificity}")


def get_atom_descriptors(at_idx: int, mol: Chem.Mol) -> dict[str, str]:
    """
    Generate comprehensive SMIRKS-ready descriptors for an atom.

    Extracts atomic properties needed for SMIRKS pattern generation, including
    atomic number, degree (number of bonds), formal charge, ring membership,
    and aromaticity.

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
        - 'ring_info': Ring membership as ";rN" or ";!r3;!r4;!r5;!r6;!r7;!r8"
        - 'aromaticity': Aromatic (";a") or non-aromatic (";A") designation

    Examples
    --------
    >>> mol = Chem.MolFromSmiles("CCO")
    >>> descriptors = get_atom_descriptors(0, mol)  # First carbon
    >>> descriptors['atomic_num']  # '#6'
    >>> descriptors['degree']      # 'X4'
    >>> descriptors['ring_info']   # ';!r3;!r4;!r5;!r6;!r7;!r8' (not in ring)
    """
    # Figure out if the atom is in a ring of size 3 - 8
    ring_sizes = []
    for ring in mol.GetRingInfo().AtomRings():
        if at_idx in ring:
            ring_size = len(ring)
            if 3 <= ring_size <= 8:
                ring_sizes.append(ring_size)

    atom = mol.GetAtomWithIdx(at_idx)

    descriptors = {
        "atomic_num": f"#{atom.GetAtomicNum()}",
        "degree": f"X{atom.GetDegree()}",
        "charge": atom.GetFormalCharge(),
        "ring_info": f";r{min(ring_sizes)}"
        if ring_sizes
        else ";!r3;!r4;!r5;!r6;!r7;!r8",
        "aromaticity": ";a" if atom.GetIsAromatic() else ";A",
    }

    # Format charge
    descriptors["charge"] = (
        f"+{descriptors['charge']}"
        if descriptors["charge"] >= 0
        else str(descriptors["charge"])
    )

    return descriptors


def get_bond_descriptors(
    atom_idxs: tuple[int, int], mol: Chem.Mol, max_ring_size: int = 8
) -> dict[str, str]:
    """
    Generate comprehensive SMIRKS-ready descriptors for a bond.

    Parameters
    ----------
    atom_idxs : tuple[int, int]
        Indices of the two atoms forming the bond.
    mol : rdkit.Chem.Mol
        RDKit molecule object containing the bond.
    max_ring_size : int, optional, default=8
        Maximum ring size detected to consider the bond to be in a ring.

    Returns
    -------
    dict[str, str]
        Dictionary containing SMIRKS-formatted bond descriptors:
        - 'bond_smarts': Bond type as SMIRKS symbol ('-', '=', '#', ':', or '~')
        - 'ring_info': Ring membership as ';@' (in ring) or ';!@' (not in ring)

    Examples
    --------
    >>> mol = Chem.MolFromSmiles("CCO")
    >>> bond_desc = get_bond_descriptors((0, 1), mol)  # C-C bond
    >>> bond_desc['bond_smarts']  # '-'
    >>> bond_desc['ring_info']    # ';!@' (not in ring)
    """
    bond = mol.GetBondBetweenAtoms(*atom_idxs)
    if bond is None:
        raise ValueError(f"No bond found between atoms {atom_idxs} in the molecule.")

    # Get the bond order description
    TYPE_TO_SMARTS = {
        Chem.BondType.SINGLE: "-",
        Chem.BondType.DOUBLE: "=",
        Chem.BondType.TRIPLE: "#",
        Chem.BondType.AROMATIC: ":",
    }

    bond_smarts = TYPE_TO_SMARTS.get(bond.GetBondType(), "~")
    # Check if bond is in a ring of size <= max_ring_size
    in_small_ring = False
    for ring in mol.GetRingInfo().BondRings():
        if bond.GetIdx() in ring and len(ring) <= max_ring_size:
            in_small_ring = True
            break

    ring_info = ";@" if in_small_ring else ";!@"

    return {"bond_smarts": bond_smarts, "ring_info": ring_info}


class SMIRKSFactory:
    """
    Factory for generating SMIRKS patterns with configurable specificity levels.

    Provides a unified interface for creating atom and bond SMIRKS patterns
    with different levels of chemical detail and specificity.

    Parameters
    ----------
    atom_include_ring_info : bool, default=False
        Include ring membership information in atom patterns.
    atom_bonded_behavior : BondedAtomBehavior, default=NONE
        How to handle bonded atom information.
    atom_terminal_behavior : TerminalBehavior, default=STANDARD
        How to handle terminal atoms in patterns.
    bond_include_ring_info : bool, default=False
        Include ring membership information in bond patterns.
    bond_specificity : BondSpecificity, default=STANDARD
        Level of detail for bond type specification.

    Examples
    --------
    >>> factory = SMIRKSFactory()
    >>> mol = Chem.MolFromSmiles("CCO")
    >>> atom_smirks = factory.get_atom_smirks(0, 0, mol, (0, 2))
    >>> print(atom_smirks)  # '[#6X4:1]'
    """

    def __init__(
        self,
        # Atom configuration
        atom_include_ring_info: bool = False,
        atom_bonded_behavior: BondedAtomBehavior = BondedAtomBehavior.NONE,
        atom_terminal_behavior: TerminalBehavior = TerminalBehavior.STANDARD,
        # Bond configuration
        bond_include_ring_info: bool = False,
        bond_specificity: BondSpecificity = BondSpecificity.STANDARD,
    ):
        """
        Initialize SMIRKS factory with configuration options.

        Parameters
        ----------
        atom_include_ring_info : bool, default=False
            Whether to include ring membership information in atom patterns.
        atom_bonded_behavior : BondedAtomBehavior, default=BondedAtomBehavior.NONE
            How to handle bonded atom information in patterns.
        atom_terminal_behavior : TerminalBehavior, default=TerminalBehavior.STANDARD
            How to handle terminal atoms in patterns.
        bond_include_ring_info : bool, default=False
            Whether to include ring membership information in bond patterns.
        bond_specificity : BondSpecificity, default=BondSpecificity.STANDARD
            Level of specificity for bond patterns.
        """
        self.atom_config = AtomSMIRKSConfig(
            include_ring_info=atom_include_ring_info,
            bonded_atom_behavior=atom_bonded_behavior,
            terminal_behavior=atom_terminal_behavior,
        )

        self.bond_config = BondSMIRKSConfig(
            include_ring_info=bond_include_ring_info,
            specificity=bond_specificity,
        )

    def get_atom_smirks(
        self,
        at_idx: int,
        at_id: int,
        mol: Chem.Mol,
        terminal_idxs: tuple[int, int],
    ) -> str:
        """
        Generate atom SMIRKS pattern with factory configuration.

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
            SMIRKS atom pattern according to factory configuration.

        Examples
        --------
        >>> factory = SMIRKSFactory()
        >>> mol = Chem.MolFromSmiles("CCO")
        >>> pattern = factory.get_atom_smirks(0, 0, mol, (0, 2))
        >>> print(pattern)  # '[#6X4:1]'
        """
        mol = Chem.AddHs(mol, explicitOnly=False)
        return self._generate_atom_smirks(
            at_idx, at_id, mol, terminal_idxs, self.atom_config
        )

    def get_bond_smirks(
        self,
        atom_idxs: tuple[int, int],
        central_bond: bool,
        mol: Chem.Mol,
    ) -> str:
        """
        Generate bond SMIRKS pattern with factory configuration.

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
            SMIRKS bond pattern according to factory configuration.

        Examples
        --------
        >>> factory = SMIRKSFactory()
        >>> mol = Chem.MolFromSmiles("CCO")
        >>> pattern = factory.get_bond_smirks((0, 1), False, mol)
        >>> print(pattern)  # '-'
        """
        mol = Chem.AddHs(mol, explicitOnly=False)
        return self._generate_bond_smirks(
            atom_idxs, central_bond, mol, self.bond_config
        )

    def create_specificity_level(self, name: str) -> SpecificityLevel:
        """
        Create a SpecificityLevel with the factory's current configuration.

        The returned SpecificityLevel object encapsulates the factory's SMIRKS
        generation methods and is used by molecular component processing pipelines
        to generate consistent SMIRKS patterns for force field parameterization.

        Parameters
        ----------
        name : str
            Name for the specificity level.

        Returns
        -------
        SpecificityLevel
            Configured specificity level for use with molecular components.
            Contains bound methods for atom and bond SMIRKS generation that
            will use this factory's configuration settings.

        Examples
        --------
        Basic usage:
        >>> factory = SMIRKSFactory(atom_include_ring_info=True)
        >>> level = factory.create_specificity_level("RingAware")
        >>> level.name  # 'RingAware'

        Generate SMIRKS patterns:
        >>> mol = Chem.MolFromSmiles("CCO")
        >>> # Generate atom SMIRKS for first carbon (index 0, position 0, terminals at 0,2)
        >>> atom_smirks = level.get_atom_smirks(0, 0, mol, (0, 2))
        >>> print(atom_smirks)  # '[#6X4;!r3;!r4;!r5;!r6;!r7;!r8:1]'
        >>> # Generate bond SMIRKS for C-C bond (atoms 0,1, not central bond)
        >>> bond_smirks = level.get_bond_smirks((0, 1), False, mol)
        >>> print(bond_smirks)  # '-;!@'

        Use in component processing:
        >>> levels = {0: factory.create_specificity_level("Level0")}
        >>> # levels dict is passed to get_mm_components_by_specificity_by_type()
        >>> # which uses level.get_atom_smirks() and level.get_bond_smirks()
        >>> # to generate SMIRKS patterns for molecular components

        Integration with force field generation:
        >>> bond_levels = {
        ...     0: factory1.create_specificity_level("Standard"),
        ...     1: factory2.create_specificity_level("HighSpec")
        ... }
        >>> # Used in SPECIFICITY_LEVELS_BY_COMPONENT dictionary
        >>> # for hierarchical force field parameter generation
        """
        return SpecificityLevel(
            name=name,
            get_atom_smirks=self.get_atom_smirks,
            get_bond_smirks=self.get_bond_smirks,
        )

    @staticmethod
    def _generate_atom_smirks(
        at_idx: int,
        at_id: int,
        mol: Chem.Mol,
        terminal_idxs: tuple[int, int],
        config: AtomSMIRKSConfig,
    ) -> str:
        """
        Core atom SMIRKS generation logic.

        This consolidates all the logic from the various get_atom_smirks_* functions
        into a single configurable implementation.
        """
        ds = get_atom_descriptors(at_idx, mol)
        is_terminal = at_id in terminal_idxs
        ring_part = ds["ring_info"] if config.include_ring_info else ""

        # Handle terminal behavior
        if is_terminal and config.terminal_behavior == TerminalBehavior.WILDCARD:
            base_pattern = f"[*{ring_part}:{at_id + 1}]"
        elif is_terminal and config.terminal_behavior == TerminalBehavior.H_NO_H:
            atomic_num = ds["atomic_num"] if ds["atomic_num"] == "#1" else "!#1"
            base_pattern = f"[{atomic_num}{ring_part}:{at_id + 1}]"
        else:
            # Standard pattern construction
            pattern_parts = [ds["atomic_num"], ds["degree"], ring_part]
            base_pattern = f"[{''.join(pattern_parts)}:{at_id + 1}]"

        # Handle bonded atom information
        if config.bonded_atom_behavior == BondedAtomBehavior.NONE:
            return base_pattern

        # Check if we should skip bonded atoms for terminals when using central-only behavior
        if (
            config.bonded_atom_behavior
            in [
                BondedAtomBehavior.CENTRAL_EXPLICIT_ATOMS,
                BondedAtomBehavior.CENTRAL_EXPLICIT_ATOMS_BONDS,
            ]
            and is_terminal
        ):
            return base_pattern

        # Generate bonded atom patterns
        bonded_atoms = mol.GetAtomWithIdx(at_idx).GetNeighbors()
        bonded_patterns = []

        for bonded_atom in bonded_atoms:
            bonded_idx = bonded_atom.GetIdx()
            bonded_ds = get_atom_descriptors(bonded_idx, mol)

            # Determine bond type
            bond_descriptors = get_bond_descriptors((at_idx, bonded_idx), mol)
            if config.bonded_atom_behavior in [
                BondedAtomBehavior.EXPLICIT_ATOMS_BONDS,
                BondedAtomBehavior.CENTRAL_EXPLICIT_ATOMS_BONDS,
            ]:
                bond_type = bond_descriptors["bond_smarts"]
            else:  # Wildcard behaviors
                bond_type = (
                    "~" + bond_descriptors["ring_info"]
                    if config.include_ring_info
                    else ""
                )

            # Build bonded atom pattern - use bonded atom's ring info, not central atom's
            bonded_pattern_parts = [bonded_ds["atomic_num"], bonded_ds["degree"]]
            if config.include_ring_info:
                bonded_pattern_parts.append(bonded_ds["ring_info"])

            bonded_patterns.append(f"({bond_type}[{''.join(bonded_pattern_parts)}])")

        return base_pattern + "".join(bonded_patterns)

    @staticmethod
    def _generate_bond_smirks(
        atom_idxs: tuple[int, int],
        central_bond: bool,
        mol: Chem.Mol,
        config: BondSMIRKSConfig,
    ) -> str:
        """
        Core bond SMIRKS generation logic.

        This consolidates all the logic from the various get_bond_smirks_* functions
        into a single configurable implementation.
        """
        ds = get_bond_descriptors(atom_idxs, mol)

        # Determine bond type based on specificity
        if config.specificity == BondSpecificity.WILDCARD:
            bond_type = "~"
        elif config.specificity == BondSpecificity.NON_CENTRAL_WILDCARD:
            bond_type = "~" if not central_bond else ds["bond_smarts"]
        else:  # STANDARD
            bond_type = ds["bond_smarts"]

        # Add ring information if requested
        if config.include_ring_info:
            bond_type += ds["ring_info"]

        return bond_type


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

    Examples
    --------
    >>> from openff.toolkit import ForceField
    >>> ff = ForceField("openff-2.0.0.offxml")
    >>> enhanced_ff = add_types_to_ff(ff, component_types, Bond)
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
