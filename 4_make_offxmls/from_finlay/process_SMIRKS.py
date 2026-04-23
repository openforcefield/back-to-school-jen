"""
SMIRKS pattern generation factory for molecular mechanics force fields.

This module generates SMIRKS patterns at configurable specificity levels for
force field parameter assignment. Lower specificity levels create general patterns
that match many molecular environments, while higher levels generate specific
patterns for detailed chemical environments.

Classes
-------
SMIRKSFactory
    Main factory for generating SMIRKS patterns with configurable specificity.
TerminalBehavior
    Enum defining how terminal atoms are handled in patterns.
BondedAtomBehavior
    Enum controlling inclusion of bonded atom information.
BondSpecificity
    Enum specifying bond pattern detail levels.

Functions
---------
get_atom_descriptors
    Extract atomic properties for SMIRKS generation.
get_bond_descriptors
    Extract bond properties for SMIRKS generation.
add_types_to_ff
    Integrate component parameters into OpenFF force fields.
create_specificity_factories
    Build SMIRKSFactory objects from configuration dictionaries.
create_specificity_levels
    Convert factory dictionaries to SpecificityLevel objects.

Examples
--------
>>> factory = SMIRKSFactory(atom_include_ring_info=True)
>>> level = factory.create_specificity_level("RingAware")
>>> mol = Chem.MolFromSmiles("CCO")
>>> atom_pattern = level.get_atom_smirks(0, 0, mol, (0, 2))
>>> bond_pattern = level.get_bond_smirks((0, 1), False, mol)
"""

import os
from copy import deepcopy
from enum import Enum
from typing import Any

from loguru import logger
from dataclasses import dataclass
from rdkit import Chem

from openff.toolkit import ForceField
from openff.toolkit.typing.engines.smirnoff.parameters import ParameterType

from .molecular_classes import (
    MMComponent,
    SpecificityLevel,
    precompute_ff_parameter_cache,
    get_parameters_for_components_cached,
)

TYPE_TO_SMARTS = {
    Chem.BondType.SINGLE: "-",
    Chem.BondType.DOUBLE: "=",
    Chem.BondType.TRIPLE: "#",
    Chem.BondType.AROMATIC: ":",
}


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
    """
    Configuration for atom SMIRKS pattern generation.

    Attributes
    ----------
    include_ring_info : bool, default=False
        Whether to include ring membership in atom patterns (e.g., ";r6" or
        ";!r3;!r4;!r5;!r6;!r7;!r8").
    bonded_atom_behavior : BondedAtomBehavior, default=BondedAtomBehavior.NONE
        Controls how neighboring atom information is encoded via recursive SMARTS.
        Any value other than ``NONE`` requires at least one level of recursion.
        If ``bonded_atom_behavior != NONE`` and ``recursion_level == 0``, the
        recursion level is automatically promoted to 1 and a warning is logged.
    terminal_behavior : TerminalBehavior, default=TerminalBehavior.STANDARD
        How terminal atoms (first/last in a component) are represented in patterns.
    recursion_level : int, default=0
        Depth of recursive neighbor encoding in atom SMIRKS patterns. 0 encodes
        only the atom itself; 1 encodes immediate neighbors; 2 encodes neighbors
        of neighbors, etc. Automatically promoted to 1 if ``bonded_atom_behavior``
        is not ``NONE`` and this is left at 0.
    """

    include_ring_info: bool = False
    bonded_atom_behavior: BondedAtomBehavior = BondedAtomBehavior.NONE
    terminal_behavior: TerminalBehavior = TerminalBehavior.STANDARD
    recursion_level: int = 0

    def __post_init__(self):
        """Validate configuration consistency and auto-promote recursion_level."""
        if not isinstance(self.bonded_atom_behavior, BondedAtomBehavior):
            raise ValueError(
                f"Invalid bonded_atom_behavior: {self.bonded_atom_behavior}"
            )
        if not isinstance(self.terminal_behavior, TerminalBehavior):
            raise ValueError(f"Invalid terminal_behavior: {self.terminal_behavior}")
        if (
            self.bonded_atom_behavior != BondedAtomBehavior.NONE
            and self.recursion_level == 0
        ):
            logger.warning(
                f"bonded_atom_behavior={self.bonded_atom_behavior.value!r} requires "
                "recursion to encode neighbor information, but recursion_level=0. "
                "Automatically promoting recursion_level to 1."
            )
            self.recursion_level = 1


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
    Extract atomic properties for SMIRKS pattern generation.

    Parameters
    ----------
    at_idx : int
        Atom index in the molecule.
    mol : rdkit.Chem.Mol
        RDKit molecule object.

    Returns
    -------
    dict[str, str]
        Atomic descriptors with keys:
        - 'atomic_num': "#6" (atomic number)
        - 'degree': "X4" (connectivity)
        - 'charge': "+1" or "-2" (formal charge)
        - 'ring_info': ";rN" where N is between 3 and 8 or ";!r3;!r4;!r5;!r6;!r7;!r8"
        - 'aromaticity': ";a" or ";A"

    Examples
    --------
    >>> mol = Chem.MolFromSmiles("CCO")
    >>> get_atom_descriptors(0, mol)
    {'atomic_num': '#6', 'degree': 'X4', ...}
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
        if descriptors["charge"] > 0
        else str(descriptors["charge"])
    )

    return descriptors


def get_bond_descriptors(
    atom_idxs: tuple[int, int], mol: Chem.Mol, max_ring_size: int = 8
) -> dict[str, str]:
    """
    Extract bond properties for SMIRKS pattern generation.

    Parameters
    ----------
    atom_idxs : tuple[int, int]
        Indices of bonded atoms.
    mol : rdkit.Chem.Mol
        RDKit molecule object.
    max_ring_size : int, default=8
        Maximum ring size to consider as "in ring".

    Returns
    -------
    dict[str, str]
        Bond descriptors with keys:
        - 'bond_smarts': "-", "=", "#", ":", or "~"
        - 'ring_info': ";@" (in ring ≤max_ring_size) or "" (not in ring)

    Examples
    --------
    >>> mol = Chem.MolFromSmiles("CCO")
    >>> get_bond_descriptors((0, 1), mol)
    {'bond_smarts': '-', 'ring_info': ''}
    """
    bond = mol.GetBondBetweenAtoms(*atom_idxs)
    if bond is None:
        raise ValueError(f"No bond found between atoms {atom_idxs} in the molecule.")

    bond_smarts = TYPE_TO_SMARTS.get(bond.GetBondType(), "~")
    # Check if bond is in a ring of size <= max_ring_size
    in_small_ring = False
    for ring in mol.GetRingInfo().BondRings():
        if bond.GetIdx() in ring and len(ring) <= max_ring_size:
            in_small_ring = True
            break

    # Can't use ";!@" otherwise rings larger than 8 are not parametrized
    ring_info = ";@" if in_small_ring else ""

    return {"bond_smarts": bond_smarts, "ring_info": ring_info}


def get_atom_recursive_smirks(
    idx: int,
    mol: Chem.Mol,
    recursion_level: int = 1,
    return_recursions: bool = False,
    include_bond_order: bool = True,
    include_ring_info: bool = False,
) -> str | list[str]:
    """
    Generate recursive SMIRKS pattern encoding atom identity and bonded connectivity.

    Builds an atom pattern that uses SMARTS recursive queries to capture the local
    bonded environment up to `recursion_level` bonds away.  Each neighbor is encoded
    as ``&$([current_atom]~[neighbor_pattern])`` so the query correctly tests for
    connectivity (bonds) rather than the (incorrect) ``&$([neighbor_pattern])`` form
    which would require the current atom to simultaneously *be* the neighbor atom type.

    Parameters
    ----------
    idx : int
        Atom index in *mol*.
    mol : rdkit.Chem.Mol
        RDKit molecule (should already have explicit Hs added if needed).
    recursion_level : int, default=1
        Depth of recursive neighbor encoding.  0 returns only the base atom pattern.
    return_recursions : bool, default=False
        When True, return the list of ``&$(...)`` recursion strings instead of the
        fully assembled atom SMIRKS.  Used internally by :func:`SMIRKSFactory._generate_atom_smirks`.

    Returns
    -------
    str | list[str]
        Full atom SMIRKS string (``return_recursions=False``) such as
        ``[$([#6X4]~[#7X4+1])]``, or the raw list of recursion strings
        (``return_recursions=True``) such as ``["&$([#6X4+0]~[#7X4+1])", ...]``.

    Examples
    --------
    >>> from rdkit import Chem
    >>> mol = Chem.AddHs(Chem.MolFromSmiles("CN"))
    >>> get_atom_recursive_smirks(0, mol, recursion_level=1)
    '[#6X4+0&$([#6X4+0]~[#1X1+0])&$([#6X4+0]~[#1X1+0])&$([#6X4+0]~[#1X1+0])]'
    """
    ds = get_atom_descriptors(idx, mol)
    if ds["charge"] == str(0):
        ds["charge"] = ""  # remove charge
    base = f"[{ds['atomic_num']}{ds['degree']}]"

    if recursion_level <= 0:
        if return_recursions:
            return []
        else:
            return base

    bonded_atoms = mol.GetAtomWithIdx(idx).GetNeighbors()
    recursions = []
    for bonded_atom in bonded_atoms:
        bonded_idx = bonded_atom.GetIdx()
        neighbor = get_atom_recursive_smirks(
            bonded_idx,
            mol,
            recursion_level=recursion_level - 1,
            include_bond_order=include_bond_order,
            include_ring_info=include_ring_info,
        )
        # include explicit bond order between the current atom and the neighbor
        if include_bond_order:
            bond = mol.GetBondBetweenAtoms(idx, bonded_idx)
            bond_smarts = TYPE_TO_SMARTS.get(bond.GetBondType(), "~")
        else:
            bond_smarts = "~"  # always wildcard when order not requested

        if include_ring_info:
            bond_ds = get_bond_descriptors((idx, bonded_idx), mol)
            bond_smarts += bond_ds["ring_info"]

        recursions.append(f"{bond_smarts}{neighbor}")

    # Sort so that chemically equivalent atoms produce the same string
    # regardless of the arbitrary neighbor iteration order from RDKit.
    recursions.sort()

    if return_recursions:
        return recursions
    else:
        if recursions:
            rec_str = "".join(f"({r})" for r in recursions[:-1]) + recursions[-1]
            base = "[$(" + base + rec_str + ")]"
        return base


class SMIRKSFactory:
    """
    Factory for generating SMIRKS patterns with configurable specificity.

    Parameters
    ----------
    atom_include_ring_info : bool, default=False
        Include ring membership in atom patterns.
    atom_bonded_behavior : BondedAtomBehavior, default=NONE
        How to handle bonded atom information.
    atom_terminal_behavior : TerminalBehavior, default=STANDARD
        How to handle terminal atoms.
    atom_recursion_level : int, default=0
        Depth of recursive neighbor encoding. 0 encodes only the atom itself;
        1 encodes immediate neighbors, etc. Automatically promoted to 1 if
        ``atom_bonded_behavior`` is not ``NONE`` and this is left at 0.
    bond_include_ring_info : bool, default=False
        Include ring membership in bond patterns.
    bond_specificity : BondSpecificity, default=STANDARD
        Level of bond type specification.

    Examples
    --------
    >>> factory = SMIRKSFactory(atom_include_ring_info=True)
    >>> mol = Chem.MolFromSmiles("CCO")
    >>> pattern = factory.get_atom_smirks(0, 0, mol, (0, 2))
    """

    def __init__(
        self,
        # Atom configuration
        atom_include_ring_info: bool = False,
        atom_bonded_behavior: BondedAtomBehavior = BondedAtomBehavior.NONE,
        atom_terminal_behavior: TerminalBehavior = TerminalBehavior.STANDARD,
        atom_recursion_level: int = 0,
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
        atom_recursion_level : int, default=0
            Depth of recursive neighbor encoding. 0 encodes only the atom itself;
            1 encodes immediate neighbors, etc. Automatically promoted to 1 if
            ``atom_bonded_behavior`` is not ``NONE`` and this is left at 0.
        bond_include_ring_info : bool, default=False
            Whether to include ring membership information in bond patterns.
        bond_specificity : BondSpecificity, default=BondSpecificity.STANDARD
            Level of specificity for bond patterns.
        """
        self.atom_config = AtomSMIRKSConfig(
            include_ring_info=atom_include_ring_info,
            bonded_atom_behavior=atom_bonded_behavior,
            terminal_behavior=atom_terminal_behavior,
            recursion_level=atom_recursion_level,
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
        Generate atom SMIRKS pattern.

        Parameters
        ----------
        at_idx : int
            Atom index in molecule.
        at_id : int
            Position in component (0-based).
        mol : rdkit.Chem.Mol
            RDKit molecule object.
        terminal_idxs : tuple[int, int]
            Indices of terminal atoms.

        Returns
        -------
        str
            SMIRKS atom pattern, e.g., "[#6X4:1]".
        """
        mol = Chem.AddHs(mol, explicitOnly=False)
        return self._generate_atom_smirks(
            at_idx,
            at_id,
            mol,
            terminal_idxs,
            self.atom_config,
            bond_include_ring_info=self.bond_config.include_ring_info,
        )

    def get_bond_smirks(
        self,
        atom_idxs: tuple[int, int],
        central_bond: bool,
        mol: Chem.Mol,
    ) -> str:
        """
        Generate bond SMIRKS pattern.

        Parameters
        ----------
        atom_idxs : tuple[int, int]
            Indices of bonded atoms.
        central_bond : bool
            Whether this is the central bond in the component.
        mol : rdkit.Chem.Mol
            RDKit molecule object.

        Returns
        -------
        str
            SMIRKS bond pattern, e.g., "-" or "~;@".
        """
        mol = Chem.AddHs(mol, explicitOnly=False)
        return self._generate_bond_smirks(
            atom_idxs, central_bond, mol, self.bond_config
        )

    def create_specificity_level(self, name: str) -> SpecificityLevel:
        """
        Create SpecificityLevel with current factory configuration.

        Parameters
        ----------
        name : str
            Name for the specificity level.

        Returns
        -------
        SpecificityLevel
            Configured level with bound SMIRKS generation methods.

        Examples
        --------
        >>> factory = SMIRKSFactory(atom_include_ring_info=True)
        >>> level = factory.create_specificity_level("RingAware")
        >>> level.name
        'RingAware'
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
        bond_include_ring_info: bool = False,
    ) -> str:
        """
        Core atom SMIRKS generation logic.

        This consolidates all the logic from the various get_atom_smirks_* functions
        into a single configurable implementation.
        """
        ds = get_atom_descriptors(at_idx, mol)
        if ds["charge"] == str(0):
            ds["charge"] = ""
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

        if config.recursion_level > 0:
            include_bond_order = (
                True
                if config.bonded_atom_behavior
                in [
                    BondedAtomBehavior.EXPLICIT_ATOMS_BONDS,
                    BondedAtomBehavior.CENTRAL_EXPLICIT_ATOMS_BONDS,
                ]
                else False
            )
            recursions = get_atom_recursive_smirks(
                at_idx,
                mol,
                recursion_level=config.recursion_level,
                return_recursions=True,
                include_bond_order=include_bond_order,
                include_ring_info=bond_include_ring_info,
            )

        # Handle terminal behavior
        if is_terminal and config.terminal_behavior == TerminalBehavior.WILDCARD:
            atom_base = f"*{ring_part}"
        elif is_terminal and config.terminal_behavior == TerminalBehavior.H_NO_H:
            atomic_num = ds["atomic_num"] if ds["atomic_num"] == "#1" else "!#1"
            atom_base = f"{atomic_num}{ring_part}"
        else:
            atom_base = f"{ds['atomic_num']}{ds['degree']}{ring_part}"

        if config.recursion_level > 0 and recursions:
            branches = "".join(f"({r})" for r in recursions)
            rec_str = f"&$([{atom_base}]{branches})"
        else:
            rec_str = ""

        base_pattern = f"[{atom_base}{rec_str}:{at_id + 1}]"

        return base_pattern

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


# Module-level globals set once per worker via Pool initializer.
# These remain available for any future parallel helpers that still use a pool.
_worker_ff = None
_worker_component_class = None


def _worker_init(ff, component_class):
    """
    Pool worker initializer — stores shared objects in module globals.

    Called once per worker process so that ``ff`` and ``component_class``
    are not re-pickled for every task.
    """
    global _worker_ff, _worker_component_class
    _worker_ff = ff
    _worker_component_class = component_class


def _process_component_for_ff(args):
    """
    Helper function for parallel processing in add_types_to_ff.

    Must be defined at module level for multiprocessing compatibility.

    Parameters
    ----------
    args : tuple
        (i, smirks, sampled_components, component_count, specificity_num)

    Returns
    -------
    ParameterType
        Generated parameter for the component.
    """
    i, smirks, components, component_count, specificity_num = args
    return _worker_component_class.get_parameter(
        smirks,
        specificity_num,
        components,
        i,
        _worker_ff,
        component_count=component_count,
    )


def add_types_to_ff(
    ff: ForceField,
    component_types: dict[int, dict[str, list[MMComponent]]],
    component_class: type[MMComponent],
    extra_parameters: list[ParameterType] | None = None,
    n_workers: int | None = None,
    base_ff: ForceField | None = None,
) -> ForceField:
    """
    Add component parameters to a force field.

    Parameters
    ----------
    ff : openff.toolkit.ForceField
        Base force field to extend (may already have accumulated parameters
        from prior component passes).
    component_types : dict[int, dict[str, list[MMComponent]]]
        Component organization: {specificity_level: {smirks: [components]}}.
    component_class : type[MMComponent]
        Component type (Bond, Angle, etc.).
    extra_parameters : list[ParameterType], optional
        Additional parameters to append.
    n_workers : int, optional
        Number of worker processes used for pre-computing the parameter cache.
        If None, uses all CPU cores.
    base_ff : openff.toolkit.ForceField, optional
        The original compact template force field used exclusively for
        label_molecules calls in the cache-build phase.  Must be provided
        when ``ff`` has already been extended with parameters from earlier
        component passes (e.g. ff already contains Bond parameters when
        processing Angles), otherwise label_molecules will evaluate every
        accumulated SMIRKS pattern per molecule, causing a severe hang.
        If None, ``ff`` is used (correct only for the first component pass).

    Returns
    -------
    openff.toolkit.ForceField
        New force field with added parameters.

    Notes
    -----
    The assembly uses a two-phase strategy to avoid serialising full
    ``Molecule`` objects over IPC for every SMIRKS type:

    1. **Cache phase** – ``forcefield.label_molecules`` is called once per
       *unique* molecule across all sampled components (using a ``fork``
       process pool).  Results are stored in a ``(mapped_smiles, indices)``
       keyed dict so downstream lookups are O(1).
    2. **Assembly phase** – Parameter objects are constructed serially on the
       main process using cache lookups.  ``ParameterType`` construction is
       O(microseconds) per type, so 500k types finish in seconds rather than
       the hours required when each task spawns a full round-trip to a worker.

    This eliminates the previous deadlock/hang caused by:
    - Pickling ``Molecule`` objects (each 10–100 KB) into the OS IPC pipe for
      every one of 500k+ tasks, causing pipe-buffer saturation.
    - ``spawn`` context forcing a full Python re-import and FF deserialisation
      in each worker for every ``Pool`` creation.

    Examples
    --------
    >>> ff = ForceField("openff-2.0.0.offxml")
    >>> enhanced_ff = add_types_to_ff(ff, component_types, Bond)
    """
    ff_copy = deepcopy(ff)
    handler = component_class.handler_class(version=component_class.handler_version)

    if n_workers is None:
        n_workers = os.cpu_count() or 1

    # ------------------------------------------------------------------
    # Phase 1: pre-compute base-FF parameter cache.
    # label_molecules is called once per unique molecule (fork pool),
    # not once per SMIRKS type × 10 samples.
    # Use base_ff (the compact template FF) if provided, so that
    # label_molecules does not evaluate the large set of accumulated
    # Bond/Angle SMIRKS patterns from earlier component passes.
    # ------------------------------------------------------------------
    cache_ff = base_ff if base_ff is not None else ff
    param_cache = precompute_ff_parameter_cache(
        component_types, cache_ff, component_class, n_workers=n_workers
    )

    # ------------------------------------------------------------------
    # Phase 2: serial parameter assembly using O(1) cache lookups.
    # ParameterType construction is trivially fast (microseconds each),
    # so there is no benefit to adding IPC overhead here.
    # ------------------------------------------------------------------
    for specificity_num, components_by_type in sorted(
        component_types.items(), key=lambda item: item[0]
    ):
        ordered_items = sorted(
            components_by_type.items(),
            key=lambda item: (
                item[0].count(";@"),
                -len(item[1]),
            ),
        )
        logger.info(f"Finished pre-sort of {component_class}.")

        total_items = len(ordered_items)
        next_log_threshold = 0.05
        processed = 0

        logger.info(
            f"Adding parameters for specificity {specificity_num} "
            f"({total_items} items)..."
        )

        for i, (smirks, components) in enumerate(ordered_items):
            cached_params = get_parameters_for_components_cached(
                components, param_cache
            )
            parameter = component_class.get_parameter(
                smirks,
                specificity_num,
                components,
                i,
                ff,
                component_count=len(components),
                cached_params=cached_params,
            )
            handler.parameters.append(parameter)
            processed += 1

            progress = processed / total_items
            if progress >= next_log_threshold:
                logger.info(
                    f"Progress: {processed}/{total_items} parameters ({progress*100:.1f}%)"
                )
                next_log_threshold += 0.05

    # Add any extra parameters at the end
    if extra_parameters:
        for parameter in extra_parameters:
            handler.parameters.append(parameter)

    tag_name = component_class.handler_class._TAGNAME
    if tag_name is not None:
        ff_copy.deregister_parameter_handler(tag_name)
    ff_copy.register_parameter_handler(handler)

    return ff_copy


def create_specificity_factories(
    config: dict
) -> tuple[dict[str, SMIRKSFactory] | None, dict[str, SMIRKSFactory] | None]:
    """
    Create SMIRKSFactory objects from configuration dictionary.

    Parameters
    ----------
    config : dict
        Configuration with 'bond_specificities' and 'angle_specificities' keys.
        Each maps names to factory configuration dictionaries containing:

        **Atom Configuration Options:**
        - atom_include_ring_info : bool
            Include ring membership (e.g., ";r6" or ";!r3;!r4;!r5;!r6;!r7;!r8")
        - atom_bonded_behavior : str
            "NONE", "CENTRAL_EXPLICIT_ATOMS", "CENTRAL_EXPLICIT_ATOMS_BONDS",
            "EXPLICIT_ATOMS", or "EXPLICIT_ATOMS_BONDS"
        - atom_terminal_behavior : str
            "STANDARD" ([#6X4:1]), "WILDCARD" ([*:1]), or "H_NO_H" ([#1:1]/[!#1:1])
        - atom_resursion_level : int, default=0
            Level of recursion in definitions

        **Bond Configuration Options:**
        - bond_include_ring_info : bool
            Include ring membership (";@" for in-ring bonds)
        - bond_specificity : str
            "STANDARD" (explicit: "-", "="), "NON_CENTRAL_WILDCARD" (mixed),
            or "WILDCARD" (all "~")

    Returns
    -------
    tuple[dict[str, SMIRKSFactory], dict[str, SMIRKSFactory]]
        (bond_specificities, angle_specificities) with factory objects.

    Examples
    --------
    >>> config = {
    ...     "bond_specificities": {
    ...         "Standard": {
    ...             "atom_include_ring_info": True,
    ...             "bond_specificity": "WILDCARD"
    ...         }
    ...     },
    ...     "angle_specificities": {
    ...         "Terminal": {
    ...             "atom_terminal_behavior": "WILDCARD",
    ...             "bond_specificity": "STANDARD"
    ...         }
    ...     }
    ... }
    >>> bond_specs, angle_specs = create_specificity_factories(config)
    >>> len(bond_specs)
    1
    """

    def config_to_factory(spec_config: dict[str, Any]) -> SMIRKSFactory:
        """Convert config dict to SMIRKSFactory with proper enum conversion."""
        factory_args = spec_config.copy()

        # Convert string enums to actual enum objects
        if "atom_terminal_behavior" in factory_args:
            factory_args["atom_terminal_behavior"] = getattr(
                TerminalBehavior, factory_args["atom_terminal_behavior"]
            )

        if "atom_bonded_behavior" in factory_args:
            factory_args["atom_bonded_behavior"] = getattr(
                BondedAtomBehavior, factory_args["atom_bonded_behavior"]
            )

        if "bond_specificity" in factory_args:
            factory_args["bond_specificity"] = getattr(
                BondSpecificity, factory_args["bond_specificity"]
            )

        return SMIRKSFactory(**factory_args)

    if "bond_specificities" in config:
        bond_specificities = {
            name: config_to_factory(spec_config)
            for name, spec_config in config["bond_specificities"].items()
        }
    else:
        bond_specificities = None

    if "angle_specificities" in config:
        angle_specificities = {
            name: config_to_factory(spec_config)
            for name, spec_config in config["angle_specificities"].items()
        }
    else:
        angle_specificities = None

    return bond_specificities, angle_specificities


def create_specificity_levels(
    specificities_by_component: dict[type[MMComponent], dict[str, SMIRKSFactory]]
) -> dict[type[MMComponent], dict[int, SpecificityLevel]]:
    """
    Create SPECIFICITY_LEVELS_BY_COMPONENT from specificity dictionaries.

    Parameters
    ----------
    specificities_by_component : dict[type[MMComponent], dict[str, SMIRKSFactory]]
        Mapping of component types to their specificity dictionaries.

    Returns
    -------
    dict[type[MMComponent], dict[int, SpecificityLevel]]
        Structure: {ComponentClass: {level_index: SpecificityLevel}}.

    Examples
    --------
    >>> specificities = {
    ...     Bond: {"Standard": SMIRKSFactory()},
    ...     Angle: {"Terminal": SMIRKSFactory()}
    ... }
    >>> levels = create_specificity_levels(specificities)
    >>> levels[Bond][0].name
    '0:Standard'
    """
    specificities: dict[type[MMComponent], dict[int, SpecificityLevel]] = {}

    for component_class, specificity_dict in specificities_by_component.items():
        specificities[component_class] = {
            i: factory.create_specificity_level(f"{i}:" + name)
            for i, (name, factory) in enumerate(specificity_dict.items())
        }

    return specificities
