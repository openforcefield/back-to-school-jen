"""
Molecular Mechanics Component Classes for Force Field Parameter Generation.

This module provides classes and utilities for representing molecular mechanics (MM)
components such as bonds, angles, and torsions, with functionality to generate
SMIRKS patterns and force field parameters.

Key Components
--------------
MMComponent : Abstract base class
    Base class for all molecular mechanics components with shared functionality
    for SMIRKS generation and parameter assignment.

Bond, Angle, ProperTorsion, ImproperTorsion : Concrete MM component classes
    Specific implementations for different component types with their respective
    parameter handlers and physics.

Examples
--------
Creating and analyzing bond components:

>>> from openff.toolkit import Molecule
>>> mol = Molecule.from_smiles("CCO")
>>> bond_indices = (0, 1)  # C-C bond
>>> bond = Bond(indices=bond_indices, mol=mol, rdkit_mol=mol.to_rdkit())
>>> print(f"Bond: {bond.indices}")
Bond: (0, 1)

Generating SMIRKS patterns for different specificity levels:

>>> from process_SMIRKS import SMIRKSFactory
>>> factory = SMIRKSFactory()
>>> specificity = factory.create_specificity_level("RingAware")
>>> smirks = bond.get_smirks(specificity)
>>> print(f"SMIRKS: {smirks}")
SMIRKS: [#6X4:1]-[#6X4:2]

Creating force field parameters from component groups:

>>> bonds = [Bond(...), Bond(...)]  # Multiple bond components
>>> base_ff = ForceField("openff-2.0.0.offxml")
>>> parameter = Bond.get_parameter(
...     smirks="[#6:1]-[#6:2]",
...     specificity_num=0,
...     components=bonds,
...     index=0,
...     base_ff=base_ff
... )
>>> print(f"Parameter length: {parameter.length}")
Parameter length: 1.52 angstrom
"""

from typing import Optional, Callable, cast
from abc import ABC, abstractmethod
from collections import namedtuple
import os
from multiprocessing import get_context as mp_get_context
from multiprocessing import TimeoutError as MPTimeoutError
import re

import numpy as np
from loguru import logger
from rdkit import Chem

from openff.units import unit as off_unit
from openff.toolkit import Molecule, ForceField
from openff.toolkit.typing.engines.smirnoff.parameters import (
    ParameterHandler,
    ParameterType,
    ParameterList,
    BondHandler,
    AngleHandler,
    ProperTorsionHandler,
    ImproperTorsionHandler,
)

SpecificityLevel = namedtuple(
    "SpecificityLevel", ["name", "get_atom_smirks", "get_bond_smirks"]
)


def get_bond_idxs(mol: Molecule) -> set[tuple[int, int]]:
    """
    Extract bond atom indices from a molecule.

    Parameters
    ----------
    mol : openff.toolkit.Molecule
        The input molecule to analyze.

    Returns
    -------
    set[tuple[int, int]]
        Set of tuples containing the atom indices of each bond,
        with indices sorted within each tuple to ensure consistency.

    Examples
    --------
    >>> from openff.toolkit import Molecule
    >>> mol = Molecule.from_smiles("CCO")
    >>> bonds = get_bond_idxs(mol)
    >>> print(bonds)
    {(0, 1), (1, 2)}
    """
    return cast(
        set[tuple[int, int]],
        {
            tuple(sorted((a.molecule_atom_index for a in bond.atoms)))
            for bond in mol.bonds
        },
    )


def get_angle_idxs(mol: Molecule) -> set[tuple[int, int, int]]:
    """
    Extract angle atom indices from a molecule.

    Parameters
    ----------
    mol : openff.toolkit.Molecule
        The input molecule to analyze.

    Returns
    -------
    set[tuple[int, int, int]]
        Set of tuples containing the atom indices of each angle in the order
        (terminal, central, terminal).

    Examples
    --------
    >>> from openff.toolkit import Molecule
    >>> mol = Molecule.from_smiles("CCO")
    >>> angles = get_angle_idxs(mol)
    >>> print(angles)
    {(0, 1, 2)}
    """
    return cast(
        set[tuple[int, int, int]],
        {tuple(a.molecule_atom_index for a in angle) for angle in mol.angles},
    )


def get_proper_torsion_idxs(mol: Molecule) -> set[tuple[int, int, int, int]]:
    """
    Extract proper torsion atom indices from a molecule.

    Parameters
    ----------
    mol : openff.toolkit.Molecule
        The input molecule to analyze.

    Returns
    -------
    set[tuple[int, int, int, int]]
        Set of tuples containing the atom indices of each proper torsion
        in the order (atom1, atom2, atom3, atom4) where atom2-atom3
        is the central bond.

    Examples
    --------
    >>> from openff.toolkit import Molecule
    >>> mol = Molecule.from_smiles("CCCO")
    >>> torsions = get_proper_torsion_idxs(mol)
    >>> print(torsions)
    {(0, 1, 2, 3)}
    """
    return cast(
        set[tuple[int, int, int, int]],
        {tuple(a.molecule_atom_index for a in torsion) for torsion in mol.propers},
    )


def get_improper_torsion_idxs(mol: Molecule) -> set[tuple[int, int, int, int]]:
    """
    Extract improper torsion atom indices from a molecule.

    Parameters
    ----------
    mol : openff.toolkit.Molecule
        The input molecule to analyze.

    Returns
    -------
    set[tuple[int, int, int, int]]
        Set of tuples containing the atom indices of each improper torsion.
        The third atom (index 2) is always the central atom.

    Examples
    --------
    >>> from openff.toolkit import Molecule
    >>> mol = Molecule.from_smiles("C(C)(C)C")  # tetrahedral carbon
    >>> impropers = get_improper_torsion_idxs(mol)
    """
    return cast(
        set[tuple[int, int, int, int]],
        {tuple(a.molecule_atom_index for a in torsion) for torsion in mol.impropers},
    )


class MMComponent(ABC):
    """
    Abstract base class for molecular mechanics components.

    Represents force field components such as bonds, angles, and torsions with
    functionality to generate SMIRKS patterns and create force field parameters.

    Parameters
    ----------
    indices : tuple[int, ...]
        Atom indices defining the component (e.g., (0, 1) for a bond).
    mol : openff.toolkit.Molecule
        The OpenFF Molecule object containing this component.
    rdkit_mol : rdkit.Chem.Mol
        The RDKit representation of the molecule for chemical analysis.

    Attributes
    ----------
    n_atoms : int
        Number of atoms in the component (must be defined by subclasses).
    handler_class : type[ParameterHandler]
        SMIRNOFF parameter handler class (must be defined by subclasses).
    handler_version : float
        Version of the parameter handler (must be defined by subclasses).
    parameter_type : type[ParameterType]
        SMIRNOFF parameter type class (must be defined by subclasses).
    getter_fn : Callable[[Molecule], set[tuple[int, ...]]]
        Function to extract components from molecules (must be defined by subclasses).

    Notes
    -----
    This is an abstract base class that enforces subclasses to define required
    class attributes. The class uses __slots__ for memory efficiency.

    All MMComponent instances automatically generate mapped SMILES strings
    during initialization and provide methods for SMIRKS pattern generation
    and parameter creation.

    Examples
    --------
    Subclasses must define all required class attributes:

    >>> class Bond(MMComponent):
    ...     n_atoms = 2
    ...     handler_class = BondHandler
    ...     handler_version = 0.3
    ...     parameter_type = BondHandler.BondType
    ...     getter_fn = get_bond_idxs

    Creating a component instance:

    >>> from openff.toolkit import Molecule
    >>> mol = Molecule.from_smiles("CCO")
    >>> bond = Bond(indices=(0, 1), mol=mol, rdkit_mol=mol.to_rdkit())
    >>> print(bond.mapped_smiles)
    [C:1][C:2][O:3]
    """

    # __slots__ = ["mapped_smiles", "indices", "mol", "rdkit_mol", "n_atoms", "handler_class", "parameter_type", "getter_fn"]
    __slots__ = ["mapped_smiles", "indices", "mol", "rdkit_mol", "n_atoms"]

    mapped_smiles: str
    indices: tuple[int, ...]
    mol: Molecule
    rdkit_mol: Chem.Mol
    n_atoms: int  # Subclass must define number of atoms
    handler_class: type[ParameterHandler]  # Subclass must define handler class
    handler_version: float
    parameter_type: type[ParameterType]  # Subclass must define parameter type
    getter_fn: Callable[
        [Molecule], set[tuple[int, ...]]
    ]  # Subclass must define getter function

    def __init_subclass__(cls, **kwargs):
        """
        Enforce that required class attributes are defined in subclasses.

        Parameters
        ----------
        **kwargs
            Additional keyword arguments passed to parent class.

        Raises
        ------
        TypeError
            If any required class attribute is missing from the subclass.
        """
        super().__init_subclass__(**kwargs)

        required_attrs = [
            "n_atoms",
            "handler_class",
            "handler_version",
            "parameter_type",
            "getter_fn",
        ]

        for attr in required_attrs:
            if not hasattr(cls, attr):
                raise TypeError(f"{cls.__name__} must define class attribute '{attr}'")

    def __init__(self, indices: tuple[int, ...], mol: Molecule, rdkit_mol: Chem.Mol):
        """
        Initialize the molecular mechanics component.

        Parameters
        ----------
        indices : tuple[int, ...]
            Atom indices defining the component.
        mol : openff.toolkit.Molecule
            OpenFF Molecule object.
        rdkit_mol : rdkit.Chem.Mol
            RDKit molecule representation.
        """
        self.indices = indices
        self.mol = mol
        self.rdkit_mol = rdkit_mol

        # Apply MDL aromaticity model
        Chem.SetAromaticity(self.rdkit_mol, Chem.AromaticityModel.AROMATICITY_MDL)
        self.mapped_smiles = mol.to_smiles(mapped=True)

    @property
    def central_bond_index(self) -> Optional[int]:
        """
        Get the index of the central bond in the component.

        Returns
        -------
        Optional[int]
            Index of the central bond for 4-atom components (torsions),
            None for other component types.
        """
        if self.n_atoms == 4:
            return 1
        return None

    @property
    def terminal_atom_indices(self) -> tuple[int, int]:
        """
        Get the indices of terminal atoms in the component.

        Returns
        -------
        tuple[int, int]
            Indices of the first and last atoms in the component.
        """
        return (0, self.n_atoms - 1)

    def _construct_smirks(self, atoms: list[str], bonds: list[str]) -> str:
        """
        Construct SMIRKS representation from atom and bond patterns.

        Parameters
        ----------
        atoms : list[str]
            List of atom SMIRKS patterns.
        bonds : list[str]
            List of bond SMIRKS patterns.

        Returns
        -------
        str
            Complete SMIRKS pattern for the component.

        Raises
        ------
        AssertionError
            If the number of atoms or bonds is inconsistent with component type.
        """
        # Make sure that the lengths of atoms and bonds are consistent
        assert (
            len(atoms) == self.n_atoms
        ), f"Expected {self.n_atoms} atoms, got {len(atoms)}"
        assert (
            len(bonds) == self.n_atoms - 1
        ), f"Expected {self.n_atoms - 1} bonds, got {len(bonds)}"

        smirks = atoms[0]
        for bond, atom in zip(bonds, atoms[1:]):
            smirks += f"{bond}{atom}"

        return smirks

    def get_smirks(self, specificity_level: SpecificityLevel) -> str:
        """
        Generate SMIRKS representation for the component.

        Parameters
        ----------
        specificity_level : SpecificityLevel
            Defines the level of detail for atom and bond descriptions.

        Returns
        -------
        str
            SMIRKS pattern representing this component. Returns the
            lexicographically smallest pattern to ensure consistency
            regardless of atom ordering.
        """
        idxs = self.indices
        n = self.n_atoms

        atoms_fwd = [
            specificity_level.get_atom_smirks(
                at_idx,
                at_id,
                self.rdkit_mol,
                self.terminal_atom_indices,
            )
            for at_idx, at_id in zip(idxs, range(n))
        ]
        atoms_bwd = [
            specificity_level.get_atom_smirks(
                at_idx,
                at_id,
                self.rdkit_mol,
                self.terminal_atom_indices,
            )
            for at_idx, at_id in zip(reversed(idxs), range(n))
        ]

        bonds_fwd = [
            specificity_level.get_bond_smirks(
                (idxs[j], idxs[j + 1]),
                central_bond=(j == self.central_bond_index),
                mol=self.rdkit_mol,
            )
            for j in range(n - 1)
        ]

        smirks_fwd = self._construct_smirks(atoms_fwd, bonds_fwd)
        smirks_bwd = self._construct_smirks(atoms_bwd, list(reversed(bonds_fwd)))

        # Return the lexicographically smallest SMIRKS representation to ensure
        # that the order of atoms does not affect the representation.
        return min(smirks_fwd, smirks_bwd)

    def matches_smirks(self, smirks: str) -> bool:
        """
        Check if component matches a given SMIRKS pattern.

        Parameters
        ----------
        smirks : str
            SMIRKS pattern to test against.

        Returns
        -------
        bool
            True if this component matches the SMIRKS pattern, False otherwise.
        """
        indices_list = self.mol.chemical_environment_matches(smirks)

        if not indices_list:
            return False

        for indices in indices_list:
            if self.indices == indices:
                return True

        return False

    def __str__(self):
        """Return string representation of the component."""
        return f"{self.mapped_smiles} ({self.indices})"

    @classmethod
    @abstractmethod
    def get_parameter(
        cls,
        smirks: str,
        specificity_num: int,
        components: list["MMComponent"],
        index: int,
        base_ff: ForceField,
        component_count: int | None = None,
        cached_params: list | None = None,
    ) -> ParameterType:
        """
        Generate force field parameter for component type.

        Parameters
        ----------
        smirks : str
            SMIRKS pattern defining the parameter scope.
        specificity_num : int
            Specificity level number for parameter identification.
        components : list[MMComponent]
            List of components to derive parameter values from.
        index : int
            Parameter index for identification.
        base_ff : openff.toolkit.ForceField
            Base force field to derive initial parameter values.  Only
            consulted when ``cached_params`` is ``None``.
        component_count : int, optional
            Explicit population count stored in the parameter id.
        cached_params : list, optional
            Pre-computed base-FF parameter objects (from
            :func:`precompute_ff_parameter_cache`).  When supplied,
            ``base_ff`` is not used and no ``label_molecules`` call is made.

        Returns
        -------
        ParameterType
            Force field parameter object for this component type.
        """
        ...


class Bond(MMComponent):
    """
    Molecular mechanics bond component.

    Represents a covalent bond between two atoms with associated force field
    parameters for bond stretching.

    Attributes
    ----------
    n_atoms : int
        Number of atoms in a bond (always 2).
    handler_class : type[BondHandler]
        SMIRNOFF bond parameter handler.
    handler_version : float
        Version 0.4 of the bond handler.
    parameter_type : type[BondHandler.BondType]
        SMIRNOFF bond parameter type.
    getter_fn : Callable[[Molecule], set[tuple[int, ...]]]
        Function to extract bond indices from molecules.

    Examples
    --------
    >>> from openff.toolkit import Molecule
    >>> mol = Molecule.from_smiles("CCO")
    >>> bond = Bond(indices=(0, 1), mol=mol, rdkit_mol=mol.to_rdkit())
    >>> bond.n_atoms
    2
    """

    n_atoms: int = 2
    handler_class: type[ParameterHandler] = BondHandler
    handler_version: float = 0.4
    parameter_type: type[ParameterType] = BondHandler.BondType
    getter_fn: Callable[[Molecule], set[tuple[int, ...]]] = cast(
        Callable[[Molecule], set[tuple[int, ...]]], get_bond_idxs
    )

    @classmethod
    def get_parameter(
        cls,
        smirks: str,
        specificity_num: int,
        components: list["MMComponent"],
        index: int,
        base_ff: ForceField,
        component_count: int | None = None,
        cached_params: list | None = None,
    ) -> BondHandler.BondType:
        """
        Generate bond parameter from component statistics.

        Creates a bond parameter by averaging force constants and equilibrium
        bond lengths from similar bonds in a base force field.

        Parameters
        ----------
        smirks : str
            SMIRKS pattern defining the bond type.
        specificity_num : int
            Specificity level for parameter identification.
        components : list[MMComponent]
            List of bond components to derive parameter values from.
        index : int
            Parameter index for identification.
        base_ff : openff.toolkit.ForceField
            Base force field to extract reference parameters.  Only used
            when ``cached_params`` is ``None``.
        component_count : int, optional
            Explicit population count stored in the parameter id.
        cached_params : list, optional
            Pre-computed base-FF parameters (avoids ``label_molecules`` call).

        Returns
        -------
        BondHandler.BondType
            Bond parameter with averaged force constant and equilibrium length.
        """
        base_ff_parameters = (
            cached_params
            if cached_params is not None
            else get_parameters_for_components(components, base_ff)
        )
        k_unit = off_unit.kilocalorie_per_mole / off_unit.angstroms**2
        mean_k = np.mean([p.k.m_as(k_unit) for p in base_ff_parameters]) * k_unit
        length_unit = off_unit.angstroms
        mean_length = (
            np.mean([p.length.m_as(length_unit) for p in base_ff_parameters])
            * length_unit
        )

        count = component_count if component_count is not None else len(components)

        parameter = BondHandler.BondType(
            smirks=smirks,
            k=mean_k,
            length=mean_length,
            id=f"specificity={specificity_num} index={index} count={count}",
        )
        return parameter

    def get_bond_length(self):
        """
        Calculate bond lengths from all molecular conformers.

        Computes the Euclidean distance between the two atoms defining this bond
        for each available conformer in the RDKit molecule. This provides geometric
        information that can be used for parameter fitting or validation.

        Returns
        -------
        list[float] or None
            List of bond lengths in Angstroms, one for each conformer.
            Returns None if no conformers are available.

        Examples
        --------
        >>> from openff.toolkit import Molecule
        >>> mol = Molecule.from_smiles("CCO")
        >>> mol.generate_conformers(n_conformers=2)
        >>> bond = Bond(indices=(0, 1), mol=mol, rdkit_mol=mol.to_rdkit())
        >>> lengths = bond.get_bond_length()
        >>> len(lengths)
        2
        >>> all(1.4 < length < 1.6 for length in lengths)  # Typical C-C bond
        True

        Notes
        -----
        The calculation uses 3D coordinates from RDKit conformers. If multiple
        conformers are available, each will yield a separate bond length value
        reflecting the molecular flexibility or different optimization states.
        """
        n_confs = self.rdkit_mol.GetNumConformers()
        if n_confs == 0:
            return None
        lengths = []
        for i in range(n_confs):
            conf = self.rdkit_mol.GetConformer(i)
            pos1 = conf.GetAtomPosition(self.indices[0])
            pos2 = conf.GetAtomPosition(self.indices[1])
            length = np.linalg.norm(
                np.array([pos1.x, pos1.y, pos1.z]) - np.array([pos2.x, pos2.y, pos2.z])
            )
            lengths.append(length)
        return lengths


class Angle(MMComponent):
    """
    Molecular mechanics angle component.

    Represents a bond angle between three atoms with associated force field
    parameters for angle bending.

    Attributes
    ----------
    n_atoms : int
        Number of atoms in an angle (always 3).
    handler_class : type[AngleHandler]
        SMIRNOFF angle parameter handler.
    handler_version : float
        Version 0.3 of the angle handler.
    parameter_type : type[AngleHandler.AngleType]
        SMIRNOFF angle parameter type.
    getter_fn : Callable[[Molecule], set[tuple[int, ...]]]
        Function to extract angle indices from molecules.

    Examples
    --------
    >>> from openff.toolkit import Molecule
    >>> mol = Molecule.from_smiles("CCO")
    >>> angle = Angle(indices=(0, 1, 2), mol=mol, rdkit_mol=mol.to_rdkit())
    >>> angle.n_atoms
    3
    """

    n_atoms: int = 3
    handler_class: type[ParameterHandler] = AngleHandler
    handler_version: float = 0.3
    parameter_type: type[ParameterType] = AngleHandler.AngleType
    getter_fn: Callable[[Molecule], set[tuple[int, ...]]] = cast(
        Callable[[Molecule], set[tuple[int, ...]]], get_angle_idxs
    )

    @classmethod
    def get_parameter(
        cls,
        smirks: str,
        specificity_num: int,
        components: list["MMComponent"],
        index: int,
        base_ff: ForceField,
        component_count: int | None = None,
        cached_params: list | None = None,
    ) -> AngleHandler.AngleType:
        """
        Generate angle parameter from component statistics.

        Creates an angle parameter by averaging force constants and equilibrium
        angles from similar angles in a base force field.

        Parameters
        ----------
        smirks : str
            SMIRKS pattern defining the angle type.
        specificity_num : int
            Specificity level for parameter identification.
        components : list[MMComponent]
            List of angle components to derive parameter values from.
        index : int
            Parameter index for identification.
        base_ff : openff.toolkit.ForceField
            Base force field to extract reference parameters.  Only used
            when ``cached_params`` is ``None``.
        component_count : int, optional
            Explicit population count stored in the parameter id.
        cached_params : list, optional
            Pre-computed base-FF parameters (avoids ``label_molecules`` call).

        Returns
        -------
        AngleHandler.AngleType
            Angle parameter with averaged force constant and equilibrium angle.
        """
        base_ff_parameters = (
            cached_params
            if cached_params is not None
            else get_parameters_for_components(components, base_ff)
        )
        k_unit = off_unit.kilocalorie_per_mole / off_unit.radians**2
        mean_k = np.mean([p.k.m_as(k_unit) for p in base_ff_parameters]) * k_unit
        angle_unit = off_unit.degrees
        mean_angle = (
            np.mean([p.angle.m_as(angle_unit) for p in base_ff_parameters]) * angle_unit
        )

        count = component_count if component_count is not None else len(components)

        parameter = AngleHandler.AngleType(
            smirks=smirks,
            k=mean_k,
            angle=mean_angle,
            id=f"specificity={specificity_num} index={index} count={count}",
        )
        return parameter

    def get_angle(self):
        """
        Calculate angle values from all molecular conformers.

        Computes the bond angle between the three atoms defining this angle
        for each available conformer in the RDKit molecule. The angle is measured
        between vectors from the central atom (second atom) to the two terminal atoms.

        Returns
        -------
        list[float] or None
            List of angle values in degrees, one for each conformer.
            Each angle is between 0 and 180 degrees. Returns None if no conformers
            are available.

        Examples
        --------
        >>> from openff.toolkit import Molecule
        >>> mol = Molecule.from_smiles("CCO")
        >>> mol.generate_conformers(n_conformers=2)
        >>> angle = Angle(indices=(0, 1, 2), mol=mol, rdkit_mol=mol.to_rdkit())
        >>> angles = angle.get_angle()
        >>> len(angles)
        2
        >>> all(100 < angle < 120 for angle in angles)  # Typical C-C-O angle
        True

        Notes
        -----
        The calculation uses 3D coordinates from RDKit conformers and applies
        the standard vector dot product formula for angle calculation:

        .. math::
            \\theta = \\arccos\\left(\\frac{\\mathbf{v_1} \\cdot \\mathbf{v_2}}{|\\mathbf{v_1}| |\\mathbf{v_2}|}\\right)

        where v1 and v2 are vectors from the central atom to the terminal atoms.
        The result is clipped to [-1, 1] to avoid numerical errors in arccos.
        """
        n_confs = self.rdkit_mol.GetNumConformers()
        if n_confs == 0:
            return None
        angles = []
        for i in range(n_confs):
            conf = self.rdkit_mol.GetConformer(i)
            pos1 = conf.GetAtomPosition(self.indices[0])
            pos2 = conf.GetAtomPosition(self.indices[1])
            pos3 = conf.GetAtomPosition(self.indices[2])
            v1 = np.array([pos1.x, pos1.y, pos1.z]) - np.array([pos2.x, pos2.y, pos2.z])
            v2 = np.array([pos3.x, pos3.y, pos3.z]) - np.array([pos2.x, pos2.y, pos2.z])
            cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            angle_rad = np.arccos(np.clip(cos_theta, -1.0, 1.0))
            angles.append(np.degrees(angle_rad))
        return angles


class ProperTorsion(MMComponent):
    """
    Molecular mechanics proper torsion component.

    Represents a proper dihedral angle between four atoms with associated
    force field parameters for torsional rotation about the central bond.

    Attributes
    ----------
    n_atoms : int
        Number of atoms in a proper torsion (always 4).
    handler_class : type[ProperTorsionHandler]
        SMIRNOFF proper torsion parameter handler.
    handler_version : float
        Version 0.4 of the proper torsion handler.
    parameter_type : type[ProperTorsionHandler.ProperTorsionType]
        SMIRNOFF proper torsion parameter type.
    getter_fn : Callable[[Molecule], set[tuple[int, ...]]]
        Function to extract proper torsion indices from molecules.

    Examples
    --------
    >>> from openff.toolkit import Molecule
    >>> mol = Molecule.from_smiles("CCCO")
    >>> torsion = ProperTorsion(indices=(0, 1, 2, 3), mol=mol, rdkit_mol=mol.to_rdkit())
    >>> torsion.n_atoms
    4
    >>> torsion.central_bond_index
    1
    """

    n_atoms: int = 4
    handler_class: type[ParameterHandler] = ProperTorsionHandler
    handler_version: float = 0.4
    parameter_type: type[ParameterType] = ProperTorsionHandler.ProperTorsionType
    getter_fn: Callable[[Molecule], set[tuple[int, ...]]] = cast(
        Callable[[Molecule], set[tuple[int, ...]]], get_proper_torsion_idxs
    )

    @classmethod
    def get_parameter(
        cls,
        smirks: str,
        specificity_num: int,
        components: list["MMComponent"],
        index: int,
        base_ff: ForceField,
        component_count: int | None = None,
        cached_params: list | None = None,
    ) -> ProperTorsionHandler.ProperTorsionType:
        """
        Generate proper torsion parameter with default values.

        Creates a proper torsion parameter with default force constants and
        phase angles. Currently uses zero force constants as placeholders.

        Parameters
        ----------
        smirks : str
            SMIRKS pattern defining the torsion type.
        specificity_num : int
            Specificity level for parameter identification.
        components : list[MMComponent]
            List of torsion components (used for counting).
        index : int
            Parameter index for identification.
        base_ff : openff.toolkit.ForceField
            Base force field (not currently used for torsions).

        Returns
        -------
        ProperTorsionHandler.ProperTorsionType
            Proper torsion parameter with default values for 4 periodicities.
        """
        # assert all(isinstance(c, ProperTorsion) for c in components), "All components must be ProperTorsion instances"
        count = component_count if component_count is not None else len(components)

        parameter = ProperTorsionHandler.ProperTorsionType(
            smirks=smirks,
            k=[0 * off_unit.kilocalorie_per_mole / off_unit.radian**2]
            * 4,  # Default K values
            phase=[0 * off_unit.degrees] * 4,  # Default phase values
            periodicity=[1, 2, 3, 4],  # Default periodicities
            idivf=[1.0] * 4,  # Default idivf values
            id=f"specificity={specificity_num} index={index} count={count}",
        )
        return parameter


class ImproperTorsion(MMComponent):
    """
    Molecular mechanics improper torsion component.

    Represents an improper dihedral angle used to maintain planarity or
    tetrahedral geometry. The third atom is always the central atom.

    Attributes
    ----------
    n_atoms : int
        Number of atoms in an improper torsion (always 4).
    handler_class : type[ImproperTorsionHandler]
        SMIRNOFF improper torsion parameter handler.
    handler_version : float
        Version 0.3 of the improper torsion handler.
    parameter_type : type[ImproperTorsionHandler.ImproperTorsionType]
        SMIRNOFF improper torsion parameter type.
    getter_fn : Callable[[Molecule], set[tuple[int, ...]]]
        Function to extract improper torsion indices from molecules.

    Notes
    -----
    Improper torsions have special symmetry considerations where the third
    atom is always treated as the central atom, affecting SMIRKS generation.

    Examples
    --------
    >>> from openff.toolkit import Molecule
    >>> mol = Molecule.from_smiles("C(C)(C)C")  # tetrahedral carbon
    >>> improper = ImproperTorsion(indices=(0, 1, 2, 3), mol=mol, rdkit_mol=mol.to_rdkit())
    >>> improper.n_atoms
    4
    """

    n_atoms: int = 4
    handler_class: type[ParameterHandler] = ImproperTorsionHandler
    handler_version: float = 0.3
    parameter_type: type[ParameterType] = ImproperTorsionHandler.ImproperTorsionType
    getter_fn: Callable[[Molecule], set[tuple[int, ...]]] = cast(
        Callable[[Molecule], set[tuple[int, ...]]], get_improper_torsion_idxs
    )

    # Need to override some methods as impropers are not symmetrical in that
    # the third atom is always the central atom.
    def _construct_smirks(self, atoms: list[str], bonds: list[str]) -> str:
        """
        Construct SMIRKS representation for improper torsions.

        Improper torsions use a special branched SMIRKS syntax where the
        third atom is always central: [a1][b1][a2]([b2][a3])[b3][a4]

        Parameters
        ----------
        atoms : list[str]
            List of atom SMIRKS patterns (length 4).
        bonds : list[str]
            List of bond SMIRKS patterns (length 3).

        Returns
        -------
        str
            Branched SMIRKS pattern for the improper torsion.

        Raises
        ------
        AssertionError
            If the number of atoms or bonds is incorrect.
        """
        # Make sure that the lengths of atoms and bonds are consistent
        assert (
            len(atoms) == self.n_atoms
        ), f"Expected {self.n_atoms} atoms, got {len(atoms)}"
        assert (
            len(bonds) == self.n_atoms - 1
        ), f"Expected {self.n_atoms - 1} bonds, got {len(bonds)}"

        return (
            f"{atoms[0]}{bonds[0]}{atoms[1]}({bonds[1]}{atoms[2]}){bonds[2]}{atoms[3]}"
        )

    def get_smirks(self, specificity_level: SpecificityLevel) -> str:
        """
        Generate SMIRKS representation for improper torsions.

        Unlike proper torsions, improper torsions are not symmetric due to
        the central atom constraint, so no reverse ordering is considered.

        Parameters
        ----------
        specificity_level : SpecificityLevel
            Defines the level of detail for atom and bond descriptions.

        Returns
        -------
        str
            Branched SMIRKS pattern with the third atom as central.
        """
        idxs = self.indices
        n = self.n_atoms

        atoms = [
            specificity_level.get_atom_smirks(
                at_idx,
                at_id,
                self.rdkit_mol,
                self.terminal_atom_indices,
            )
            for at_idx, at_id in zip(idxs, range(n))
        ]

        # Third atom is always central
        bond_atom_numbers = {(0, 1), (1, 2), (1, 3)}
        bonds = [
            specificity_level.get_bond_smirks(
                (idxs[i], idxs[j]), central_bond=(j == 2), mol=self.rdkit_mol
            )
            for i, j in bond_atom_numbers
        ]
        return self._construct_smirks(atoms, bonds)

    @classmethod
    def get_parameter(
        cls,
        smirks: str,
        specificity_num: int,
        components: list["MMComponent"],
        index: int,
        base_ff: ForceField,
        component_count: int | None = None,
        cached_params: list | None = None,
    ) -> ImproperTorsionHandler.ImproperTorsionType:
        """
        Generate improper torsion parameter with default values.

        Creates an improper torsion parameter with default force constant
        and phase angle for maintaining molecular geometry.

        Parameters
        ----------
        smirks : str
            SMIRKS pattern defining the improper torsion type.
        specificity_num : int
            Specificity level for parameter identification.
        components : list[MMComponent]
            List of improper torsion components (used for counting).
        index : int
            Parameter index for identification.
        base_ff : openff.toolkit.ForceField
            Base force field (not currently used for impropers).

        Returns
        -------
        ImproperTorsionHandler.ImproperTorsionType
            Improper torsion parameter with default values.
        """
        # assert all(isinstance(c, ImproperTorsion) for c in components), "All components must be ImproperTorsion instances"
        count = component_count if component_count is not None else len(components)

        parameter = ImproperTorsionHandler.ImproperTorsionType(
            smirks=smirks,
            k=[
                0 * off_unit.kilocalorie_per_mole / off_unit.radian**2
            ],  # Default K value
            phase=[180 * off_unit.degrees],  # Default phase value
            periodicity=[2],  # Default periodicity
            idivf=[1.0],  # Default idivf value
            id=f"specificity={specificity_num} index={index} count={count}",
        )
        return parameter


# ---------------------------------------------------------------------------
# Memory-check helpers for pre-flight fork safety
# ---------------------------------------------------------------------------


def _read_meminfo_kb(key: str) -> int:
    """Return the value (kB) for *key* from /proc/meminfo, or 0 if unavailable."""
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith(key + ":"):
                    return int(re.split(r"\s+", line.strip())[1])
    except (OSError, ValueError):
        pass
    return 0


def _read_self_rss_kb() -> int:
    """Return the resident set size (kB) of this process from /proc/self/status."""
    try:
        with open("/proc/self/status") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return int(re.split(r"\s+", line.strip())[1])
    except (OSError, ValueError):
        pass
    return 0


# Module-level globals/helpers for cache precomputation workers.
_cache_worker_ff = None


def _cache_worker_init(ff: ForceField):
    """Initialize shared force field object for cache labeling workers."""
    global _cache_worker_ff
    _cache_worker_ff = ff


def _cache_label_one(args: tuple[str, Molecule]) -> tuple[str, dict]:
    """Label one molecule in a worker and return (mapped_smiles, assignments)."""
    smiles, mol = args
    if _cache_worker_ff is None:
        raise RuntimeError("Cache worker force field was not initialized")
    assigned = _cache_worker_ff.label_molecules(mol.to_topology())[0]
    return smiles, assigned


def precompute_ff_parameter_cache(
    component_types: dict[int, dict[str, list["MMComponent"]]],
    forcefield: ForceField,
    component_class: type["MMComponent"],
    n_workers: int | None = None,
    max_samples: int = 10,
) -> dict[tuple[str, tuple], object]:
    """
    Pre-compute base FF parameter assignments for all sampled components.

    Rather than calling ``forcefield.label_molecules`` once per SMIRKS type
    inside a multiprocessing pool (which serialises full ``Molecule`` objects
    over IPC for every task), this function calls it once per *unique* molecule
    across all component types, then caches the result.  The cache is keyed by
    ``(mapped_smiles, indices)`` so individual ``get_parameter`` calls can do a
    plain dict lookup instead of an expensive SMIRKS-matching round-trip.

    Parameters
    ----------
    component_types : dict[int, dict[str, list[MMComponent]]]
        Output of ``get_components_by_type``: {specificity_level: {smirks: [comps]}}.
    forcefield : ForceField
        Base force field used to derive initial parameter values (read-only).
    component_class : type[MMComponent]
        The component class (Bond, Angle, …) being processed.
    n_workers : int, optional
        Number of parallel workers for the label_molecules step. Uses a
        ``spawn`` multiprocessing pool with module-level worker initialization
        for HPC compatibility. Falls back to a thread pool if process workers
        are unavailable.
    max_samples : int, default 10
        Maximum number of components sampled per SMIRKS type (mirrors the
        sampling in ``get_parameters_for_components``).

    Returns
    -------
    dict[tuple[str, tuple], object]
        Mapping ``{(mapped_smiles, indices_tuple): ParameterType}`` for every
        (molecule, component) pair encountered in the sampled components.

    Notes
    -----
    The cache will include entries for every unique (mapped_smiles, component
    indices) pair seen across all sampled components.  Missing entries (e.g.
    because the base FF does not cover the component) are silently skipped so
    callers should handle absent keys gracefully.
    """

    tag_name = component_class.handler_class._TAGNAME

    # --- 1. Collect unique (mapped_smiles, mol) pairs from sampled components ---
    unique_mols: dict[str, "Molecule"] = {}
    for level_dict in component_types.values():
        for comps in level_dict.values():
            for c in comps[:max_samples]:
                if c.mapped_smiles not in unique_mols:
                    unique_mols[c.mapped_smiles] = c.mol

    logger.info(
        f"Pre-computing base FF parameter cache for {len(unique_mols):,} unique molecules "
        f"({component_class.__name__})..."
    )

    # --- 2. label_molecules once per unique molecule, in parallel ---
    if n_workers is None:
        n_workers = os.cpu_count() or 1

    smiles_list = list(unique_mols.keys())
    mol_list = [unique_mols[s] for s in smiles_list]
    cache: dict[tuple[str, tuple], object] = {}

    # Use spawn context so workers start with a clean Python process (~100 MB)
    # rather than inheriting the parent's full heap via fork copy-on-write.
    # At the Angle stage the parent may be >60 GB; fork would give each of
    # n_workers workers a CoW copy of that entire heap, and Python's reference
    # counting immediately dirties those pages throughout, causing peak usage
    # of n_workers × parent_RSS.  With spawn, workers receive only what they
    # actually need: the ForceField (pickled once per worker via the
    # initializer) and individual (smiles, Molecule) pairs per task.
    ctx = mp_get_context("spawn")
    with ctx.Pool(
        processes=n_workers,
        initializer=_cache_worker_init,
        initargs=(forcefield,),
    ) as pool:
        results_iter = pool.imap_unordered(
            _cache_label_one,
            zip(smiles_list, mol_list),
            chunksize=max(1, len(smiles_list) // (n_workers * 8) or 1),
        )
        # Use a per-next timeout so that if a worker is OOM-killed (and
        # the pool silently respawns it without resubmitting the lost task)
        # we detect the stall within *timeout* seconds and raise a clear
        # error instead of hanging indefinitely.
        _WORKER_TIMEOUT = 300  # seconds between liveness checks
        next_with_timeout = getattr(results_iter, "next", None)
        if callable(next_with_timeout):
            while True:
                try:
                    smiles, assigned = next_with_timeout(timeout=_WORKER_TIMEOUT)
                except StopIteration:
                    break
                except MPTimeoutError:
                    dead = [p for p in pool._pool if not p.is_alive()]  # type: ignore[attr-defined]
                    if dead:
                        pool.terminate()
                        avail_gb = _read_meminfo_kb("MemAvailable") / 1024**2
                        rss_gb = _read_self_rss_kb() / 1024**2
                        raise RuntimeError(
                            f"{len(dead)}/{n_workers} cache-build worker process(es) died "
                            f"unexpectedly during label_molecules (OOM is likely). "
                            f"Process RSS at failure: {rss_gb:.1f} GB; "
                            f"available memory: {avail_gb:.1f} GB. "
                            f"To fix: (1) increase --mem-per-cpu in the SLURM script, or "
                            f"(2) reduce -n (number of workers)."
                        )
                    # Spurious timeout (very slow molecule or heavy node load) - keep waiting.
                    continue
                if tag_name and tag_name in assigned:
                    for indices, param in assigned[tag_name].items():
                        cache[(smiles, indices)] = param
        else:
            # Some runtimes return a plain generator here; iterate without timeout support.
            for smiles, assigned in results_iter:
                if tag_name and tag_name in assigned:
                    for indices, param in assigned[tag_name].items():
                        cache[(smiles, indices)] = param

    logger.info(f"Parameter cache built: {len(cache):,} entries.")
    return cache


def get_parameters_for_components_cached(
    components: list["MMComponent"],
    cache: dict[tuple[str, tuple], object],
    max_samples: int = 10,
) -> ParameterList:
    """
    Retrieve base FF parameters for components using a pre-computed cache.

    Drop-in replacement for ``get_parameters_for_components`` that avoids
    calling ``forcefield.label_molecules`` at runtime.

    Parameters
    ----------
    components : list[MMComponent]
        Components to look up.
    cache : dict[tuple[str, tuple], object]
        Cache produced by :func:`precompute_ff_parameter_cache`.
    max_samples : int, default 10
        Maximum number of components to sample.

    Returns
    -------
    ParameterList
        Parameters found in the cache for the sampled components.
    """
    subsampled = (
        components
        if len(components) <= max_samples
        else np.random.choice(components, max_samples, replace=False)
    )
    params = []
    for c in subsampled:
        key = (c.mapped_smiles, c.indices)
        param = cache.get(key)
        if param is not None:
            params.append(param)
    return cast(ParameterList, params)


def get_parameters_for_components(
    components: list[MMComponent], forcefield: ForceField, max_samples: int = 10
) -> ParameterList:
    """
    Extract force field parameters for a list of MM components.

    Retrieves parameters from a force field by labeling molecules containing
    the components and extracting the relevant parameter values.

    Parameters
    ----------
    components : list[MMComponent]
        List of molecular mechanics components to get parameters for.
    forcefield : openff.toolkit.ForceField
        Force field to extract parameters from.
    max_samples : int, default 10
        Maximum number of components to sample for parameter extraction.
        Used to limit computation time for large component lists.

    Returns
    -------
    ParameterList
        List of parameter objects extracted from the force field.

    Notes
    -----
    For efficiency, if the component list exceeds max_samples, a random
    subset is chosen for parameter extraction. This provides a representative
    sample while keeping computation time reasonable.

    Examples
    --------
    >>> bonds = [Bond(...), Bond(...)]  # List of bond components
    >>> ff = ForceField("openff-2.0.0.offxml")
    >>> params = get_parameters_for_components(bonds, ff, max_samples=5)
    """
    parameters = []
    subsampled_components = (
        components
        if len(components) <= max_samples
        else np.random.choice(components, max_samples, replace=False)
    )
    for c in subsampled_components:
        assigned_parameters = forcefield.label_molecules(c.mol.to_topology())[
            0
        ]  # As only one molecule
        tag_name = c.handler_class._TAGNAME
        if tag_name is not None:
            parameters.append(assigned_parameters[tag_name][c.indices])
    return cast(ParameterList, parameters)
