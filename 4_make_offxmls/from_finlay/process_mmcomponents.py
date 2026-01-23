"""
Molecular Mechanics Component Processing and Analysis.

This module provides utilities for extracting, processing, and analyzing molecular
mechanics components (bonds, angles, torsions) from molecular datasets. It handles
the conversion from SMILES strings to component objects, parallel processing for
large datasets, and hierarchical organization by specificity levels.

Key Functions
-------------
Component Extraction:
    - get_mm_components_from_huggingface: Extract components from a huggingface dataset row
    - get_all_mm_components: Extract components from HuggingFace dataset with filtering

Component Organization:
    - get_all_mm_components_by_type: Group components by SMIRKS patterns
    - get_all_mm_components_by_type_parallel: Parallel version for large datasets
    - get_mm_components_by_specificity_by_type: Hierarchical organization by specificity

Analysis and Statistics:
    - get_mm_component_type_num: Count occurrences of each component type
    - flatten_mm_component_types: Flatten hierarchical structure with counts

Examples
--------
Extracting bonds from a HuggingFace dataset row:

>>> from openff.toolkit import Molecule
>>> from molecular_classes import Bond
>>> row = {"smiles": "[C:1][C:2][O:3]"}
>>> components = get_mm_components_from_huggingface(row, Bond)
>>> print(f"Found {len(components)} bonds")
Found 2 bonds

Processing multiple SMILES with parallel execution:

>>> from datasets import Dataset
>>> from molecular_classes import Bond
>>> dataset = Dataset.from_dict({
...     "mapped_smiles": ["CCO", "CCC", "CC(C)C"]
... })
>>> all_bonds = get_all_mm_components(dataset, Bond)
>>> bond_types = get_all_mm_components_by_type_parallel(all_bonds, specificity_level)

Hierarchical organization by specificity:

>>> specificity_levels = {0: low_spec, 1: medium_spec, 2: high_spec}
>>> hierarchical = get_mm_components_by_specificity_by_type(
...     components, specificity_levels, cutoff_population=10
... )
>>> # hierarchical[0] = low specificity types
>>> # hierarchical[1] = medium specificity types (for rare patterns)
"""

import os
from functools import partial
from typing import Iterable
from collections import Counter, defaultdict
from multiprocessing import get_context

from loguru import logger
import numpy as np
from datasets import Dataset
from rdkit import Chem
from rdkit.Geometry import Point3D

from openff.toolkit import Molecule

from .molecular_classes import MMComponent, SpecificityLevel


def get_mm_components_from_huggingface(
    ds_row,
    component_type: type[MMComponent],
    load_coords=True,
) -> list[MMComponent]:
    """
    Extract molecular mechanics components from HuggingFace dataset row.

    Parameters
    ----------
    ds_row : dict
        HuggingFace dataset row containing 'smiles' field and optional 'coords'.
        The 'smiles' field should contain mapped SMILES strings with atom indexing.
        The 'coords' field (if present) should be a flat list of xyz coordinates:
        [x1,y1,z1, x2,y2,z2, ..., xN,yN,zN] for each conformer sequentially.
    component_type : type[MMComponent]
        Component class (Bond, Angle, ProperTorsion, ImproperTorsion).
    load_coords : bool, optional
        If true and input has coordinates, the conformer is created in the RDKit molecule

    Returns
    -------
    list[MMComponent]
        Component instances extracted from the molecule.

    Examples
    --------
    >>> from molecular_classes import Bond
    >>> row = {"smiles": "[C:1][C:2][O:3]"}
    >>> bonds = get_mm_components_from_huggingface(row, Bond)
    >>> len(bonds)
    2

    >>> # With coordinate data for multiple conformers
    >>> import numpy as np
    >>> row_with_coords = {
    ...     "smiles": "[C:1][C:2][O:3]",  # 3 atoms
    ...     "coords": [
    ...         0.0, 0.0, 0.0, 1.5, 0.0, 0.0, 2.0, 1.0, 0.0,  # Conformer 1 flattened xyz-coord
    ...         0.1, 0.1, 0.1, 1.4, 0.1, 0.1, 2.1, 0.9, 0.1   # Conformer 2 flattened xyz-coord
    ...     ]
    ... }
    >>> bonds = get_mm_components_from_huggingface(row_with_coords, Bond)
    >>> len(bonds)
    2
    >>> bonds[0].rdkit_mol.GetNumConformers()
    2

    >>> # Extract angles from a molecule
    >>> from molecular_classes import Angle
    >>> angles = get_mm_components_from_huggingface(row, Angle)
    >>> len(angles)
    1
    """
    mol = Molecule.from_mapped_smiles(ds_row["smiles"], allow_undefined_stereo=True)
    if mol is None:
        raise ValueError(f"Invalid mapped SMILES: {ds_row['smiles']}")

    rdkit_mol = mol.to_rdkit()
    coords = ds_row.get("coords")
    if coords is not None and load_coords:
        arr = np.array(coords, dtype=float).reshape(-1, 3)
        num_atoms = rdkit_mol.GetNumAtoms()
        num_conformers = len(arr) // (num_atoms)

        if len(arr) % num_atoms != 0:
            raise ValueError(
                f"Coordinate array length ({len(arr)}) is not divisible by number of atoms ({num_atoms})"
            )

        # Add each conformer to the molecule
        for conf_idx in range(num_conformers):
            conf = Chem.Conformer(num_atoms)
            start_idx = conf_idx * num_atoms
            end_idx = start_idx + num_atoms
            conf_coords = arr[start_idx:end_idx]

            for atom_idx, pos in enumerate(conf_coords):
                conf.SetAtomPosition(atom_idx, Point3D(pos[0], pos[1], pos[2]))
            rdkit_mol.AddConformer(conf, assignId=True)

    components = [
        component_type(
            indices=idxs,
            mol=mol,
            rdkit_mol=rdkit_mol,
        )
        for idxs in component_type.getter_fn(mol)
    ]
    if not components:
        raise ValueError(
            f"The following SMILES string produces no components: {ds_row['smiles']}"
        )

    return components


def is_unwanted_smirks(component: MMComponent, unwanted_smirks: list[str]) -> bool:
    """
    Check if component matches any unwanted SMIRKS pattern.

    Parameters
    ----------
    component : MMComponent
        Component to check.
    unwanted_smirks : list[str]
        SMIRKS patterns to match against.

    Returns
    -------
    bool
        True if component matches any pattern, False otherwise.

    Examples
    --------
    >>> unwanted = ["[#6:1]-[#8:2]"]
    >>> is_unwanted_smirks(bond, unwanted)
    True
    """
    return any(component.matches_smirks(smirks) for smirks in unwanted_smirks)


def _process_dataset_row(row, component_type, unwanted_smirks):
    """
    Process a single dataset row to extract and filter components.

    Helper function for parallel processing in get_all_mm_components.

    Parameters
    ----------
    row : dict
        Dataset row with 'smiles' and optional 'coords' fields.
    component_type : type[MMComponent]
        Component class to extract.
    unwanted_smirks : list[str] | None
        SMIRKS patterns to exclude.

    Returns
    -------
    tuple[list[MMComponent], int, int]
        Tuple of (filtered_components, n_total, n_filtered) where:
        - filtered_components: Components that passed filtering
        - n_total: Total components extracted before filtering
        - n_filtered: Number of components after filtering
    """
    components = get_mm_components_from_huggingface(row, component_type)
    if unwanted_smirks is None or len(unwanted_smirks) == 0:
        return components, len(components), len(components)

    filtered_components = [
        x
        for x in components
        if not is_unwanted_smirks(x, unwanted_smirks=unwanted_smirks)
    ]
    return filtered_components, len(components), len(filtered_components)


def get_all_mm_components(
    dataset: Dataset,
    component_type: type[MMComponent],
    unwanted_smirks: list[str] | None = None,
    n_workers: int | None = None,
) -> list[MMComponent]:
    """
    Extract components from HuggingFace dataset with filtering using parallel processing.

    Parameters
    ----------
    dataset : Dataset
        HuggingFace dataset with 'smiles' and optional 'coords' fields.
    component_type : type[MMComponent]
        Component class to extract (Bond, Angle, ProperTorsion, ImproperTorsion).
    unwanted_smirks : list[str], optional
        SMIRKS patterns to exclude from results.
    n_workers : int, optional
        Number of worker processes. If None, uses all available CPU cores.

    Returns
    -------
    list[MMComponent]
        All component instances from valid molecules, filtered if specified.

    Examples
    --------
    >>> from datasets import Dataset
    >>> from molecular_classes import Bond
    >>> dataset = Dataset.from_dict({
    ...     "smiles": ["[C:1][C:2][O:3]", "[C:1][C:2][C:3]"]
    ... })
    >>> bonds = get_all_mm_components(dataset, Bond)
    >>> print(f"Total bonds: {len(bonds)}")
    Total bonds: 4

    >>> # Filter out C-O bonds
    >>> unwanted = ["[#6:1]-[#8:2]"]
    >>> filtered_bonds = get_all_mm_components(dataset, Bond, unwanted_smirks=unwanted)
    >>> print(f"Filtered bonds: {len(filtered_bonds)}")
    Filtered bonds: 3
    """
    if n_workers is None:
        n_workers = os.cpu_count()
    n_workers = n_workers or 1
    logger.info(f"Unwanted SMIRKS to filter: {unwanted_smirks}")
    logger.info(f"Using {n_workers} workers for parallel processing")

    all_components = []
    n_components, n_filtered = 0, 0
    lx = len(dataset)

    process_row = partial(
        _process_dataset_row,
        component_type=component_type,
        unwanted_smirks=unwanted_smirks,
    )

    logger.info(f"Processing {lx} molecules from HuggingFace Dataset")

    # Use 'spawn' context for better compatibility on HPC systems
    # 'spawn' avoids fork-safety issues with scientific libraries
    ctx = get_context("spawn")

    with ctx.Pool(processes=n_workers) as pool:
        # Use imap_unordered for lazy iteration and better memory efficiency
        # Results are yielded as they complete (unordered for better performance)
        results_iter = pool.imap_unordered(process_row, dataset, chunksize=10)

        # Track progress at 5% intervals
        next_log_threshold = 0.05
        processed = 0

        for filtered_components, n_total, n_filt in results_iter:
            all_components.extend(filtered_components)
            n_components += n_total
            n_filtered += n_filt
            processed += 1

            # Log at 5% intervals
            progress = processed / lx
            if progress >= next_log_threshold:
                logger.info(
                    f"Progress: {processed}/{lx} molecules ({progress*100:.1f}%)"
                )
                next_log_threshold += 0.05

    logger.info(f"Filtered out {n_components - n_filtered} unwanted components.")

    return all_components


def get_all_mm_components_by_type(
    mm_components: Iterable[MMComponent], specificity_level: SpecificityLevel
) -> dict[str, list[MMComponent]]:
    """
    Group components by SMIRKS patterns at given specificity level.

    Parameters
    ----------
    mm_components : Iterable[MMComponent]
        Components to group.
    specificity_level : SpecificityLevel
        Determines SMIRKS pattern specificity.

    Returns
    -------
    dict[str, list[MMComponent]]
        SMIRKS patterns mapped to matching components.

    Examples
    --------
    >>> from molecular_classes import SpecificityLevel
    >>> specificity = SpecificityLevel(name="Standard", ...)
    >>> grouped = get_all_mm_components_by_type(bonds, specificity)
    >>> for smirks, components in grouped.items():
    ...     print(f"{smirks}: {len(components)} components")
    [#6:1]-[#6:2]: 150 components
    [#6:1]-[#8:2]: 75 components
    """
    all_component_types = defaultdict(list)
    mm_components = list(mm_components)
    total = len(mm_components)

    logger.info(f"Processing {total} components")
    next_log_threshold = 0.05

    for idx, component in enumerate(mm_components, start=1):
        smirks = component.get_smirks(specificity_level)
        all_component_types[smirks].append(component)

        # Log at 5% intervals
        progress = idx / total
        if progress >= next_log_threshold:
            logger.info(f"Progress: {idx}/{total} components ({progress*100:.1f}%)")
            next_log_threshold += 0.05

    return all_component_types


def _get_component_smirks(component, specificity_level):
    """
    Extract SMIRKS pattern for a single component.

    Helper function for parallel processing in get_all_mm_components_by_type_parallel.

    Parameters
    ----------
    component : MMComponent
        Component to extract SMIRKS from.
    specificity_level : SpecificityLevel
        Determines SMIRKS pattern specificity.

    Returns
    -------
    tuple[str, MMComponent]
        Tuple of (smirks_pattern, component).
    """
    return component.get_smirks(specificity_level), component


def get_all_mm_components_by_type_parallel(
    mm_components: Iterable[MMComponent],
    specificity_level: SpecificityLevel,
    n_workers: int | None = None,
) -> dict[str, list[MMComponent]]:
    """
    Group components by SMIRKS patterns using parallel processing.

    Parameters
    ----------
    mm_components : Iterable[MMComponent]
        Components to group.
    specificity_level : SpecificityLevel
        Determines SMIRKS pattern specificity.
    n_workers : int, optional
        Number of worker processes. If None, uses all available CPU cores.
        If os.cpu_count() returns None, defaults to 1 worker.

    Returns
    -------
    dict[str, list[MMComponent]]
        SMIRKS patterns mapped to matching components.

    Examples
    --------
    >>> # Process 10,000 torsions using all CPU cores
    >>> grouped = get_all_mm_components_by_type_parallel(
    ...     torsions, specificity_level
    ... )
    >>> print(f"Found {len(grouped)} unique SMIRKS patterns")
    Found 150 unique SMIRKS patterns

    >>> # Use specific number of workers
    >>> grouped = get_all_mm_components_by_type_parallel(
    ...     torsions, specificity_level, n_workers=4
    ... )
    """
    mm_components = list(mm_components)
    total = len(mm_components)
    if n_workers is None:
        n_workers = os.cpu_count()
    n_workers = n_workers or 1

    logger.info(f"Processing {total} components with {n_workers} workers")

    # Use partial to avoid lambda pickling issues
    get_smirks_fn = partial(_get_component_smirks, specificity_level=specificity_level)

    all_component_types = defaultdict(list)
    next_log_threshold = 0.05

    # Use 'spawn' context for better HPC compatibility
    ctx = get_context("spawn")

    with ctx.Pool(processes=n_workers) as pool:
        # imap_unordered for lazy iteration and better performance
        processed = 0
        for smirks, component in pool.imap_unordered(
            get_smirks_fn, mm_components, chunksize=100
        ):
            all_component_types[smirks].append(component)
            processed += 1

            # Log at 5% intervals
            progress = processed / total
            if progress >= next_log_threshold:
                logger.info(
                    f"Progress: {processed}/{total} components ({progress*100:.1f}%)"
                )
                next_log_threshold += 0.05

    return all_component_types


def _filter_too_specific(smirks_and_components, cutoff_population):
    """
    Filter component types that fall below the population cutoff.

    Helper function for parallel processing in get_mm_components_by_specificity_by_type.

    Parameters
    ----------
    smirks_and_components : tuple[str, list[MMComponent]]
        Tuple of (smirks_pattern, component_list).
    cutoff_population : int
        Minimum number of components required.

    Returns
    -------
    tuple[str, list[MMComponent]] | None
        Returns the input tuple if below cutoff, None otherwise.
    """
    smirks, component_list = smirks_and_components
    if len(component_list) < cutoff_population:
        return smirks, component_list
    return None


def get_mm_components_by_specificity_by_type(
    mm_components: Iterable[MMComponent],
    specificity_levels: dict[int, SpecificityLevel],
    cutoff_population: int = 10,
    n_workers: int | None = None,
) -> dict[int, dict[str, list[MMComponent]]]:
    """
    Organize components hierarchically by specificity level.

    Creates a hierarchical organization where components are first grouped at
    the highest specificity level, then rare patterns (below population cutoff)
    are moved to lower specificity levels for broader coverage.

    Parameters
    ----------
    mm_components : Iterable[MMComponent]
        Components to organize.
    specificity_levels : dict[int, SpecificityLevel]
        Specificity level mapping (higher numbers = more specific).
    cutoff_population : int, default=10
        Minimum components required to stay at current specificity level.
    n_workers : int, default=None
        Number of worker processes. If None, uses all available CPU cores.
        If os.cpu_count() returns None, defaults to 1 worker.

    Returns
    -------
    dict[int, dict[str, list[MMComponent]]]
        Hierarchical structure organized as:
        {
            specificity_level: {
                "smirks_pattern": [component1, component2, ...],
                ...
            },
            ...
        }

        Higher specificity levels contain common patterns with detailed SMIRKS.
        Lower specificity levels contain rare patterns with broader SMIRKS.

    Examples
    --------
    >>> specificity_levels = {
    ...     0: low_specificity,    # Terminal wildcards
    ...     1: medium_specificity, # Terminal H/non-H
    ...     2: high_specificity    # Full atomic detail
    ... }
    >>> hierarchical = get_mm_components_by_specificity_by_type(
    ...     torsions, specificity_levels, cutoff_population=10
    ... )
    >>> # hierarchical[2] = high specificity patterns
    >>> # hierarchical[1] = medium specificity patterns
    >>> # hierarchical[0] = low specificity patterns
    >>> len(hierarchical[2])  # High specificity patterns
    25
    >>> list(hierarchical[2].keys())[0]  # Example high-specificity SMIRKS
    '[#6X4:1]-[#6X4:2]-[#8X2:3]-[#1:4]'
    """
    components_by_specificity = {}
    specificity_order = sorted(specificity_levels.keys(), reverse=True)
    components_to_process = list(mm_components)

    for i, specificity_num in enumerate(specificity_order):
        specificity_level = specificity_levels[specificity_num]
        logger.info(
            f"Getting Components by Type with Specificity {specificity_num}: {specificity_level.name}"
        )
        components_by_type = get_all_mm_components_by_type_parallel(
            components_to_process,
            specificity_level,
            n_workers=n_workers,
        )
        components_by_specificity[specificity_num] = components_by_type

        logger.info("Finding component types that are too specific...")
        # Prepare for next (less specific) level
        if i < len(specificity_order) - 1:
            # Find component types below cutoff in parallel
            # Use partial to avoid lambda pickling issues
            filter_fn = partial(
                _filter_too_specific, cutoff_population=cutoff_population
            )

            # Use 'spawn' context for better HPC compatibility
            ctx = get_context("spawn")

            with ctx.Pool(processes=n_workers) as pool:
                results = list(
                    pool.imap_unordered(
                        filter_fn, components_by_type.items(), chunksize=10
                    )
                )

            # Separate too specific components and remove from current level
            too_specific = []
            for result in results:
                if result is not None:
                    smirks, component_list = result
                    too_specific.extend(component_list)
                    del components_by_specificity[specificity_num][smirks]

            components_to_process = too_specific

    return components_by_specificity


def get_mm_component_type_num(
    components_by_type: dict[str, list[MMComponent]],
) -> dict[str, int]:
    """
    Count component occurrences by SMIRKS pattern.

    Parameters
    ----------
    components_by_type : dict[str, list[MMComponent]]
        SMIRKS patterns mapped to component lists.

    Returns
    -------
    dict[str, int]
        SMIRKS patterns mapped to occurrence counts.

    Examples
    --------
    >>> components = {"[#6:1]-[#6:2]": [bond1, bond2, bond3]}
    >>> get_mm_component_type_num(components)
    {"[#6:1]-[#6:2]": 3}
    """
    return {
        smirks: len(components) for smirks, components in components_by_type.items()
    }


def flatten_mm_component_types(
    components_by_specificity: dict[int, dict[str, list[MMComponent]]],
) -> dict[str, int]:
    """
    Flatten hierarchical component organization to total counts.

    Parameters
    ----------
    components_by_specificity : dict[int, dict[str, list[MMComponent]]]
        Hierarchical structure from get_mm_components_by_specificity_by_type.

    Returns
    -------
    dict[str, int]
        SMIRKS patterns mapped to total occurrence counts across all levels.

    Examples
    --------
    >>> hierarchical = {
    ...     0: {"[*:1]-[*:2]": [comp1, comp2]},
    ...     1: {"[#6:1]-[#6:2]": [comp3, comp4]}
    ... }
    >>> flatten_mm_component_types(hierarchical)
    {"[*:1]-[*:2]": 2, "[#6:1]-[#6:2]": 2}
    """
    all_component_type_counts: Counter = Counter()
    for components_by_type in components_by_specificity.values():
        all_component_type_counts.update(get_mm_component_type_num(components_by_type))
    return dict(all_component_type_counts)
