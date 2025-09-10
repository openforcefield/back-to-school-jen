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
import math
from typing import Iterable
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

from loguru import logger
import numpy as np
from tqdm import tqdm
from datasets import Dataset
from rdkit import Chem
from rdkit.Geometry import Point3D

from openff.toolkit import Molecule

from .molecular_classes import MMComponent, SpecificityLevel


def get_mm_components_from_huggingface(
    ds_row,
    component_type: type[MMComponent],
) -> list[MMComponent]:
    """
    Extract molecular mechanics components from HuggingFace dataset row.

    Parameters
    ----------
    ds_row : dict
        HuggingFace dataset row containing 'smiles' field and optional 'coords'.
    component_type : type[MMComponent]
        Component class (Bond, Angle, ProperTorsion, ImproperTorsion).

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
    ...     "smiles": "[C:1][C:2][O:3]",
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
        raise ValueError(f"Invalid mapped SMILES: {ds_row["smiles"]}")

    rdkit_mol = mol.to_rdkit()
    coords = ds_row.get("coords")
    if coords is not None:
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

    component_idxs = component_type.getter_fn(mol)
    return [
        component_type(
            indices=idxs,
            mol=mol,
            rdkit_mol=rdkit_mol,
        )
        for idxs in component_idxs
    ]


def get_all_mm_components(
    dataset: Dataset,
    component_type: type[MMComponent],
    unwanted_smirks: list[str] | None = None,
) -> list[MMComponent]:
    """
    Extract components from HuggingFace dataset with filtering.

    Parameters
    ----------
    dataset : Dataset
        HuggingFace dataset with 'smiles' and optional 'coords' fields.
    component_type : type[MMComponent]
        Component class to extract (Bond, Angle, ProperTorsion, ImproperTorsion).
    unwanted_smirks : list[str], optional
        SMIRKS patterns to exclude from results.

    Returns
    -------
    list[MMComponent]
        All component instances from valid molecules, filtered if specified.

    Examples
    --------
    >>> from datasets import Dataset
    >>> from molecular_classes import Bond
    >>> dataset = Dataset.from_dict({
    ...     "mapped_smiles": ["[C:1][C:2][O:3]", "[C:1][C:2][C:3]"]
    ... })
    >>> bonds = get_all_mm_components(dataset, Bond)
    >>> print(f"Total bonds: {len(bonds)}")
    Total bonds: 4

    >>> # Filter out C-O bonds
    >>> unwanted = ["[#6:1]-[#8:2]"]
    >>> filtered_bonds = get_all_mm_components(dataset, Bond, unwanted_smirks=unwanted)
    >>> print(f"Filtered bonds: {len(filtered_bonds)}")
    Filtered bonds: 3

    >>> # Use custom column name
    >>> dataset = Dataset.from_dict({
    ...     "smiles": ["[C:1][C:2][O:3]", "[C:1][C:2][C:3]"]
    ... })
    >>> bonds = get_all_mm_components(dataset, Bond, smiles_column="smiles")
    """
    all_components = []
    for row in tqdm(dataset, desc="Processing HuggingFace Dataset"):
        all_components.extend(get_mm_components_from_huggingface(row, component_type))

    if unwanted_smirks:
        all_components_filtered = []
        logger.info(f"Filtering out unwanted SMIRKS: {unwanted_smirks}")
        for component in tqdm(all_components, desc="Filtering unwanted components"):
            matched = False
            for smirks in unwanted_smirks:
                if component.matches_smirks(smirks):
                    matched = True
                    break
            if not matched:
                all_components_filtered.append(component)
        logger.info(
            f"Filtered out {len(all_components) - len(all_components_filtered)} unwanted torsions."
        )
        all_components = all_components_filtered

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
    for component in tqdm(mm_components, desc="Processing Components"):
        smirks = component.get_smirks(specificity_level)
        all_component_types[smirks].append(component)

    return all_component_types


def process_mm_component_chunk(component_chunk, specificity_level):
    """
    Process a chunk of components for parallel execution.

    Worker function for parallel processing that groups a subset of components
    by their SMIRKS patterns at the specified specificity level.

    Parameters
    ----------
    component_chunk : list[MMComponent]
        Subset of components to process in this worker.
    specificity_level : SpecificityLevel
        Defines the specificity level for SMIRKS generation.

    Returns
    -------
    dict[str, list[MMComponent]]
        Dictionary mapping SMIRKS patterns to component lists for this chunk.

    Notes
    -----
    This function is designed for use with ProcessPoolExecutor and should
    not be called directly. Use get_all_mm_components_by_type_parallel instead.
    """
    chunk_dict = defaultdict(list)
    for component in component_chunk:
        smirks = component.get_smirks(specificity_level)
        chunk_dict[smirks].append(component)
    return chunk_dict


def merge_dicts(dicts):
    """
    Merge multiple dictionaries with list values.

    Combines dictionaries from parallel processing workers by extending
    lists for matching keys.

    Parameters
    ----------
    dicts : list[dict[str, list]]
        List of dictionaries to merge, each mapping strings to lists.

    Returns
    -------
    dict[str, list]
        Merged dictionary with all lists combined for each key.

    Examples
    --------
    >>> dict1 = {"A": [1, 2], "B": [3]}
    >>> dict2 = {"A": [4], "C": [5, 6]}
    >>> merged = merge_dicts([dict1, dict2])
    >>> print(merged)
    {"A": [1, 2, 4], "B": [3], "C": [5, 6]}
    """
    merged = defaultdict(list)
    for d in dicts:
        for k, v in d.items():
            merged[k].extend(v)
    return merged


def get_all_mm_components_by_type_parallel(
    mm_components: Iterable[MMComponent],
    specificity_level: SpecificityLevel,
    n_workers=None,
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
        Number of worker processes. Default uses all CPU cores.

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
    if n_workers is None:
        n_workers = os.cpu_count()
    chunk_size = math.ceil(len(mm_components) / n_workers)
    component_chunks = [
        mm_components[i : i + chunk_size]
        for i in range(0, len(mm_components), chunk_size)
    ]

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = [
            executor.submit(process_mm_component_chunk, chunk, specificity_level)
            for chunk in component_chunks
        ]
        results = []
        for f in as_completed(futures):
            results.append(f.result())

    return merge_dicts(results)


def get_mm_components_by_specificity_by_type(
    mm_components: Iterable[MMComponent],
    specificity_levels: dict[int, SpecificityLevel],
    cutoff_population: int = 10,
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
    cutoff_population : int, default 10
        Minimum components required to stay at current specificity level.

    Returns
    -------
    dict[int, dict[str, list[MMComponent]]]
        Nested structure: {level: {smirks: [components]}}.

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
    >>> len(hierarchical[2])  # High specificity patterns
    25
    """
    components_by_specificity = {}
    specificity_order = sorted(specificity_levels.keys(), reverse=True)
    components_to_process = list(mm_components)

    for i, specificity_num in enumerate(specificity_order):
        specificity_level = specificity_levels[specificity_num]
        components_by_type = get_all_mm_components_by_type_parallel(
            components_to_process, specificity_level
        )
        components_by_specificity[specificity_num] = components_by_type

        # Prepare for next (less specific) level
        if i < len(specificity_order) - 1:
            # Find component types below cutoff
            too_specific = [
                component
                for component_list in components_by_type.values()
                if len(component_list) < cutoff_population
                for component in component_list
            ]
            # Remove these from current level
            for smirks in [
                s for s, cs in components_by_type.items() if len(cs) < cutoff_population
            ]:
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
