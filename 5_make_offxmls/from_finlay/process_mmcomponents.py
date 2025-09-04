"""
Molecular Mechanics Component Processing and Analysis.

This module provides utilities for extracting, processing, and analyzing molecular
mechanics components (bonds, angles, torsions) from molecular datasets. It handles
the conversion from SMILES strings to component objects, parallel processing for
large datasets, and hierarchical organization by specificity levels.

Key Functions
-------------
Component Extraction:
    - get_mm_components_from_smiles: Extract components from a single SMILES string
    - get_all_mm_components: Extract components from multiple SMILES with filtering

Component Organization:
    - get_all_mm_components_by_type: Group components by SMIRKS patterns
    - get_all_mm_components_by_type_parallel: Parallel version for large datasets
    - get_mm_components_by_specificity_by_type: Hierarchical organization by specificity

Analysis and Statistics:
    - get_mm_component_type_num: Count occurrences of each component type
    - flatten_mm_component_types: Flatten hierarchical structure with counts

Examples
--------
Extracting bonds from a SMILES string:

>>> from openff.toolkit import Molecule
>>> from molecular_classes import Bond
>>> components = get_mm_components_from_smiles("CCO", Bond)
>>> print(f"Found {len(components)} bonds")
Found 2 bonds

Processing multiple SMILES with parallel execution:

>>> smiles_list = ["CCO", "CCC", "CC(C)C"]
>>> all_bonds = get_all_mm_components(smiles_list, Bond)
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
from tqdm import tqdm

from openff.toolkit import Molecule

from .molecular_classes import MMComponent, SpecificityLevel


def get_mm_components_from_smiles(
    mapped_smiles: str,
    component_type: type[MMComponent],
) -> list[MMComponent]:
    """
    Extract molecular mechanics components from a mapped SMILES string.

    Converts a mapped SMILES string to an OpenFF Molecule and extracts all
    components of the specified type (bonds, angles, or torsions).

    Parameters
    ----------
    mapped_smiles : str
        SMILES string with atom mapping (e.g., "[C:1][C:2][O:3]").
    component_type : type[MMComponent]
        Type of component to extract (Bond, Angle, ProperTorsion, ImproperTorsion).

    Returns
    -------
    list[MMComponent]
        List of component objects extracted from the molecule.

    Raises
    ------
    ValueError
        If the mapped SMILES string is invalid or cannot be parsed.

    Examples
    --------
    >>> from molecular_classes import Bond
    >>> bonds = get_mm_components_from_smiles("[C:1][C:2][O:3]", Bond)
    >>> print(f"Found {len(bonds)} bonds")
    Found 2 bonds

    >>> from molecular_classes import Angle
    >>> angles = get_mm_components_from_smiles("[C:1][C:2][O:3]", Angle)
    >>> print(f"Found {len(angles)} angles")
    Found 1 angles
    """
    mol = Molecule.from_mapped_smiles(mapped_smiles, allow_undefined_stereo=True)
    if mol is None:
        raise ValueError(f"Invalid mapped SMILES: {mapped_smiles}")

    rdkit_mol = mol.to_rdkit()

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
    mapped_smiles_iterable: Iterable[str],
    component_type: type[MMComponent],
    unwanted_smirks: list[str] | None = None,
) -> list[MMComponent]:
    """
    Extract components from multiple SMILES strings with optional filtering.

    Processes a collection of mapped SMILES strings to extract all components
    of the specified type, with optional filtering to remove unwanted patterns.

    Parameters
    ----------
    mapped_smiles_iterable : Iterable[str]
        Collection of mapped SMILES strings to process.
    component_type : type[MMComponent]
        Type of component to extract (Bond, Angle, ProperTorsion, ImproperTorsion).
    unwanted_smirks : list[str], optional
        SMIRKS patterns to filter out from the results. Components matching
        any of these patterns will be excluded.

    Returns
    -------
    list[MMComponent]
        List of component objects from all valid SMILES strings, excluding
        any that match unwanted SMIRKS patterns.

    Notes
    -----
    Invalid SMILES strings are logged and skipped. The function provides
    progress tracking and filtering statistics via logging.

    Examples
    --------
    >>> from molecular_classes import Bond
    >>> smiles_list = ["[C:1][C:2][O:3]", "[C:1][C:2][C:3]"]
    >>> bonds = get_all_mm_components(smiles_list, Bond)
    >>> print(f"Total bonds: {len(bonds)}")
    Total bonds: 4

    >>> # Filter out C-O bonds
    >>> unwanted = ["[#6:1]-[#8:2]"]
    >>> filtered_bonds = get_all_mm_components(smiles_list, Bond, unwanted)
    >>> print(f"Filtered bonds: {len(filtered_bonds)}")
    Filtered bonds: 3
    """
    all_components = []
    for mapped_smiles in tqdm(mapped_smiles_iterable, desc="Processing Mapped SMILES"):
        try:
            all_components.extend(
                get_mm_components_from_smiles(mapped_smiles, component_type)
            )
        except ValueError as e:
            logger.info(f"Skipping invalid mapped SMILES {mapped_smiles}: {e}")

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
    Group components by their SMIRKS patterns at a given specificity level.

    Converts components (e.g., Bond) to SMIRKS representations and groups identical patterns
    together for parameterization and analysis.

    Parameters
    ----------
    mm_components : Iterable[MMComponent]
        Collection of components to group.
    specificity_level : SpecificityLevel
        Defines how specific the SMIRKS patterns should be (affects atom
        and bond descriptions).

    Returns
    -------
    dict[str, list[MMComponent]]
        Dictionary mapping SMIRKS patterns to lists of components that
        match those patterns.

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

    Parallel version of get_all_mm_components_by_type for handling large
    datasets efficiently by distributing work across multiple processes.

    Parameters
    ----------
    mm_components : Iterable[MMComponent]
        Collection of components to group.
    specificity_level : SpecificityLevel
        Defines how specific the SMIRKS patterns should be.
    n_workers : int, optional
        Number of worker processes to use. If None, uses all available CPU cores.

    Returns
    -------
    dict[str, list[MMComponent]]
        Dictionary mapping SMIRKS patterns to lists of components that
        match those patterns.

    Notes
    -----
    This function splits the component list into chunks, processes each chunk
    in parallel, and merges the results. It provides progress tracking and
    is essential for handling large molecular datasets efficiently.

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
        for f in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Processing Components (parallel)",
        ):
            results.append(f.result())

    return merge_dicts(results)


def get_mm_components_by_specificity_by_type(
    mm_components: Iterable[MMComponent],
    specificity_levels: dict[int, SpecificityLevel],
    cutoff_population: int = 10,
) -> dict[int, dict[str, list[MMComponent]]]:
    """
    Organize components hierarchically by specificity levels.

    Creates a hierarchical organization where components are first grouped at
    the highest specificity level, then rare patterns (below population cutoff)
    are moved to lower specificity levels for broader coverage.

    Parameters
    ----------
    mm_components : Iterable[MMComponent]
        Collection of components to organize.
    specificity_levels : dict[int, SpecificityLevel]
        Mapping from specificity level numbers to SpecificityLevel objects.
        Higher numbers indicate higher specificity.
    cutoff_population : int, default 10
        Minimum number of components required at a given specificity level.
        Patterns with fewer components are moved to lower specificity levels.

    Returns
    -------
    dict[int, dict[str, list[MMComponent]]]
        Nested dictionary structure:
        - First level: specificity level number
        - Second level: SMIRKS pattern string
        - Third level: list of components matching that pattern

    Notes
    -----
    The algorithm processes specificity levels from highest to lowest:
    1. Group all remaining components at current specificity level
    2. Keep patterns with sufficient population at this level
    3. Move rare patterns to next lower specificity level
    4. Repeat until all levels processed

    This ensures common patterns get specific parameters while rare patterns
    get broader, more general parameters for better coverage.

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
    >>>
    >>> # High specificity for common patterns
    >>> print(f"Level 2: {len(hierarchical[2])} specific patterns")
    >>> # Medium specificity for moderately rare patterns
    >>> print(f"Level 1: {len(hierarchical[1])} medium patterns")
    >>> # Low specificity for rare patterns
    >>> print(f"Level 0: {len(hierarchical[0])} general patterns")
    Level 2: 25 specific patterns
    Level 1: 15 medium patterns
    Level 0: 8 general patterns
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
    Count the number of occurrences for each component type.

    Converts component groupings to simple counts for analysis and statistics.

    Parameters
    ----------
    components_by_type : dict[str, list[MMComponent]]
        Dictionary mapping SMIRKS patterns to lists of components.

    Returns
    -------
    dict[str, int]
        Dictionary mapping SMIRKS patterns to their occurrence counts.

    Examples
    --------
    >>> components = {
    ...     "[#6:1]-[#6:2]": [bond1, bond2, bond3],
    ...     "[#6:1]-[#8:2]": [bond4, bond5]
    ... }
    >>> counts = get_mm_component_type_num(components)
    >>> print(counts)
    {"[#6:1]-[#6:2]": 3, "[#6:1]-[#8:2]": 2}
    """
    return {
        smirks: len(components) for smirks, components in components_by_type.items()
    }


def flatten_mm_component_types(
    components_by_specificity: dict[int, dict[str, list[MMComponent]]],
) -> dict[str, int]:
    """
    Flatten hierarchical component organization to simple counts.

    Converts the nested specificity level structure to a flat dictionary
    with total counts for each SMIRKS pattern across all specificity levels.

    Parameters
    ----------
    components_by_specificity : dict[int, dict[str, list[MMComponent]]]
        Hierarchical organization from get_mm_components_by_specificity_by_type.

    Returns
    -------
    dict[str, int]
        Dictionary mapping SMIRKS patterns to their total occurrence counts
        across all specificity levels.

    Notes
    -----
    If the same SMIRKS pattern appears at multiple specificity levels,
    the counts are summed. This is useful for getting overall statistics
    about pattern frequency in the dataset.

    Examples
    --------
    >>> hierarchical = {
    ...     0: {"[*:1]-[*:2]": [comp1, comp2]},
    ...     1: {"[#6:1]-[#6:2]": [comp3, comp4, comp5]}
    ... }
    >>> flattened = flatten_mm_component_types(hierarchical)
    >>> print(flattened)
    {"[*:1]-[*:2]": 2, "[#6:1]-[#6:2]": 3}
    """
    all_component_type_counts: Counter = Counter()
    for components_by_type in components_by_specificity.values():
        all_component_type_counts.update(get_mm_component_type_num(components_by_type))
    return dict(all_component_type_counts)
