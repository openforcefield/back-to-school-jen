"""
Force Field Coverage Analysis and Parameterization Validation.

This module provides utilities for analyzing force field coverage of molecular
datasets, identifying missing parameters, and validating parameterization
completeness. It supports both individual molecule analysis and parallel
processing of large molecular datasets.

Key Functions
-------------
Parameterization Validation:
    - check_molecule_can_be_parameterised: Test if a molecule can be fully parameterized
    - check_all_molecules_parameterisable_parallel_chunks: Parallel validation for datasets

Component Coverage Analysis:
    - check_all_components_fully_covered: Identify missing parameters for all component types
    - check_all_components_fully_covered_parallel_chunks: Parallel version for large datasets
    - check_torsions_fully_covered: Specific analysis for torsion parameters

Utilities:
    - chunked_iterable: Split iterables into chunks for parallel processing

Examples
--------
Check if a single molecule can be parameterized:

>>> from openff.toolkit import ForceField
>>> ff = ForceField("openff-2.0.0.offxml")
>>> can_param = check_molecule_can_be_parameterised("[C:1][C:2][O:3]", ff)
>>> print(f"Can parameterize: {can_param}")
Can parameterize: True

Analyze coverage for multiple molecules:

>>> smiles_list = ["[C:1][C:2][O:3]", "[C:1][C:2][C:3]"]
>>> missing = check_all_components_fully_covered_parallel_chunks(smiles_list, ff)
>>> if missing:
...     print(f"Found missing parameters for {len(missing)} molecules")
... else:
...     print("All molecules fully covered")

Check specific torsion coverage:

>>> covered, missing_torsions = check_torsions_fully_covered("[C:1][C:2][C:3][C:4]", ff)
>>> if not covered:
...     print(f"Missing {len(missing_torsions)} torsion parameters")
"""

import os
import math
from concurrent.futures import ProcessPoolExecutor, as_completed

from tqdm import tqdm
from loguru import logger

from openff.units import unit as off_unit
from openff.toolkit import Molecule, ForceField
from openff.toolkit.topology import Topology
from openff.interchange import Interchange

from .molecular_classes import Bond, Angle, ProperTorsion, ImproperTorsion


def check_molecule_can_be_parameterised(mapped_smiles: str, ff: ForceField) -> bool:
    """
    Test if a molecule can be fully parameterized by a force field.

    Attempts to create an Interchange object from the molecule and force field
    to validate that all required parameters are available. Uses dummy charges
    to bypass charge assignment and focus purely on parameter availability.

    Parameters
    ----------
    mapped_smiles : str
        Mapped SMILES string representing the molecule to test.
    ff : openff.toolkit.ForceField
        Force field to test parameterization against.

    Returns
    -------
    bool
        True if the molecule can be fully parameterized, False otherwise.

    Notes
    -----
    This function uses dummy charges (all zeros) to skip the charge assignment
    step, ensuring that any failures are due to missing force field parameters
    rather than charge assignment issues.

    Examples
    --------
    >>> from openff.toolkit import ForceField
    >>> ff = ForceField("openff-2.0.0.offxml")
    >>> can_param = check_molecule_can_be_parameterised("[C:1][C:2][O:3]", ff)
    >>> print(f"Ethanol parameterizable: {can_param}")
    Ethanol parameterizable: True

    >>> # Test unusual molecule that might lack parameters
    >>> exotic = check_molecule_can_be_parameterised("[Si:1][Si:2]", ff)
    >>> print(f"Silicon bond parameterizable: {exotic}")
    Silicon bond parameterizable: False
    """
    mol = Molecule.from_mapped_smiles(mapped_smiles, allow_undefined_stereo=True)
    mol._partial_charges = [
        0.0 for _ in range(mol.n_atoms)
    ] * off_unit.elementary_charge  # Dummy charges to skip charge assignment
    try:
        Interchange.from_smirnoff(
            force_field=ff, topology=mol.to_topology()
        )  # , charge_from_molecules=True)
        return True
    except Exception as e:
        logger.info(f"Error parameterising molecule {mapped_smiles}: {e}")
        return False


def check_molecules_fully_covered_chunk(smiles_chunk, ff):
    """
    Process a chunk of SMILES for parallel parameterization testing.

    Worker function that tests parameterization capability for a subset
    of molecules in parallel processing workflows.

    Parameters
    ----------
    smiles_chunk : list[str]
        List of mapped SMILES strings to test.
    ff : openff.toolkit.ForceField
        Force field to test against.

    Returns
    -------
    list[tuple[str, bool]]
        List of tuples containing (smiles, can_parameterize) for each molecule.

    Notes
    -----
    This function is designed for use with ProcessPoolExecutor and should
    not be called directly. Use check_all_molecules_parameterisable_parallel_chunks
    instead.
    """
    return [
        (smiles, check_molecule_can_be_parameterised(smiles, ff))
        for smiles in smiles_chunk
    ]


def chunked_iterable(iterable, chunk_size):
    """
    Split an iterable into chunks of specified size.

    Yields successive chunks from an iterable, useful for parallel processing
    of large datasets by distributing work across multiple processes.

    Parameters
    ----------
    iterable : Iterable
        The iterable to split into chunks.
    chunk_size : int
        Maximum size of each chunk.

    Yields
    ------
    list
        Successive chunks of the input iterable.

    Examples
    --------
    >>> data = range(10)
    >>> chunks = list(chunked_iterable(data, 3))
    >>> print(chunks)
    [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]

    >>> smiles_list = ["CC", "CCO", "CCC", "CCCC"]
    >>> for chunk in chunked_iterable(smiles_list, 2):
    ...     print(f"Processing chunk: {chunk}")
    Processing chunk: ['CC', 'CCO']
    Processing chunk: ['CCC', 'CCCC']
    """
    it = iter(iterable)
    while True:
        chunk = []
        try:
            for _ in range(chunk_size):
                chunk.append(next(it))
        except StopIteration:
            if chunk:
                yield chunk
            break
        if chunk:
            yield chunk


def check_all_molecules_parameterisable_parallel_chunks(
    mapped_smiles_list, ff, n_workers: int | None = None
):
    """
    Test parameterization capability for multiple molecules using parallel processing.

    Distributes parameterization testing across multiple processes for efficient
    analysis of large molecular datasets.

    Parameters
    ----------
    mapped_smiles_list : list[str]
        List of mapped SMILES strings to test.
    ff : openff.toolkit.ForceField
        Force field to test parameterization against.
    n_workers : int, optional
        Number of worker processes to use. If None, uses all available CPU cores.

    Returns
    -------
    list[tuple[str, bool]]
        List of tuples containing (smiles, can_parameterize) for each molecule.

    Notes
    -----
    This function automatically determines optimal chunk sizes based on the
    number of workers to ensure balanced workload distribution.

    Examples
    --------
    >>> from openff.toolkit import ForceField
    >>> ff = ForceField("openff-2.0.0.offxml")
    >>> smiles_list = ["[C:1][C:2]", "[C:1][O:2]", "[N:1][C:2]"]
    >>> results = check_all_molecules_parameterisable_parallel_chunks(smiles_list, ff)
    >>> parameterizable = [smiles for smiles, can_param in results if can_param]
    >>> print(f"{len(parameterizable)}/{len(smiles_list)} molecules parameterizable")
    3/3 molecules parameterizable
    """
    results = []
    if n_workers is None:
        n_workers = os.cpu_count() or 1  # Fallback to 1 if cpu_count() returns None
    # Calculate the chunk size to be the maximum possible while still having at least one chunk per worker
    chunk_size = math.ceil(len(mapped_smiles_list) / n_workers)
    chunks = list(chunked_iterable(mapped_smiles_list, chunk_size))
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = [
            executor.submit(check_molecules_fully_covered_chunk, chunk, ff)
            for chunk in chunks
        ]
        for f in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Checking Parameterisability (parallel chunks)",
        ):
            results.extend(f.result())
    return results


def check_all_components_fully_covered(
    mapped_smiles: list[str], ff: ForceField
) -> dict[str, dict[str, list[tuple[int, ...]]]]:
    """
    Identify missing force field parameters for all molecular mechanics components.

    Analyzes bonds, angles, proper torsions, and improper torsions for a list of
    molecules to determine which components cannot be parameterized by the given
    force field. Returns detailed information about missing parameters organized
    by molecule and component type.

    Parameters
    ----------
    mapped_smiles : list[str]
        List of mapped SMILES strings representing molecules to analyze.
        Each SMILES must have atom mapping to enable proper component identification.
    ff : openff.toolkit.ForceField
        Force field to test component coverage against.

    Returns
    -------
    dict[str, dict[str, list[tuple[int, ...]]]]
        Dictionary mapping SMILES strings to their missing parameters. Structure:
        {
            "smiles_string": {
                "Bonds": [(atom1, atom2), ...],
                "Angles": [(atom1, atom2, atom3), ...],
                "ProperTorsions": [(atom1, atom2, atom3, atom4), ...],
                "ImproperTorsions": [(atom1, atom2, atom3, atom4), ...]
            },
            ...
        }
        Only molecules with missing parameters are included in the result.
        Empty dict means all molecules are fully parameterized.

    Notes
    -----
    This function uses the OpenFF Toolkit's labeling system to identify which
    molecular mechanics components can be assigned parameters. Components that
    cannot be labeled are considered missing from the force field.

    The function processes molecules sequentially with a progress bar. For large
    datasets, consider using the parallel version.

    Examples
    --------
    >>> from openff.toolkit import ForceField
    >>> ff = ForceField("openff-2.0.0.offxml")
    >>> molecules = ["[C:1][C:2]", "[C:1][O:2]", "[Si:1][Si:2]"]
    >>> missing = check_all_components_fully_covered(molecules, ff)
    >>> if missing:
    ...     for smiles, components in missing.items():
    ...         print(f"Molecule {smiles} missing:")
    ...         for comp_type, indices in components.items():
    ...             print(f"  {comp_type}: {len(indices)} missing")
    ... else:
    ...     print("All molecules fully parameterized")
    Molecule [Si:1][Si:2] missing:
      Bonds: 1 missing

    >>> # Check specific molecule details
    >>> if "[Si:1][Si:2]" in missing:
    ...     si_missing = missing["[Si:1][Si:2]"]
    ...     bond_indices = si_missing.get("Bonds", [])
    ...     print(f"Missing Si-Si bond between atoms: {bond_indices[0]}")
    Missing Si-Si bond between atoms: (0, 1)
    """
    all_unassigned = {}
    logger.info("Checking Component Coverage")
    for smiles in mapped_smiles:
        mol = Molecule.from_mapped_smiles(smiles, allow_undefined_stereo=True)
        top = Topology.from_molecules([mol])
        labels = ff.label_molecules(top)[0]  # As only one molecule

        unassigned = {}
        for component_class in [Bond, Angle, ProperTorsion, ImproperTorsion]:
            component_type = component_class.handler_class._TAGNAME
            component_params = labels[
                component_type
            ]  # dict with keys being atom index tuples and values being parameters
            covered_indices = list(component_params.keys())
            unassigned_indices = []
            for at_indices in component_class.getter_fn(mol):
                if at_indices not in covered_indices:
                    unassigned_indices.append(at_indices)
            if unassigned_indices:
                unassigned[component_type] = unassigned_indices
        #                logger.debug(covered_indices)
        #                logger.debug(component_class.getter_fn(mol))

        if unassigned:
            all_unassigned[smiles] = unassigned

    return all_unassigned


def check_all_components_fully_covered_parallel_chunks(
    mapped_smiles_list: list[str], ff: ForceField, n_workers: int | None = None
) -> dict[str, dict[str, list[tuple[int, ...]]]]:
    """
    Check component coverage for multiple molecules using parallel processing.

    Distributes component coverage analysis across multiple processes for efficient
    processing of large molecular datasets.

    Parameters
    ----------
    mapped_smiles_list : list[str]
        List of mapped SMILES strings to analyze.
    ff : openff.toolkit.ForceField
        Force field to test component coverage against.
    n_workers : int, optional
        Number of worker processes to use. If None, uses all available CPU cores.

    Returns
    -------
    dict[str, dict[str, list[tuple[int, ...]]]]
        Dictionary mapping SMILES to component coverage results.

    Notes
    -----
    This function distributes chunks across multiple processes for efficient
    analysis of large molecular datasets. Each chunk is processed independently
    and results are merged.

    Examples
    --------
    >>> from openff.toolkit import ForceField
    >>> ff = ForceField("openff-2.0.0.offxml")
    >>> smiles_list = ["[C:1][C:2]", "[C:1][O:2]", "[Si:1][Si:2]"]
    >>> results = check_all_components_fully_covered_parallel_chunks(smiles_list, ff)
    >>> print(f"Found missing parameters in {len(results)} molecules")
    Found missing parameters in 1 molecules
    """
    results = {}
    if n_workers is None:
        n_workers = os.cpu_count() or 1  # Fallback to 1 if cpu_count() returns None
    # Calculate the chunk size to be the maximum possible while still having at least one chunk per worker
    chunk_size = math.ceil(len(mapped_smiles_list) / n_workers)
    chunks = list(chunked_iterable(mapped_smiles_list, chunk_size))
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = [
            executor.submit(check_all_components_fully_covered, chunk, ff)
            for chunk in chunks
        ]
        for f in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Checking Component Coverage (parallel chunks)",
        ):
            results.update(f.result())
    return results


def check_torsions_fully_covered(
    mapped_smiles: str, ff: ForceField
) -> tuple[bool, list[tuple[int, int, int, int]]]:
    """
    Check if all proper torsions in a molecule are covered by the force field.

    Analyzes proper torsion coverage by attempting to assign force field parameters
    to all proper torsions in the molecule and identifying any unassigned torsions.

    Parameters
    ----------
    mapped_smiles : str
        Mapped SMILES string representing the molecule to analyze.
    ff : openff.toolkit.ForceField
        Force field to test torsion coverage against.

    Returns
    -------
    tuple[bool, list[tuple[int, int, int, int]]]
        Tuple containing:
        - bool: True if all torsions are covered, False otherwise
        - list: Atom indices of unassigned torsions as (i, j, k, l) tuples

    Notes
    -----
    This function specifically focuses on proper torsions and uses the force field
    labeling system to identify missing torsion parameters. Improper torsions
    are not analyzed by this function.

    Examples
    --------
    >>> from openff.toolkit import ForceField
    >>> ff = ForceField("openff-2.0.0.offxml")
    >>> all_covered, unassigned = check_torsions_fully_covered("[C:1][C:2][O:3]", ff)
    >>> if all_covered:
    ...     print("All torsions are parameterized")
    ... else:
    ...     print(f"Missing parameters for {len(unassigned)} torsions")
    All torsions are parameterized

    >>> # Test molecule with potentially missing parameters
    >>> covered, missing = check_torsions_fully_covered("[Si:1][Si:2][Si:3][Si:4]", ff)
    >>> print(f"Coverage: {covered}, Missing torsions: {len(missing)}")
    Coverage: False, Missing torsions: 1
    """
    mol = Molecule.from_mapped_smiles(mapped_smiles, allow_undefined_stereo=True)
    # mol.generate_conformers(n_conformers=1)  # Needed to avoid issues in Topology

    top = Topology.from_molecules([mol])
    labels = ff.label_molecules(top)
    torsion_params = labels[0][
        "ProperTorsions"
    ]  # list of (parameter, atom_indices) tuples
    covered_indices = list(torsion_params.keys())

    unassigned = []
    for torsion in mol.propers:
        atom_indices = tuple(a.molecule_atom_index for a in torsion)
        if atom_indices not in covered_indices:
            unassigned.append(atom_indices)

    return len(unassigned) == 0, unassigned
