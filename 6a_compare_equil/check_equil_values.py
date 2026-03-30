"""Check force field equilibrium values against dataset distributions.

This module compares force field equilibrium bond lengths and angles against
the actual distribution of these internal coordinates in training and validation
datasets. It generates box plots showing the spread of observed values with
the force field's equilibrium value overlaid in red.

The module uses HuggingFace datasets containing molecular geometries (SMILES and
coordinates) and assigns force field parameters directly from the OFFXML file.

Command-line Arguments
----------------------
--offxml : str
    Path to OFFXML force field file to analyze.
--dataset-paths : str, nargs='+'
    Paths to dataset directories (HuggingFace format with SMILES and coordinates).
    Multiple paths can be provided, separated by spaces.
--dataset-labels : str, nargs='+', optional
    Labels for each dataset to use in plots. Must match the number of dataset paths.
    If not provided, datasets will be labeled as 'Dataset 1', 'Dataset 2', etc.
--output-dir : str, optional
    Directory for output plots (default: current directory).
--n-processes : int, optional
    Number of parallel processes for molecule processing. If not specified,
    defaults to 1 (sequential processing). Recommended to set this to the
    number of allocated CPUs in your SLURM job (e.g., $SLURM_CPUS_PER_TASK).
    Note: The script automatically sets OMP_NUM_THREADS=1 to prevent
    thread oversubscription.

Examples
--------
Basic usage with a single dataset:
$ python check_equil_values.py \\
    --offxml ../4_make_offxmls/b_no_bond_with_ring_in_atom/openff_unconstrained.offxml \\
    --dataset-paths ../3_split_train_test/full_split_uci/data-train

With multiple datasets and custom labels:
$ python check_equil_values.py \\
    --offxml ../4_make_offxmls/b_no_bond_with_ring_in_atom/openff_unconstrained.offxml \\
    --dataset-paths ../3_split_train_test/full_split_uci/data-train ../3_split_train_test/full_split_uci/data-test \\
    --dataset-labels "Training" "Validation" \\
    --output-dir ./equilibrium_analysis

With multiple parallel processes:
$ python check_equil_values.py \\
    --offxml openff-2.2.1.offxml \\
    --dataset-paths ../3_split_train_test/full_split_uci/data-train \\
    --n-processes 8

With three or more datasets:
$ python check_equil_values.py \\
    --offxml openff-2.2.1.offxml \\
    --dataset-paths data-train data-test data-external \\
    --dataset-labels "Train" "Test" "External" \\
    --n-processes 8

Output Structure
----------------
Creates the following plots in output-dir/plots/:
- bond_equilibrium_comparison.pdf    # Bond lengths: all datasets side-by-side
- angle_equilibrium_comparison.pdf   # Angles: all datasets side-by-side

Creates the following cache files in output-dir/ic_values_cache/:
- dataset_0_ic_values.json    # Cached internal coordinate values from first dataset
- dataset_1_ic_values.json    # Cached internal coordinate values from second dataset
- ... (one file per dataset)

If cache files exist, dataset analysis will be skipped and cached values will be used.
To force reanalysis, delete the cache files.

Each plot shows:
- Box plots of observed values (different colors for each dataset)
- Green diamond marker at the force field equilibrium value
- Sample counts for each dataset
- Parameters sorted by first dataset frequency (most common at top)
"""

import argparse
import pathlib
import os
import gc
import json
from collections import defaultdict
from multiprocessing import Pool, freeze_support
import csv

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from loguru import logger
import datasets

from openff.toolkit import Molecule, ForceField
from openff.units import unit

logger.remove()
logger.add(lambda msg: print(msg, end=""), level="INFO")


def load_dataset(data_path: pathlib.Path | str) -> datasets.Dataset:
    """Load HuggingFace dataset from disk.

    Parameters
    ----------
    data_path : pathlib.Path | str
        Path to directory containing HuggingFace dataset.

    Returns
    -------
    datasets.Dataset
        Loaded dataset with molecular data (SMILES, coords, etc.).

    Raises
    ------
    FileNotFoundError
        If dataset directory does not exist.

    Examples
    --------
    >>> dataset = load_dataset("../3_split_train_test/full_split/data-train")
    >>> len(dataset)
    15000
    """
    data_path = pathlib.Path(data_path)

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    logger.info(f"Loading dataset from: {data_path}")
    dataset = datasets.Dataset.load_from_disk(str(data_path))
    logger.info(f"Loaded {len(dataset)} entries")

    return dataset


def get_force_field_parameters(
    ff: ForceField,
) -> dict[str, dict[str, float]]:
    """Extract equilibrium values from force field parameters.

    Parameters
    ----------
    ff : ForceField
        OpenFF ForceField object.

    Returns
    -------
    dict[str, dict[str, float]]
        Nested dict: {parameter_type: {smirks_id: equilibrium_value}}.
        For bonds: equilibrium_value is length in Angstroms.
        For angles: equilibrium_value is angle in degrees.

    Examples
    --------
    >>> ff = ForceField("openff-2.2.1.offxml")
    >>> params = get_force_field_parameters(ff)
    >>> params["Bonds"]["[#6X4:1]-[#6X4:2]"]
    1.526
    """
    logger.info("Extracting force field parameters...")

    parameters: dict[str, dict[str, float]] = {
        "Bonds": {},
        "Angles": {},
    }

    # Bond parameters
    bond_handler = ff.get_parameter_handler("Bonds")
    for i, param in enumerate(bond_handler.parameters):
        # Convert to Angstroms
        length_angstrom = param.length.to(unit.angstrom).magnitude
        # Use simple identifier: b{i}
        param_id = f"b{i}"
        parameters["Bonds"][param_id] = length_angstrom

    # Angle parameters
    angle_handler = ff.get_parameter_handler("Angles")
    for i, param in enumerate(angle_handler.parameters):
        # Convert to degrees
        angle_degrees = param.angle.to(unit.degree).magnitude
        # Use simple identifier: a{i}
        param_id = f"a{i}"
        parameters["Angles"][param_id] = angle_degrees

    logger.info(f"Extracted {len(parameters['Bonds'])} bond parameters")
    logger.info(f"Extracted {len(parameters['Angles'])} angle parameters")

    return parameters


def compute_internal_coordinate(
    coords: np.ndarray,
    indices: tuple[int, ...],
    ic_type: str,
) -> float:
    """Compute internal coordinate value from Cartesian coordinates.

    Parameters
    ----------
    coords : np.ndarray
        Cartesian coordinates array, shape (n_atoms, 3), in Angstroms.
    indices : tuple[int, ...]
        Atom indices for the internal coordinate.
    ic_type : str
        Type of internal coordinate: "Bond" or "Angle".

    Returns
    -------
    float
        Internal coordinate value.
        For bonds: distance in Angstroms.
        For angles: angle in degrees.

    Examples
    --------
    >>> coords = np.array([[0, 0, 0], [1.5, 0, 0], [2.0, 1.0, 0]])
    >>> compute_internal_coordinate(coords, (0, 1), "Bond")
    1.5
    >>> compute_internal_coordinate(coords, (0, 1, 2), "Angle")
    71.56...
    """
    if ic_type == "Bond":
        i, j = indices
        vec = coords[j] - coords[i]
        return float(np.linalg.norm(vec))

    elif ic_type == "Angle":
        i, j, k = indices
        vec1 = coords[i] - coords[j]
        vec2 = coords[k] - coords[j]

        cos_angle = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        # Clamp to [-1, 1] to avoid numerical issues
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle_rad = np.arccos(cos_angle)
        return float(np.degrees(angle_rad))

    else:
        raise ValueError(f"Unsupported IC type: {ic_type}")


class MoleculeProcessor:
    """Worker class for processing molecules with a shared force field.

    This class provides a clean way to manage the force field in worker processes.
    Each worker process will have its own instance with a loaded force field.
    """

    def __init__(self, force_field_path: str):
        """Initialize the processor with a force field.

        Parameters
        ----------
        force_field_path : str
            Path to the force field file.
        """
        self.force_field = ForceField(force_field_path)

        # Create mapping from parameter object to simple identifier
        # This allows us to map param objects to b{i} or a{i} identifiers
        bond_handler = self.force_field.get_parameter_handler("Bonds")
        self.bond_param_to_id = {
            param: f"b{i}" for i, param in enumerate(bond_handler.parameters)
        }

        angle_handler = self.force_field.get_parameter_handler("Angles")
        self.angle_param_to_id = {
            param: f"a{i}" for i, param in enumerate(angle_handler.parameters)
        }

    def __call__(self, args: tuple) -> dict | None:
        """Process a single molecule to extract internal coordinates.

        Parameters
        ----------
        args : tuple
            (smiles, coords_list)
            - smiles: SMILES string with atom mapping
            - coords_list: List of conformer coordinates (each as flat list)

        Returns
        -------
        dict or None
            Internal coordinate values grouped by SMIRKS, or None if processing failed.
            Format: {ic_type: {smirks_id: [values]}}
        """
        smiles, coords_list = args

        try:
            molecule = Molecule.from_mapped_smiles(smiles, allow_undefined_stereo=True)
            ff = self.force_field

            # Get parameter labels for this molecule
            labels = ff.label_molecules(molecule.to_topology())[0]

            param_labels: dict[str, dict[tuple, str]] = {
                "Bond": {},
                "Angle": {},
            }

            # Extract bond labels
            if "Bonds" in labels:
                for indices, param in labels["Bonds"].items():
                    norm_indices = tuple(sorted(indices))
                    # Use simple identifier from mapping
                    param_labels["Bond"][norm_indices] = self.bond_param_to_id[param]

            # Extract angle labels
            if "Angles" in labels:
                for indices, param in labels["Angles"].items():
                    # Use simple identifier from mapping
                    param_labels["Angle"][indices] = self.angle_param_to_id[param]

            # Collect internal coordinate values for this molecule
            mol_values: dict[str, dict[str, list[float]]] = {
                "Bond": defaultdict(list),
                "Angle": defaultdict(list),
            }

            n_atoms = molecule.n_atoms

            for coords_flat in coords_list:
                try:
                    # Reshape flat coordinates to (n_atoms, 3)
                    coords = np.array(coords_flat).reshape(n_atoms, 3)

                    # Compute bonds
                    for indices, smirks_id in param_labels["Bond"].items():
                        value = compute_internal_coordinate(coords, indices, "Bond")
                        mol_values["Bond"][smirks_id].append(value)

                    # Compute angles
                    for indices, smirks_id in param_labels["Angle"].items():
                        value = compute_internal_coordinate(coords, indices, "Angle")
                        mol_values["Angle"][smirks_id].append(value)

                except Exception as e:
                    logger.debug(f"Error processing conformer: {e}")
                    continue

            # Convert defaultdicts to regular dicts and convert to simple types
            result = {
                ic_type: {str(k): list(v) for k, v in smirks_dict.items()}
                for ic_type, smirks_dict in mol_values.items()
            }

            return result

        except Exception as e:
            logger.debug(f"Error processing molecule {smiles}: {e}")
            return None


def _dataset_generator(dataset: datasets.Dataset, batch_size: int = 100):
    """Generator that yields batches of dataset entries as plain Python objects.

    This avoids loading the entire dataset into memory while ensuring that
    file handles from the HuggingFace dataset are not passed to child processes.

    Parameters
    ----------
    dataset : datasets.Dataset
        HuggingFace dataset with molecular data.
    batch_size : int
        Number of entries to yield per batch.

    Yields
    ------
    list[tuple[str, list]]
        Batch of (smiles, coords_list) tuples as plain Python objects.
    """
    batch = []
    for i, entry in enumerate(dataset):
        # Convert to plain Python objects (no file handles)
        smiles = str(entry["smiles"])

        # Handle coords - the dataset stores ALL conformers as a single flattened 1D array
        # We need to figure out how many atoms and conformers there are
        coords = entry["coords"]

        # Debug: Print first entry to see structure
        if i == 0:
            logger.info(f"First entry debug - SMILES: {smiles[:50]}...")
            logger.info(f"Coords type: {type(coords)}")
            if hasattr(coords, "shape"):
                logger.info(f"Coords shape: {coords.shape}")

        # Convert torch tensor or numpy array to list
        if hasattr(coords, "tolist"):
            coords_flat = coords.tolist()
        else:
            coords_flat = list(coords)

        # The coords are stored as a single flat list containing all conformers
        # We need to infer the number of atoms from the SMILES
        # For now, just pass the entire flattened array and let the worker figure it out
        # Actually, we need to parse the molecule to get n_atoms, then split into conformers
        try:
            mol = Molecule.from_mapped_smiles(smiles, allow_undefined_stereo=True)
            n_atoms = mol.n_atoms
            coords_per_conf = n_atoms * 3
            n_conformers = len(coords_flat) // coords_per_conf

            # Split into list of conformers
            coords_list = []
            for conf_idx in range(n_conformers):
                start = conf_idx * coords_per_conf
                end = start + coords_per_conf
                coords_list.append(coords_flat[start:end])

            if i == 0:
                logger.info(f"  n_atoms: {n_atoms}, n_conformers: {n_conformers}")
                logger.info(f"  coords_per_conf: {coords_per_conf}")
                logger.info(f"  Coords_list length: {len(coords_list)}")
                if coords_list:
                    logger.info(f"  First conformer length: {len(coords_list[0])}")
        except Exception as e:
            logger.warning(f"Error parsing molecule at index {i}: {e}")
            continue

        batch.append((smiles, coords_list))

        if len(batch) >= batch_size:
            yield batch
            batch = []

    # Yield remaining items
    if batch:
        yield batch


def analyze_dataset_internal_coordinates(
    dataset: datasets.Dataset,
    force_field_path: str,
    n_processes: int | None = None,
    batch_size: int = 1000,
) -> dict[str, dict[str, list[float]]]:
    """Analyze internal coordinates in dataset grouped by SMIRKS parameter.

    Uses a memory-efficient batched approach to avoid loading the entire dataset
    into memory while still avoiding file handle issues with multiprocessing.

    Parameters
    ----------
    dataset : datasets.Dataset
        HuggingFace dataset with molecular data.
    force_field_path : str
        Path to force field file for parameter assignment.
    n_processes : int | None, optional
        Number of parallel processes. If None, defaults to 1 (sequential processing).
        Set explicitly for parallel processing (default: None).
    batch_size : int, optional
        Number of dataset entries to process in each batch. Larger batches use more
        memory but reduce overhead. Default: 1000.

    Returns
    -------
    dict[str, dict[str, list[float]]]
        Nested dict: {ic_type: {smirks_id: [values]}}.
        For bonds: values in Angstroms.
        For angles: values in degrees.

    Examples
    --------
    >>> # Auto-detect CPU count
    >>> ic_values = analyze_dataset_internal_coordinates(dataset, "openff-2.2.1.offxml")
    >>> np.mean(ic_values["Bond"]["[#6X4:1]-[#6X4:2]"])
    1.528

    >>> # Use specific number of processes
    >>> ic_values = analyze_dataset_internal_coordinates(dataset, "openff-2.2.1.offxml", n_processes=4)

    >>> # Sequential processing
    >>> ic_values = analyze_dataset_internal_coordinates(dataset, "openff-2.2.1.offxml", n_processes=1)

    >>> # Custom batch size for memory control
    >>> ic_values = analyze_dataset_internal_coordinates(dataset, "openff-2.2.1.offxml",
    ...                                                   n_processes=4, batch_size=500)
    """
    if n_processes is None:
        n_processes = 1
        print("No n_processes specified, defaulting to 1 (sequential processing)")

    print("Analyzing internal coordinates in dataset...")
    print(f"Using {n_processes} parallel process(es)")
    print(f"Processing in batches of {batch_size} entries (memory-efficient mode)")

    total_entries = len(dataset)
    print(f"Total dataset entries: {total_entries}")

    # Initialize result structure
    ic_values: dict[str, dict[str, list[float]]] = {
        "Bond": defaultdict(list),
        "Angle": defaultdict(list),
    }

    # Process molecules
    processed = 0
    last_percent_reported = -5

    if n_processes == 1:
        # Sequential processing - create processor once
        print("Processing molecules sequentially in batches...")
        processor = MoleculeProcessor(force_field_path)

        for batch in _dataset_generator(dataset, batch_size):
            for work_item in batch:
                result = processor(work_item)
                if result is not None:
                    # Merge results
                    for ic_type in ["Bond", "Angle"]:
                        for smirks_id, values in result[ic_type].items():
                            ic_values[ic_type][smirks_id].extend(values)

                processed += 1
                percent_complete = int(100 * processed / total_entries)
                if percent_complete >= last_percent_reported + 5:
                    print(
                        f"Progress: {percent_complete}% complete ({processed}/{total_entries} entries)"
                    )
                    last_percent_reported = percent_complete

            # Force garbage collection after each batch
            gc.collect()

    else:
        # Parallel processing with batched data loading
        print(f"Processing molecules in parallel with {n_processes} worker(s)...")

        # Use chunk size for pool.imap_unordered
        chunk_size = max(10, batch_size // (n_processes * 2))
        print(f"Using multiprocessing chunk size: {chunk_size}")

        # Create processor instance that will be pickled and sent to each worker
        processor = MoleculeProcessor(force_field_path)

        with Pool(processes=n_processes) as pool:
            # Process dataset in batches to avoid memory issues
            for batch in _dataset_generator(dataset, batch_size):
                # Use imap_unordered for better performance (order doesn't matter for aggregation)
                for result in pool.imap_unordered(
                    processor, batch, chunksize=chunk_size
                ):
                    if result is not None:
                        # Merge results
                        for ic_type in ["Bond", "Angle"]:
                            for smirks_id, values in result[ic_type].items():
                                ic_values[ic_type][smirks_id].extend(values)

                    processed += 1
                    percent_complete = int(100 * processed / total_entries)
                    if percent_complete >= last_percent_reported + 5:
                        print(
                            f"Progress: {percent_complete}% complete ({processed}/{total_entries} entries)"
                        )
                        last_percent_reported = percent_complete

                # Force garbage collection after each batch
                gc.collect()

    print(f"Analysis complete! Processed {processed} entries")

    # Count total values collected
    total_values = sum(
        len(values) for ic_dict in ic_values.values() for values in ic_dict.values()
    )
    print(f"Total internal coordinate values collected: {total_values}")

    # Convert defaultdicts to regular dicts
    result = {}
    for ic_type, smirks_dict in ic_values.items():
        result[ic_type] = dict(smirks_dict)

    return result


def save_ic_values_to_json(
    ic_values: dict[str, dict[str, list[float]]],
    output_path: pathlib.Path,
) -> None:
    """Save internal coordinate values to JSON file.

    Parameters
    ----------
    ic_values : dict[str, dict[str, list[float]]]
        Internal coordinate values: {ic_type: {smirks_id: [values]}}.
    output_path : pathlib.Path
        Path to save JSON file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(ic_values, f, indent=2)

    logger.info(f"Saved IC values to: {output_path}")


def load_ic_values_from_json(
    json_path: pathlib.Path,
) -> dict[str, dict[str, list[float]]]:
    """Load internal coordinate values from JSON file.

    Parameters
    ----------
    json_path : pathlib.Path
        Path to JSON file.

    Returns
    -------
    dict[str, dict[str, list[float]]]
        Internal coordinate values: {ic_type: {smirks_id: [values]}}.
    """
    with open(json_path, "r") as f:
        ic_values = json.load(f)

    logger.info(f"Loaded IC values from: {json_path}")
    return ic_values


def save_parameter_mapping(
    ff_params: dict[str, float],
    force_field: ForceField,
    ic_type: str,
    output_path: pathlib.Path,
) -> dict[str, str]:
    """Save mapping from simple IDs to SMIRKS patterns as CSV.

    Parameters
    ----------
    ff_params : dict[str, float]
        Force field parameters: {param_id: equilibrium_value}.
    force_field : ForceField
        OpenFF ForceField object to get SMIRKS patterns.
    ic_type : str
        Type of internal coordinate: "Bond" or "Angle".
    output_path : pathlib.Path
        Path to save CSV file.

    Returns
    -------
    dict[str, str]
        Mapping from param_id to simple_id (e.g., "b0", "a1").
    """

    # Get handler
    handler_name = "Bonds" if ic_type == "Bond" else "Angles"
    handler = force_field.get_parameter_handler(handler_name)

    # Create mapping: param_id -> simple_id
    id_prefix = "b" if ic_type == "Bond" else "a"
    param_to_simple_id = {}

    for i, param in enumerate(handler.parameters):
        simple_id = f"{id_prefix}{i}"
        param_id = param.id if param.id else param.smirks
        param_to_simple_id[param_id] = simple_id

    # Save to CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["simple_id", "param_id", "smirks", "equilibrium_value"])

        for param_id, simple_id in param_to_simple_id.items():
            # Get the actual parameter object to extract SMIRKS
            for param in handler.parameters:
                check_id = param.id if param.id else param.smirks
                if check_id == param_id:
                    equil_val = ff_params.get(param_id, "N/A")
                    writer.writerow([simple_id, param_id, param.smirks, equil_val])
                    break

    logger.info(f"Saved parameter mapping to: {output_path}")
    return param_to_simple_id


def save_parameter_molecule_mapping(
    dataset: datasets.Dataset,
    force_field: ForceField,
    output_path: pathlib.Path,
) -> None:
    """Save mapping of parameters to molecules that contain them.

    Creates a JSON file with format:
    {
        "bonds": {
            "b0": ["[C:1][C:2][O:3]", "[C:1][C:2][N:3]", ...],
            "b1": ["[C:1][H:2]", ...],
            ...
        },
        "angles": {
            "a0": ["[C:1][C:2][O:3]", ...],
            "a1": ["[C:1][C:2][N:3]", ...],
            ...
        }
    }

    Parameters
    ----------
    dataset : datasets.Dataset
        HuggingFace dataset with molecular data.
    force_field : ForceField
        OpenFF ForceField object.
    output_path : pathlib.Path
        Path to save JSON file.
    """
    logger.info("Building parameter-to-molecules mapping...")

    # Create mappings from parameter to simple ID
    bond_handler = force_field.get_parameter_handler("Bonds")
    bond_param_to_id = {
        param: f"b{i}" for i, param in enumerate(bond_handler.parameters)
    }

    angle_handler = force_field.get_parameter_handler("Angles")
    angle_param_to_id = {
        param: f"a{i}" for i, param in enumerate(angle_handler.parameters)
    }

    # Track which molecules contain each parameter
    # Format: {simple_id: list of SMILES}
    bond_to_molecules = defaultdict(list)
    angle_to_molecules = defaultdict(list)

    # Process each molecule in the dataset
    total = len(dataset)
    for i, entry in enumerate(dataset):
        if (i + 1) % 1000 == 0:
            logger.info(f"Processing molecule {i+1}/{total}...")

        smiles = str(entry["smiles"])

        try:
            molecule = Molecule.from_mapped_smiles(smiles, allow_undefined_stereo=True)
            labels = force_field.label_molecules(molecule.to_topology())[0]

            # Track which bond parameters this molecule has
            bond_params_in_mol = set()
            if "Bonds" in labels:
                for indices, param in labels["Bonds"].items():
                    simple_id = bond_param_to_id[param]
                    bond_params_in_mol.add(simple_id)

            # Add molecule to each bond parameter's list (only once per molecule)
            for simple_id in bond_params_in_mol:
                bond_to_molecules[simple_id].append(smiles)

            # Track which angle parameters this molecule has
            angle_params_in_mol = set()
            if "Angles" in labels:
                for indices, param in labels["Angles"].items():
                    simple_id = angle_param_to_id[param]
                    angle_params_in_mol.add(simple_id)

            # Add molecule to each angle parameter's list (only once per molecule)
            for simple_id in angle_params_in_mol:
                angle_to_molecules[simple_id].append(smiles)

        except Exception as e:
            logger.debug(f"Error processing molecule {i}: {e}")
            continue

    # Create the output structure
    mapping = {
        "bonds": {
            k: v
            for k, v in sorted(bond_to_molecules.items(), key=lambda x: int(x[0][1:]))
        },
        "angles": {
            k: v
            for k, v in sorted(angle_to_molecules.items(), key=lambda x: int(x[0][1:]))
        },
    }

    # Save to JSON
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(mapping, f, indent=2)

    # Print summary statistics
    logger.info(f"Bond parameters with molecules: {len(bond_to_molecules)}")
    logger.info(f"Angle parameters with molecules: {len(angle_to_molecules)}")
    logger.info(f"Saved parameter-to-molecules mapping to: {output_path}")


def save_statistics_to_csv(
    dataset_ic_values: list[dict[str, list[float]]],
    dataset_labels: list[str],
    ff_params: dict[str, float],
    ic_type: str,
    force_field: ForceField,
    output_path: pathlib.Path,
) -> None:
    """Save per-parameter statistics to a CSV file.

    For each force field parameter, writes its SMIRKS pattern, equilibrium
    value, and per-dataset descriptive statistics (n, mean, std, min, Q1,
    median, Q3, max, IQR, lower whisker, upper whisker).  Whisker bounds
    follow the Tukey convention used by matplotlib boxplot (Q1/Q3 ± 1.5×IQR,
    clamped to the observed min/max).

    Parameters
    ----------
    dataset_ic_values : list[dict[str, list[float]]]
        Observed IC values per dataset: [{param_id: [values]}, ...].
    dataset_labels : list[str]
        Labels for each dataset (used as column prefixes).
    ff_params : dict[str, float]
        Force field equilibrium values keyed by simple param ID (e.g. "b0").
    ic_type : str
        "Bond" or "Angle".
    force_field : ForceField
        OpenFF ForceField object used to look up SMIRKS patterns.
    output_path : pathlib.Path
        Destination CSV file path.

    Notes
    -----
    Columns written per dataset (prefixed with the dataset label)::

        {label}_n, {label}_mean, {label}_std, {label}_min,
        {label}_q1, {label}_median, {label}_q3, {label}_max,
        {label}_iqr, {label}_whisker_low, {label}_whisker_high
    """
    handler_name = "Bonds" if ic_type == "Bond" else "Angles"
    id_prefix = "b" if ic_type == "Bond" else "a"
    handler = force_field.get_parameter_handler(handler_name)
    simple_id_to_smirks = {
        f"{id_prefix}{i}": param.smirks for i, param in enumerate(handler.parameters)
    }

    sorted_param_ids = sorted(ff_params.keys(), key=lambda x: int(x[len(id_prefix) :]))

    stat_cols = [
        "n",
        "mean",
        "std",
        "min",
        "q1",
        "median",
        "q3",
        "max",
        "iqr",
        "whisker_low",
        "whisker_high",
    ]
    header = ["param_id", "smirks", "ff_equilibrium"]
    for label in dataset_labels:
        for stat in stat_cols:
            header.append(f"{label}_{stat}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for param_id in sorted_param_ids:
            smirks = simple_id_to_smirks.get(param_id, "")
            equil_val = ff_params.get(param_id, "")
            row: list = [param_id, smirks, equil_val]

            for dataset_ic_vals in dataset_ic_values:
                values = dataset_ic_vals.get(param_id, [])
                if values:
                    arr = np.array(values)
                    q1 = float(np.percentile(arr, 25))
                    median = float(np.percentile(arr, 50))
                    q3 = float(np.percentile(arr, 75))
                    iqr = q3 - q1
                    whisker_low = max(float(arr.min()), q1 - 1.5 * iqr)
                    whisker_high = min(float(arr.max()), q3 + 1.5 * iqr)
                    row.extend(
                        [
                            len(values),
                            float(arr.mean()),
                            float(arr.std()),
                            float(arr.min()),
                            q1,
                            median,
                            q3,
                            float(arr.max()),
                            iqr,
                            whisker_low,
                            whisker_high,
                        ]
                    )
                else:
                    row.extend([""] * len(stat_cols))

            writer.writerow(row)

    logger.info(f"Saved {ic_type} statistics CSV to: {output_path}")


def save_large_plot_with_fallback(
    fig,
    output_path: pathlib.Path,
    dpi: int = 600,
) -> bool:
    """Save large matplotlib figure with fallback strategies.

    Tries to save as PDF first. If that fails due to memory issues,
    provides helpful error messages and suggestions.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure to save.
    output_path : pathlib.Path
        Path to save the PDF file.
    dpi : int, optional
        DPI for the output (default: 600).

    Returns
    -------
    bool
        True if saved successfully, False otherwise.
    """
    logger.info(f"Attempting to save plot to: {output_path}")
    logger.info(
        f"Plot dimensions: {fig.get_figwidth():.1f} x {fig.get_figheight():.1f} inches"
    )

    # Ensure parent directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Strategy 1: Try direct PDF save
    logger.info("Rendering figure as PDF (this may take a while for large plots)...")
    try:
        fig.savefig(output_path, format="pdf", bbox_inches="tight", dpi=dpi)

        # Verify the file was created and has content
        if output_path.exists():
            file_size = output_path.stat().st_size
            if file_size > 0:
                logger.info(
                    f"✓ Successfully saved PDF: {output_path} ({file_size:,} bytes)"
                )
                return True
            else:
                logger.error(
                    f"✗ PDF file created but is empty (0 bytes): {output_path}"
                )
                return False
        else:
            logger.error(f"✗ PDF file was not created: {output_path}")
            return False

    except MemoryError as e:
        logger.error(f"✗ MemoryError while saving PDF: {e}")
        logger.error("The plot is too large to render as PDF with current memory")

        # Strategy 2: Try SVG as alternative vector format
        svg_path = output_path.with_suffix(".svg")
        logger.info(f"Attempting fallback to SVG format: {svg_path}")
        try:
            fig.savefig(svg_path, format="svg", bbox_inches="tight")
            if svg_path.exists() and svg_path.stat().st_size > 0:
                logger.info(f"✓ Successfully saved as SVG: {svg_path}")
                logger.info(
                    "You can convert SVG to PDF using: inkscape --export-pdf=output.pdf input.svg"
                )
                logger.info("Or use: cairosvg input.svg -o output.pdf")
                return False  # Didn't save PDF, but saved SVG
            else:
                logger.error("✗ SVG save also failed")
                return False
        except Exception as svg_e:
            logger.error(f"✗ SVG save failed: {svg_e}")
            return False

    except Exception as e:
        logger.error(f"✗ Error saving PDF: {e}")
        return False


def plot_equilibrium_comparison(
    dataset_ic_values: list[dict[str, list[float]]],
    dataset_labels: list[str],
    ff_params: dict[str, float],
    ic_type: str,
    output_path: pathlib.Path,
    param_id_mapping: dict[str, str] | None = None,
) -> None:
    """Generate box plot comparing dataset IC values to force field equilibrium.

    Parameters
    ----------
    dataset_ic_values : list[dict[str, list[float]]]
        List of observed internal coordinate values for each dataset.
        Each dict has format: {param_id: [values]}.
    dataset_labels : list[str]
        Labels for each dataset to use in the plot legend.
    ff_params : dict[str, float]
        Force field equilibrium values: {param_id: equilibrium_value}.
    ic_type : str
        Type of internal coordinate: "Bond" or "Angle".
    output_path : pathlib.Path
        Path to save the plot.
    param_id_mapping : dict[str, str], optional
        Mapping from param_id to simple_id (e.g., "b0", "a1").
        If provided, simple IDs will be used as y-axis labels.

    Examples
    --------
    >>> plot_equilibrium_comparison(
    ...     [bond_values_train, bond_values_val],
    ...     ["Training", "Validation"],
    ...     bond_params, "Bond",
    ...     Path("./plots/bond_comparison.pdf")
    ... )
    """
    if len(dataset_ic_values) != len(dataset_labels):
        raise ValueError(
            f"Number of datasets ({len(dataset_ic_values)}) must match "
            f"number of labels ({len(dataset_labels)})"
        )

    logger.info(
        f"Generating {ic_type} equilibrium comparison plot with {len(dataset_labels)} datasets..."
    )

    # Filter to only parameters with data in first dataset (for consistency)
    first_dataset_ic_values = dataset_ic_values[0]
    params_with_data = {
        param_id: values
        for param_id, values in first_dataset_ic_values.items()
        if param_id in ff_params and len(values) > 0
    }

    if not params_with_data:
        logger.warning(f"No {ic_type} parameters with data to plot")
        return

    # Sort by first dataset count (most common first)
    sorted_params = sorted(
        params_with_data.items(),
        key=lambda x: len(x[1]),
        reverse=True,
    )

    param_ids = [p[0] for p in sorted_params]
    equil_values = [ff_params[param_id] for param_id in param_ids]

    # Collect data and counts for all datasets
    all_dataset_values = []
    all_dataset_counts = []

    for dataset_ic_vals in dataset_ic_values:
        dataset_values = []
        dataset_counts = []
        for param_id in param_ids:
            if param_id in dataset_ic_vals and len(dataset_ic_vals[param_id]) > 0:
                dataset_values.append(dataset_ic_vals[param_id])
                dataset_counts.append(len(dataset_ic_vals[param_id]))
            else:
                dataset_values.append([])
                dataset_counts.append(0)
        all_dataset_values.append(dataset_values)
        all_dataset_counts.append(dataset_counts)

    # Determine plot size based on number of parameters
    fig_height = 3 + len(params_with_data) * 0.4
    fig, ax = plt.subplots(figsize=(12, fig_height))

    # Set up positions for side-by-side box plots
    # Each parameter gets a unit of space, with offsets for each dataset
    base_positions = np.arange(len(params_with_data))

    n_datasets = len(dataset_ic_values)

    # Define color palette for up to 10 datasets
    colors = [
        ("lightblue", "blue", "darkblue", "o"),
        ("lightcoral", "red", "darkred", "s"),
        ("lightgreen", "green", "darkgreen", "^"),
        ("lightyellow", "orange", "darkorange", "v"),
        ("plum", "purple", "indigo", "D"),
        ("lightcyan", "cyan", "darkcyan", "<"),
        ("peachpuff", "brown", "saddlebrown", ">"),
        ("lightgray", "gray", "black", "p"),
        ("pink", "hotpink", "deeppink", "h"),
        ("wheat", "gold", "darkgoldenrod", "*"),
    ]

    if n_datasets == 1:
        box_width = 0.6
        offsets = [0]
    else:
        # Calculate offsets to space datasets evenly
        box_width = 0.8 / n_datasets
        total_width = box_width * n_datasets
        offsets = np.linspace(
            -total_width / 2 + box_width / 2,
            total_width / 2 - box_width / 2,
            n_datasets,
        )

    # Create box plots for each dataset
    for dataset_idx, (dataset_values, dataset_label) in enumerate(
        zip(all_dataset_values, dataset_labels)
    ):
        positions = base_positions + offsets[dataset_idx]

        # Get colors for this dataset
        facecolor, edgecolor, mediancolor, marker = colors[dataset_idx % len(colors)]

        # Filter out empty data to avoid matplotlib errors
        filtered_data = []
        filtered_positions = []
        for i, (vals, pos) in enumerate(zip(dataset_values, positions)):
            if len(vals) > 0:
                filtered_data.append(vals)
                filtered_positions.append(pos)

        # Only create boxplot if we have some non-empty data
        if filtered_data:
            ax.boxplot(
                filtered_data,
                positions=filtered_positions,
                vert=False,
                widths=box_width,
                patch_artist=True,
                boxprops=dict(facecolor=facecolor, alpha=0.7, edgecolor=edgecolor),
                medianprops=dict(color=mediancolor, linewidth=2),
                whiskerprops=dict(color=edgecolor, linewidth=1),
                capprops=dict(color=edgecolor, linewidth=1),
                flierprops=dict(
                    marker=marker, markerfacecolor=edgecolor, markersize=3, alpha=0.5
                ),
            )
        else:
            logger.warning(
                f"No data to plot for {ic_type} in dataset '{dataset_label}' after filtering empty arrays"
            )

    # Overlay force field equilibrium values as green diamonds
    for i, (pos, equil_val) in enumerate(zip(base_positions, equil_values)):
        ax.plot(
            equil_val,
            pos,
            marker="D",
            markersize=8,
            color="darkgreen",
            markeredgecolor="black",
            markeredgewidth=0.5,
            zorder=10,
            label="FF Equilibrium" if i == 0 else "",
        )

    # Set labels and title
    if ic_type == "Bond":
        ax.set_xlabel("Bond Length (Å)", fontsize=12)
    elif ic_type == "Angle":
        ax.set_xlabel("Angle (degrees)", fontsize=12)
    else:
        ax.set_xlabel("Value", fontsize=12)
    ax.set_ylabel("Parameter ID", fontsize=12)

    # Set y-axis labels with counts
    # Use simple IDs if mapping is provided, otherwise use full param_ids
    if param_id_mapping is not None:
        display_ids = [
            param_id_mapping.get(param_id, param_id) for param_id in param_ids
        ]
    else:
        display_ids = param_ids

    # Build labels with counts for all datasets
    labels_with_counts = []
    for i, display_id in enumerate(display_ids):
        counts_str = ", ".join(
            f"{label}: {all_dataset_counts[ds_idx][i]}"
            for ds_idx, label in enumerate(dataset_labels)
        )
        labels_with_counts.append(f"{display_id}\n({counts_str})")

    ax.set_yticks(base_positions)
    ax.set_yticklabels(labels_with_counts, fontsize=10)

    # Invert y-axis so most common is at top
    ax.invert_yaxis()

    # Add legend
    legend_elements = []
    for dataset_idx, dataset_label in enumerate(dataset_labels):
        facecolor, edgecolor, _, _ = colors[dataset_idx % len(colors)]
        legend_elements.append(
            Patch(facecolor=facecolor, edgecolor=edgecolor, label=dataset_label)
        )
    legend_elements.append(
        Line2D(
            [0],
            [0],
            marker="D",
            color="w",
            markerfacecolor="darkgreen",
            markeredgecolor="black",
            markersize=8,
            label="FF Equilibrium",
        )
    )
    ax.legend(handles=legend_elements, loc="lower right", fontsize=10)

    # Add grid for easier reading
    ax.grid(axis="x", alpha=0.3, linestyle="--")

    fig.tight_layout()

    # Save with error handling and confirmation
    logger.info(f"Total parameters in plot: {len(params_with_data)}")
    success = save_large_plot_with_fallback(fig, output_path, dpi=600)

    plt.close(fig)
    logger.info("Figure closed and memory released")

    if not success:
        logger.warning(
            "Consider splitting the plot or reducing the number of parameters shown"
        )


def main(
    offxml: pathlib.Path | str,
    dataset_paths: list[pathlib.Path | str],
    dataset_labels: list[str] | None = None,
    output_dir: pathlib.Path | str = ".",
    n_processes: int | None = None,
    make_param_to_mol_mapping: bool = False,
) -> None:
    """Run equilibrium value comparison analysis.

    Parameters
    ----------
    offxml : pathlib.Path | str
        Path to OFFXML force field file.
    dataset_paths : list[pathlib.Path | str]
        Paths to dataset directories (HuggingFace format).
    dataset_labels : list[str], optional
        Labels for each dataset. If None, defaults to 'Dataset 1', 'Dataset 2', etc.
    output_dir : pathlib.Path | str, optional
        Output directory for plots (default: current directory).
    n_processes : int | None, optional
        Number of parallel processes. If None, defaults to 1 (sequential processing).
        Set explicitly for parallel processing (default: None).
    make_param_to_mol_mapping : bool, optional
        If True, a csv is saved where for each parameter type, the molecules that use
        it are listed. (default: False)

    Examples
    --------
    >>> # Single dataset
    >>> main("openff-2.2.1.offxml", ["data-train"])

    >>> # Multiple datasets with labels
    >>> main("openff-2.2.1.offxml", ["data-train", "data-test"],
    ...      dataset_labels=["Training", "Validation"], n_processes=4)

    >>> # Three datasets
    >>> main("openff-2.2.1.offxml", ["data-train", "data-test", "data-external"],
    ...      dataset_labels=["Train", "Test", "External"])
    """
    offxml = pathlib.Path(offxml)
    dataset_paths_resolved: list[pathlib.Path] = [
        pathlib.Path(p) for p in dataset_paths
    ]
    output_dir = pathlib.Path(output_dir)

    # Generate default labels if not provided
    if dataset_labels is None:
        dataset_labels = [f"Dataset {i+1}" for i in range(len(dataset_paths_resolved))]

    # Validate inputs
    if not offxml.exists():
        raise FileNotFoundError(f"Force field not found: {offxml}")

    if len(dataset_labels) != len(dataset_paths_resolved):
        raise ValueError(
            f"Number of dataset labels ({len(dataset_labels)}) must match "
            f"number of dataset paths ({len(dataset_paths_resolved)})"
        )

    for i, dataset_path in enumerate(dataset_paths_resolved):
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset {i+1} not found: {dataset_path}")

    # Load force field parameters
    logger.info(f"Loading force field: {offxml}")
    ff = ForceField(str(offxml))
    ff_params = get_force_field_parameters(ff)

    # Create output directory for JSON cache files
    cache_dir = output_dir / "ic_values_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Load or analyze all datasets
    all_ic_values = []
    for i, (dataset_path, dataset_label) in enumerate(
        zip(dataset_paths_resolved, dataset_labels)
    ):
        cache_file = cache_dir / f"dataset_{i}_ic_values.json"

        if cache_file.exists():
            logger.info(f"Found cached IC values for '{dataset_label}': {cache_file}")
            logger.info(
                f"Skipping '{dataset_label}' dataset analysis (delete cache to rerun)"
            )
            ic_values = load_ic_values_from_json(cache_file)
        else:
            dataset = load_dataset(dataset_path)
            print(f"\nAnalyzing '{dataset_label}' dataset...")
            ic_values = analyze_dataset_internal_coordinates(
                dataset, str(offxml), n_processes=n_processes
            )
            # Save to cache
            save_ic_values_to_json(ic_values, cache_file)
            # Clean up dataset reference to free file handles
            del dataset
            gc.collect()

        all_ic_values.append(ic_values)

    mappings_dir = output_dir / "parameter_mappings"
    mappings_dir.mkdir(parents=True, exist_ok=True)

    # Create and save parameter mappings
    bond_mapping = save_parameter_mapping(
        ff_params["Bonds"],
        ff,
        "Bond",
        mappings_dir / "bond_parameter_mapping.csv",
    )

    angle_mapping = save_parameter_mapping(
        ff_params["Angles"],
        ff,
        "Angle",
        mappings_dir / "angle_parameter_mapping.csv",
    )

    if make_param_to_mol_mapping:
        print("\nAnalyzing which molecules contain each parameter...")
        filename = mappings_dir / "parameter_to_molecules.json"
        if not os.path.isfile(filename):
            # Use first dataset for mapping
            first_dataset = load_dataset(dataset_paths_resolved[0])
            save_parameter_molecule_mapping(
                first_dataset,
                ff,
                filename,
            )
            del first_dataset
            gc.collect()

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    bond_data = [ic_vals["Bond"] for ic_vals in all_ic_values]
    angle_data = [ic_vals["Angle"] for ic_vals in all_ic_values]

    # Save per-parameter statistics CSVs first (independent of plotting)
    save_statistics_to_csv(
        bond_data,
        dataset_labels,
        ff_params["Bonds"],
        "Bond",
        ff,
        plots_dir / "bond_statistics.csv",
    )
    save_statistics_to_csv(
        angle_data,
        dataset_labels,
        ff_params["Angles"],
        "Angle",
        ff,
        plots_dir / "angle_statistics.csv",
    )

    # Generate plots with all datasets
    print("\nGenerating plots...")
    plot_equilibrium_comparison(
        bond_data,
        dataset_labels,
        ff_params["Bonds"],
        "Bond",
        plots_dir / "bond_equilibrium_comparison.pdf",
        param_id_mapping=bond_mapping,
    )

    plot_equilibrium_comparison(
        angle_data,
        dataset_labels,
        ff_params["Angles"],
        "Angle",
        plots_dir / "angle_equilibrium_comparison.pdf",
        param_id_mapping=angle_mapping,
    )

    logger.info("Analysis complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Check force field equilibrium values against dataset distributions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    Basic usage with a single dataset:
        python check_equil_values.py \\
            --offxml ../4_make_offxmls/b_no_bond_with_ring_in_atom/openff_unconstrained.offxml \\
            --dataset-paths ../3_split_train_test/full_split_uci/data-train

    With multiple datasets and custom labels:
        python check_equil_values.py \\
            --offxml ../4_make_offxmls/b_no_bond_with_ring_in_atom/openff_unconstrained.offxml \\
            --dataset-paths ../3_split_train_test/full_split_uci/data-train \\
                            ../3_split_train_test/full_split_uci/data-test \\
            --dataset-labels "Training" "Validation" \\
            --output-dir ./equilibrium_analysis

    With multiple parallel processes:
        python check_equil_values.py \\
            --offxml openff-2.2.1.offxml \\
            --dataset-paths data-train data-test \\
            --dataset-labels "Train" "Test" \\
            --n-processes 8

    In a SLURM script (recommended):
        python check_equil_values.py \\
            --offxml openff-2.2.1.offxml \\
            --dataset-paths data-train data-test data-external \\
            --dataset-labels "Train" "Test" "External" \\
            --n-processes ${SLURM_CPUS_PER_TASK}

Output Files:
    plots/
    ├── bond_equilibrium_comparison.pdf    # Bond comparison (all datasets)
    └── angle_equilibrium_comparison.pdf   # Angle comparison (all datasets)

    ic_values_cache/
    ├── dataset_0_ic_values.json           # Cached IC values for first dataset
    ├── dataset_1_ic_values.json           # Cached IC values for second dataset
    └── ...                                # (one file per dataset)

    parameter_mappings/
    ├── bond_parameter_mapping.csv         # Mapping of parameter IDs to SMIRKS
    └── angle_parameter_mapping.csv        # Mapping of parameter IDs to SMIRKS

Caching:
    The script automatically caches analyzed IC values to JSON files. If cache files
    exist, dataset analysis is skipped and cached values are loaded instead.
    To force reanalysis, delete the cache files in ic_values_cache/.

Each plot shows:
    - Side-by-side box plots for all datasets (different colors)
    - Green diamond marker at the force field's equilibrium value
    - Sample counts for each dataset
    - Parameters sorted by first dataset frequency (most common at top)
        """,
    )
    parser.add_argument(
        "--offxml",
        type=str,
        required=True,
        help="Path to OFFXML force field file",
    )
    parser.add_argument(
        "--dataset-paths",
        type=str,
        nargs="+",
        required=True,
        help="Paths to dataset directories (HuggingFace format). Separate multiple paths with spaces.",
    )
    parser.add_argument(
        "--dataset-labels",
        type=str,
        nargs="+",
        default=None,
        help="Labels for each dataset. Must match number of dataset paths. "
        "If not provided, defaults to 'Dataset 1', 'Dataset 2', etc.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Directory for output plots (default: current directory)",
    )
    parser.add_argument(
        "--n-processes",
        type=int,
        default=None,
        help="Number of parallel processes (default: 1 for sequential). "
        "Recommended: set to $SLURM_CPUS_PER_TASK in SLURM jobs. "
        "Script automatically sets OMP_NUM_THREADS=1 to prevent thread oversubscription.",
    )
    parser.add_argument(
        "--make-param-to-mol-mapping",
        type=bool,
        default=False,
        help="If True, a csv is saved where for each parameter type, the molecules that use it are listed.",
    )

    args = parser.parse_args()

    freeze_support()
    main(
        offxml=args.offxml,
        dataset_paths=args.dataset_paths,
        dataset_labels=args.dataset_labels,
        output_dir=args.output_dir,
        n_processes=args.n_processes,
        make_param_to_mol_mapping=args.make_param_to_mol_mapping,
    )
