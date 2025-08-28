"""Functions to obtain and process data from QCArchive for the workflow.

This module provides functionality to download, process, and filter QCArchive datasets
for molecular property prediction workflows. It includes functions to:

1. Retrieve a dataset from QCArchive
2. Process the QCArchive dataset into a structured dataset

The module can be used as a command-line script or imported as a library.

Command-line Usage
------------------
python get_data_qca.py --datasets "Dataset Name" --dataset_type TYPE --data_file PATH

Command-line Arguments
----------------------
--datasets : List[str]
    One or more QCArchive dataset names to retrieve and process. Multiple
    dataset names should be separated by spaces. Each dataset name should
    be quoted if it contains spaces.

--dataset_type : str
    Type of datasets to retrieve. Must be one of:
    - 'optimization' : Geometry optimization datasets
    - 'singlepoint' : Single-point energy calculations
    - 'torsiondrive' : Torsion drive scan datasets

--data_file : str
    Output path for the processed dataset. The data will be saved in
    HuggingFace datasets format with Arrow files.

Examples
--------
# Process single optimization dataset
python get_data_qca.py --datasets "OpenFF Gen 2 Opt Set 1 Roche" \\
                       --dataset_type optimization \\
                       --data_file ./qcarchive_data

# Process multiple torsiondrive datasets
python get_data_qca.py --datasets "OpenFF Gen 2 Torsion Set 1 Roche" \\
                                  "OpenFF Gen 2 Torsion Set 2 Coverage" \\
                       --dataset_type torsiondrive \\
                       --data_file ./torsion_data

Outputs
-------
data-dir : dir
├── dataset_info.json          # Dataset metadata and schema
├── state.json                 # Dataset state information
└── data-00000-of-00001.arrow  # Actual data in Apache Arrow format
    ├── smiles (str)
    ├── coords (list) # flattened, reshape with np.array(coords_list).reshape(-1, 3)
    ├── energy (list) # flattened
    └── forces (list) # flattened

Constants
---------
HARTREE_TO_KCAL : float
    Conversion factor from Hartree to kcal/mol
BOHR_TO_ANGSTROM : float
    Conversion factor from Bohr to Angstrom
"""

from __future__ import annotations

import os
import argparse
import json
import pathlib
from typing import List, Union
from collections import defaultdict

import descent.targets.energy
import torch
import numpy as np
from loguru import logger
from openff.units import unit
from tqdm import tqdm

import qcportal
from openff.qcsubmit.results import (
    BasicResultCollection,
    OptimizationResultCollection,
    TorsionDriveResultCollection,
)

collection_types = {
    "optimization": OptimizationResultCollection,
    "singlepoint": BasicResultCollection,
    "torsiondrive": TorsionDriveResultCollection,
}


HARTREE_TO_KCAL: float = (1 * unit.hartree * unit.avogadro_constant).m_as(
    unit.kilocalories_per_mole
)
BOHR_TO_ANGSTROM: float = (1.0 * unit.bohr).m_as(unit.angstrom)


def retrieve_datasets(
    dataset_names: List[str], dataset_type: str, spec_name: str = "default"
) -> Union[
    BasicResultCollection, OptimizationResultCollection, TorsionDriveResultCollection
]:
    """Downloads datasets from QCArchive and returns a QCSubmit result collection.

    Connects to the QCArchive portal and retrieves the specified datasets
    based on their type and specification name. Supports optimization,
    singlepoint, and torsiondrive dataset types.

    Parameters
    ----------
    dataset_names : List[str]
        List of dataset names to retrieve from QCArchive.
    dataset_type : str
        Type of dataset to retrieve. Must be one of: 'optimization',
        'singlepoint', or 'torsiondrive'.
    spec_name : str, optional
        Specification name for the dataset retrieval, by default "default".

    Returns
    -------
    Union[BasicResultCollection, OptimizationResultCollection, TorsionDriveResultCollection]
        QCSubmit result collection containing the retrieved dataset data.
        The specific type depends on the dataset_type parameter.

    Raises
    ------
    KeyError
        If dataset_type is not one of the supported types.

    Notes
    -----
    This function requires an active internet connection to access the
    QCArchive portal at https://api.qcarchive.molssi.org:443.

    Examples
    --------
    >>> datasets = ["OpenFF Gen 2 Opt Set 1 Roche"]
    >>> collection = retrieve_datasets(datasets, "optimization")
    >>> print(f"Retrieved {len(collection.entries)} entries")
    """
    if dataset_type not in collection_types:
        raise KeyError(
            f"{dataset_type} is not a valid dataset type. Must be one of: {list(collection_types.keys())}"
        )
    logger.info(
        f"Fetching {dataset_type} datasets {dataset_names} from QCArchive with {collection_types[dataset_type]}"
    )

    client = qcportal.PortalClient(
        "https://api.qcarchive.molssi.org:443", cache_dir="."
    )
    result_collection = collection_types[dataset_type].from_server(
        client=client,
        datasets=dataset_names,
        spec_name=spec_name,
    )
    return result_collection


def process_result_collection(
    result_collection: Union[
        BasicResultCollection,
        OptimizationResultCollection,
        TorsionDriveResultCollection,
    ],
    data_file: Union[str, pathlib.Path],
) -> None:
    """Process QCArchive result collection into a structured dataset.

    Extracts molecular data (SMILES, coordinates, energies, forces) from the
    QCArchive result collection and creates a structured dataset using the descent
    library. For optimization and torsiondrive datasets, extracts data from
    minimum energy conformations. Converts units from atomic units to kcal/mol
    and Angstroms as appropriate.

    Parameters
    ----------
    result_collection : Union[BasicResultCollection, OptimizationResultCollection, TorsionDriveResultCollection]
        QCArchive result collection containing the molecular data. The collection
        type determines how data is extracted (e.g., final geometries for
        optimizations, grid points for torsiondrives).
    data_file : Union[str, pathlib.Path]
        Directory path where the processed dataset will be saved. The dataset
        will be saved in HuggingFace datasets format with Arrow files.

    Raises
    ------
    FileNotFoundError
        If the output directory cannot be created.
    KeyError
        If expected properties ('return_energy', 'scf total gradient') are
        missing from the result collection records.
    AttributeError
        If result collection records don't have expected attributes
        (e.g., 'minimum_optimizations' for torsiondrive data).

    Notes
    -----
    This function performs the following operations:
    - Groups molecular data by canonical SMILES representation
    - Converts coordinates from Bohr to Angstrom
    - Converts energies from Hartree to kcal/mol
    - Converts gradients to forces (kcal/mol/Angstrom)
    - Creates descent-compatible dataset entries
    - Saves dataset in HuggingFace format
    - Saves unique SMILES list as JSON file

    For torsiondrive datasets, only minimum energy optimizations are extracted.
    The function uses mapped SMILES to preserve atom ordering consistency.

    Examples
    --------
    >>> collection = retrieve_datasets(["dataset_name"], "torsiondrive")
    >>> process_result_collection(collection, "./processed_data")
    """

    logger.info("Processing collection...")
    data_by_smiles = defaultdict(list)
    records_and_molecules = list(result_collection.to_records())
    for record, _ in tqdm(records_and_molecules):  # lazily group by CMILES
        if isinstance(result_collection, BasicResultCollection):
            molecule = record.molecule
            mapped_smiles = (
                molecule.identifiers.canonical_isomeric_explicit_hydrogen_mapped_smiles
            )
            coords = molecule.geometry * BOHR_TO_ANGSTROM
            energy = record.properties["return_energy"] * HARTREE_TO_KCAL
            gradient = np.array(record.properties["scf_total_gradient"]).reshape(
                (-1, 3)
            )
            forces = (-gradient) * HARTREE_TO_KCAL / BOHR_TO_ANGSTROM
            entry = {
                "coords": coords,
                "energy": energy,
                "forces": forces,
            }
            data_by_smiles[mapped_smiles].append(entry)
        elif isinstance(result_collection, OptimizationResultCollection):
            last = record.trajectory[-1]
            last_mol = last.molecule
            mapped_smiles = (
                last_mol.identifiers.canonical_isomeric_explicit_hydrogen_mapped_smiles
            )
            coords = last_mol.geometry * BOHR_TO_ANGSTROM
            energy = last.properties["return_energy"] * HARTREE_TO_KCAL
            gradient = np.array(last.properties["scf total gradient"]).reshape((-1, 3))
            forces = (-gradient) * HARTREE_TO_KCAL / BOHR_TO_ANGSTROM
            entry = {
                "coords": coords,
                "energy": energy,
                "forces": forces,
            }
            data_by_smiles[mapped_smiles].append(entry)
        if isinstance(result_collection, TorsionDriveResultCollection):
            # take only the optimized grid points
            for opt in record.minimum_optimizations.values():
                last = opt.trajectory[-1]
                last_mol = last.molecule
                mapped_smiles = last_mol.identifiers.canonical_isomeric_explicit_hydrogen_mapped_smiles
                coords = last_mol.geometry * BOHR_TO_ANGSTROM
                energy = last.properties["return_energy"] * HARTREE_TO_KCAL
                gradient = np.array(last.properties["scf total gradient"]).reshape(
                    (-1, 3)
                )
                forces = (-gradient) * HARTREE_TO_KCAL / BOHR_TO_ANGSTROM
                entry = {
                    "coords": coords,
                    "energy": energy,
                    "forces": forces,
                }
                data_by_smiles[mapped_smiles].append(entry)

    descent_entries = []
    for mapped_smiles, entries in data_by_smiles.items():
        entry = {
            "smiles": mapped_smiles,
            "coords": torch.tensor(np.stack([x["coords"] for x in entries], axis=0)),
            "energy": torch.tensor(
                np.stack([np.atleast_1d(x["energy"]) for x in entries], axis=0)
            ),
            "forces": torch.tensor(np.stack([x["forces"] for x in entries], axis=0)),
        }
        descent_entries.append(entry)

    dataset = descent.targets.energy.create_dataset(entries=descent_entries)
    dataset.save_to_disk(data_file)
    unique_smiles = dataset.unique("smiles")
    logger.info(
        f"Found {len(dataset)} ({len(unique_smiles)} unique) SMILES in requested datasets"
    )
    path = os.path.split(data_file)[0]
    with open(os.path.join(path, "smiles.json"), "w") as file:
        json.dump(list(unique_smiles), file)


def main(
    datasets: List[str], dataset_type: str, data_file: Union[str, pathlib.Path]
) -> None:
    """Main processing function for QCArchive data workflow.

    Orchestrates the complete QCArchive data processing workflow by retrieving
    datasets from QCArchive and processing them into structured format for
    machine learning applications.

    Parameters
    ----------
    datasets : List[str]
        List of QCArchive dataset names to retrieve and process.
    dataset_type : str
        Type of datasets to retrieve. Must be one of: 'optimization',
        'singlepoint', or 'torsiondrive'.
    data_file : Union[str, pathlib.Path]
        Output path for the processed dataset in HuggingFace format.

    Returns
    -------
    None

    Raises
    ------
    KeyError
        If an invalid dataset_type is specified.
    ConnectionError
        If unable to connect to QCArchive portal.
    FileNotFoundError
        If the output directory cannot be created.

    Notes
    -----
    The function performs the following workflow:
    1. Connects to QCArchive portal and retrieves specified datasets
    2. Processes the result collection into descent-compatible format
    3. Saves the processed data in HuggingFace datasets format
    4. Creates a JSON file with unique SMILES strings

    This function is typically called from the command-line interface but
    can also be used programmatically when importing the module.

    Examples
    --------
    >>> # Programmatic usage:
    >>> datasets = ["OpenFF Gen 2 Opt Set 1 Roche"]
    >>> main(datasets, "optimization", "./qcarchive_data")

    >>> # Multiple datasets:
    >>> datasets = ["Dataset 1", "Dataset 2"]
    >>> main(datasets, "torsiondrive", "./processed_torsions")
    """

    result_collection = retrieve_datasets(datasets, dataset_type)
    process_result_collection(result_collection, data_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download and process QCArchive datasets for molecular ML workflows.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Process single optimization dataset
    python get_data_qca.py --datasets "OpenFF Gen 2 Opt Set 1 Roche" \\
                           --dataset_type optimization \\
                           --data_file ./qcarchive_data

    # Process multiple torsiondrive datasets
    python get_data_qca.py --datasets "OpenFF Gen 2 Torsion Set 1 Roche" \\
                                      "OpenFF Gen 2 Torsion Set 2 Coverage" \\
                           --dataset_type torsiondrive \\
                           --data_file ./torsion_data
        """,
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        required=True,
        help="List of dataset names",
    )
    parser.add_argument(
        "--dataset_type",
        type=str,
        required=True,
        help="Dataset type of all listed datasets",
    )
    parser.add_argument(
        "--data_file",
        type=str,
        required=True,
        help="Output HuggingFace formatted data",
    )
    args = parser.parse_args()

    main(args.datasets, args.dataset_type, args.data_file)
