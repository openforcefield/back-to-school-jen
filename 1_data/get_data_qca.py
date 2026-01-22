"""Functions to obtain and process data from QCArchive for the workflow.

This module provides functionality to download, process, and filter QCArchive datasets
for molecular property prediction workflows. It includes functions to:

1. Retrieve a dataset from QCArchive
2. Process the QCArchive dataset into a HuggingFace structured dataset

The module can be used as a command-line script or imported as a library.

Command-line Usage
------------------
python get_data_qca.py --datasets "Dataset Name" --dataset_type TYPE --data_file PATH

Command-line Arguments
----------------------
--datasets : list[str]
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

import argparse
import json
import pathlib
from typing import Union

import descent.targets.energy
import numpy as np
from loguru import logger
from openff.units import unit
from tqdm import tqdm

import qcportal

HARTREE_TO_KCAL: float = (1 * unit.hartree * unit.avogadro_constant).m_as(
    unit.kilocalories_per_mole
)
BOHR_TO_ANGSTROM: float = (1.0 * unit.bohr).m_as(unit.angstrom)


def retrieve_data(
    input_data: list[str],
    input_type: str,
    dataset_type: str,
    spec_name: str = "default",
) -> list[dict]:
    """Retrieve data from QCArchive based on dataset names or record IDs.

    Parameters
    ----------
    input_data : list[str]
        List of dataset names or record IDs to retrieve from QCArchive.
    input_type : str
        Type of input data: "dataset" or "record".
    dataset_type : str
        Type of dataset: "singlepoint" or "optimization".
    spec_name : str, optional
        Specification name for dataset retrieval, by default "default".

    Returns
    -------
    list[dict]
        List of records containing molecular data.

    Raises
    ------
    ValueError
        If input_type is not "dataset" or "record".
    ValueError
        If dataset_type is not "singlepoint" or "optimization".
    """
    if dataset_type not in ["singlepoint", "optimization"]:
        raise ValueError("dataset_type must be 'singlepoint' or 'optimization'")

    client = qcportal.PortalClient("https://api.qcarchive.molssi.org:443")

    if input_type == "dataset":
        logger.info(f"Fetching {dataset_type} datasets {input_data} from QCArchive")
        records = []
        for dataset_name in input_data:
            dataset = client.get_dataset(dataset_type, dataset_name)
            records.extend(
                [
                    rec
                    for _, _, rec in dataset.iterate_records(
                        specification_name=[spec_name]
                    )
                ]
            )
    elif input_type == "record":
        logger.info(f"Fetching records {input_data} from QCArchive")
        records = [*client.query_records(record_id=input_data)]
    else:
        raise ValueError("input_type must be 'dataset' or 'record'")

    return records


def process_records(
    records: list[dict],
    data_file: Union[str, pathlib.Path],
    dataset_type: str,
) -> None:
    """Process QCArchive records into a structured dataset.

    Parameters
    ----------
    records : list[dict]
        List of records containing molecular data.
    data_file : Union[str, pathlib.Path]
        Directory path where the processed dataset will be saved.
    dataset_type : str
        Type of dataset: "singlepoint" or "optimization".

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
    """
    logger.info("Processing records...")
    data_file = pathlib.Path(data_file)
    no_gradient = []
    all_data = []

    for record in tqdm(records):
        rec_id = record.id
        if dataset_type == "singlepoint":
            molecule = record.molecule
        elif dataset_type == "optimization":
            molecule = record.initial_molecule
            record = record.trajectory[-1]
        else:
            raise ValueError("dataset_type must be 'singlepoint' or 'optimization'")

        mapped_smiles = (
            molecule.identifiers.canonical_isomeric_explicit_hydrogen_mapped_smiles
        )
        coords = molecule.geometry * BOHR_TO_ANGSTROM
        energy = record.properties["return_energy"] * HARTREE_TO_KCAL
        if "scf_total_gradient" not in record.properties:
            no_gradient.append(rec_id)
            forces = np.ones_like(coords) * np.nan
        else:
            gradient = np.array(record.properties["scf_total_gradient"]).reshape(
                (-1, 3)
            )
            forces = (-gradient) * HARTREE_TO_KCAL / BOHR_TO_ANGSTROM
        all_data.append(
            {
                "smiles": mapped_smiles,
                "coords": [coords],
                "energy": [energy],
                "forces": [forces],
            }
        )

    logger.warning(
        f"There are {len(no_gradient)} records without gradients. Record IDs: {no_gradient}"
    )

    dataset = descent.targets.energy.create_dataset(all_data)
    logger.info(f"Saving HuggingFace dataset to: {data_file.resolve()}")
    dataset.save_to_disk(data_file)
    unique_smiles = dataset.unique("smiles")
    logger.info(
        f"Found {len(dataset)} ({len(unique_smiles)} unique) SMILES in requested records"
    )

    filename = data_file / "smiles.json"
    logger.info(f"Saving SMILES strings to: {filename}")
    with open(filename, "w") as file:
        json.dump(list(unique_smiles), file)


def load_record_ids(input_data: list[str]) -> list[str]:
    """Load record IDs from a list or a text file.

    Parameters
    ----------
    input_data : list[str]
        List of record IDs or a single text file containing record IDs.

    Returns
    -------
    list[str]
        List of record IDs.

    Raises
    ------
    FileNotFoundError
        If the specified file does not exist.
    ValueError
        If the file format is invalid.
    """
    if len(input_data) == 1 and input_data[0].endswith(".txt"):
        file_path = pathlib.Path(input_data[0])
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        with open(file_path, "r") as f:
            lines = f.readlines()

        # Extract record IDs, ignoring comments
        record_ids = [
            line.strip() for line in lines if not line.startswith("#") and line.strip()
        ]
        if not record_ids:
            raise ValueError(f"No valid record IDs found in file: {file_path}")

        logger.info(f"Loaded {len(record_ids)} record IDs from file: {file_path}")
        return record_ids

    # Assume input_data is already a list of record IDs
    logger.info(f"Using {len(input_data)} record IDs provided directly")
    return input_data


def main(
    input_data: list[str],
    input_type: str,
    dataset_type: str,
    data_file: Union[str, pathlib.Path],
) -> None:
    """Main processing function for QCArchive data workflow.

    Orchestrates the complete QCArchive data processing workflow by retrieving
    data from QCArchive and processing it into structured format for
    fitting applications.

    Parameters
    ----------
    input_data : list[str]
        List of dataset names, record IDs, or a text file containing record IDs.
    input_type : str
        Type of input data: "dataset" or "record".
    dataset_type : str
        Type of dataset: "singlepoint" or "optimization".
    data_file : Union[str, pathlib.Path]
        Output path for the processed dataset in HuggingFace format.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If input_type is not "dataset" or "record".
    """
    if input_type == "record":
        input_data = load_record_ids(input_data)
    print(input_data)
    records = retrieve_data(input_data, input_type, dataset_type)
    process_records(records, data_file, dataset_type)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download and process QCArchive data for molecular fitting workflows.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Process singlepoint datasets
    python get_data_qca.py --input_data "OpenFF Gen 2 Opt Set 1 Roche" \
                           --input_type dataset \
                           --dataset_type singlepoint \
                           --data_file ./qcarchive_data

    # Process optimization datasets
    python get_data_qca.py --input_data "OpenFF Gen 2 Opt Set 1 Roche" \
                           --input_type dataset \
                           --dataset_type optimization \
                           --data_file ./qcarchive_data

    # Process records from a text file
    python get_data_qca.py --input_data record_ids.txt \
                           --input_type record \
                           --dataset_type singlepoint \
                           --data_file ./record_data
        """,
    )
    parser.add_argument(
        "--input_data",
        type=str,
        nargs="+",
        required=True,
        help="List of dataset names, record IDs, or a text file containing record IDs",
    )
    parser.add_argument(
        "--input_type",
        type=str,
        required=True,
        choices=["dataset", "record"],
        help="Type of input data: 'dataset' or 'record'",
    )
    parser.add_argument(
        "--dataset_type",
        type=str,
        required=True,
        choices=["singlepoint", "optimization"],
        help="Type of dataset: 'singlepoint' or 'optimization'",
    )
    parser.add_argument(
        "--data_file",
        type=str,
        required=True,
        help="Output HuggingFace formatted data",
    )
    args = parser.parse_args()

    main(args.input_data, args.input_type, args.dataset_type, args.data_file)
