"""Functions to obtain and process SPICE2 data for the workflow.

This module provides functionality to download, process, and filter SPICE2 dataset
for molecular property prediction workflows. It includes functions to:

1. Download SPICE2 HDF5 data from Zenodo
2. Process the raw HDF5 data into a structured dataset

Based on: https://github.com/fjclark/descent-workflow/blob/main/workflow/get_data.py

Outputs
-------
data-raw : dir
├── dataset_info.json          # Dataset metadata and schema
├── state.json                 # Dataset state information
└── data-00000-of-00001.arrow  # Actual data in Apache Arrow format
    ├── smiles (str)
    ├── coords (list) # flattened, reshape with np.array(coords_list).reshape(-1, 3)
    ├── energy (list) # flattened
    └── forces (list) # flattened

Constants:
    HARTREE_TO_KCAL: Conversion factor from Hartree to kcal/mol
    BOHR_TO_ANGSTROM: Conversion factor from Bohr to Angstrom
    SPICE2_SOURCES: Set of SPICE2 dataset sources to include (excludes B/Si sets)
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
from typing import Any, Dict, List, Set

import descent.targets.energy
import h5py
import numpy as np
import numpy.typing as npt
from loguru import logger
from openff.units import unit
from tqdm import tqdm


HARTREE_TO_KCAL: float = (1 * unit.hartree * unit.avogadro_constant).m_as(
    unit.kilocalories_per_mole
)
BOHR_TO_ANGSTROM: float = (1.0 * unit.bohr).m_as(unit.angstrom)

# This avoids Boron and Silicon as they're in 'SPICE PubChem Boron Silicon v1.0'
SPICE2_SOURCES: Set[str] = {
    "SPICE DES Monomers Single Points Dataset v1.1",
    "SPICE Dipeptides Single Points Dataset v1.3",
    "SPICE PubChem Set 1 Single Points Dataset v1.3",
    "SPICE PubChem Set 2 Single Points Dataset v1.3",
    "SPICE PubChem Set 3 Single Points Dataset v1.3",
    "SPICE PubChem Set 4 Single Points Dataset v1.3",
    "SPICE PubChem Set 5 Single Points Dataset v1.3",
    "SPICE PubChem Set 6 Single Points Dataset v1.3",
    "SPICE PubChem Set 7 Single Points Dataset v1.0",
    "SPICE PubChem Set 8 Single Points Dataset v1.0",
    "SPICE PubChem Set 9 Single Points Dataset v1.0",
    "SPICE PubChem Set 10 Single Points Dataset v1.0",
}


def download_spice2_data(data_dir: pathlib.Path) -> None:
    """Download the SPICE2 dataset from Zenodo if not already present.

    Downloads the SPICE-2.0.1.hdf5 file from Zenodo to the specified directory.
    If the file already exists, the download is skipped.

    Parameters
    ----------
    data_dir : pathlib.Path
        Path to the directory where the SPICE data should be downloaded.
        The directory will be created if it doesn't exist.

    Raises
    ------
    subprocess.CalledProcessError
        If the download command fails.

    Notes
    -----
    The download is approximately 42GB and may take significant time depending
    on network speed.
    """
    logger.info("Downloading SPICE data from Zenodo. This may take a while...")

    output_file = data_dir / "SPICE-2.0.1.hdf5"
    if output_file.exists():
        logger.info(f"SPICE data already exists at {output_file}. Skipping download.")
        return

    cmds = [
        f"mkdir -p {data_dir}",
        f" wget -O {output_file} https://zenodo.org/record/10975225/files/SPICE-2.0.1.hdf5?download=1",
    ]

    for cmd in cmds:
        subprocess.run(
            cmd,
            check=True,
            shell=True,
        )


def process_dataset_spice2(data_dir: pathlib.Path) -> None:
    """Process the raw SPICE2 HDF5 data into a structured dataset.

    Extracts molecular data (SMILES, coordinates, energies, forces) from the
    SPICE-2.0.1.hdf5 file and creates a structured dataset using the descent
    library. Only includes data from specified SPICE2 sources, excluding
    Boron and Silicon datasets.

    Parameters
    ----------
    data_dir : pathlib.Path
        Path to the directory containing the SPICE-2.0.1.hdf5 file.
        The processed dataset will be saved to data_dir/data-raw.

    Raises
    ------
    FileNotFoundError
        If the SPICE-2.0.1.hdf5 file is not found.
    KeyError
        If expected keys are missing from the HDF5 data.

    Notes
    -----
    This function has the following side effects:
    - Creates data_dir/data-raw directory
    - Saves processed dataset to data_dir/data-raw
    - Saves unique SMILES list to data_dir/data-raw/smiles.json
    """

    output_dir = data_dir / "data-raw"

    with h5py.File(data_dir / "SPICE-2.0.1.hdf5") as spice:
        all_data: List[Dict[str, Any]] = []
        all_smiles: Set[str] = set()

        for record in tqdm(spice.values(), desc="Extracting dataset", ncols=80):
            smiles: str = record["smiles"].asstr()[0]
            subset: str = record["subset"].asstr()[0]

            # Only extract the data if it's of the desired type
            if subset not in SPICE2_SOURCES:
                continue

            # extract the data
            all_smiles.add(smiles)
            n_conformers: int = record["conformations"].shape[0]
            assert len(record["dft_total_energy"]) == n_conformers

            energies: List[float] = [
                record["dft_total_energy"][i] * HARTREE_TO_KCAL
                for i in range(n_conformers)
            ]
            coords: List[npt.NDArray[np.float64]] = [
                record["conformations"][i] * BOHR_TO_ANGSTROM
                for i in range(n_conformers)
            ]
            forces: List[npt.NDArray[np.float64]] = [
                record["dft_total_gradient"][i]
                * -1
                * (HARTREE_TO_KCAL / BOHR_TO_ANGSTROM)
                for i in range(n_conformers)
            ]
            all_data.append(
                {
                    "smiles": smiles,
                    "coords": coords,
                    "energy": energies,
                    "forces": forces,
                }
            )

        dataset = descent.targets.energy.create_dataset(all_data)
        dataset.save_to_disk(output_dir)
        unique_smiles = dataset.unique("smiles")
        logger.info(
            f"Found {len(dataset)} ({len(unique_smiles)} unique) SMILES in SPICE2"
        )
        with open(output_dir / "smiles.json", "w") as file:
            json.dump(list(unique_smiles), file)


def main() -> None:
    """Command-line interface for the SPICE2 data processing workflow.

    Parses command-line arguments and runs the complete data processing
    workflow. The only required argument is --data-dir specifying where
    to store the downloaded and processed data.

    This is the main workflow function that orchestrates the entire data
    preparation process including download, processing, and filtering.

    Parameters
    ----------
    data_dir : Union[pathlib.Path, str]
        Path to the directory where SPICE2 data should be stored.
        Can be either a pathlib.Path or string path.

    Notes
    -----
    Command-line Arguments:
        --data-dir : str
            Directory path for SPICE2 data storage and processing.
    This function has the following side effects:
    - Downloads SPICE-2.0.1.hdf5 if not present
    - Creates processed dataset in data_dir/data-raw
    - Creates filtered dataset in data_dir/data-filtered-by-forces
    - Generates visualization and metadata files

    Examples
    --------
    >>> python get_data.py --data-dir ./spice_data
    >>> python get_data.py --data-dir /path/to/data/directory
    """
    parser = argparse.ArgumentParser(
        description="Download and process SPICE2 data for molecular ML workflows.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python get_data.py --data-dir ./spice_data
    python get_data.py --data-dir /path/to/data/directory
        """,
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Directory to download and process SPICE2 data. "
        "Will be created if it doesn't exist.",
    )
    args = parser.parse_args()

    data_dir = pathlib.Path(args.data_dir)
    logger.info("Getting data for SPICE...")
    download_spice2_data(data_dir)
    process_dataset_spice2(data_dir)
    logger.info("Done getting data for SPICE.")


if __name__ == "__main__":
    main()
