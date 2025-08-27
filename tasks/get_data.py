"""Functions to obtain and process SPICE2 data for the workflow based on:
https://github.com/fjclark/descent-workflow/blob/main/workflow/get_data.py


"""

import json
import pathlib
import typing
import subprocess
import argparse

from loguru import logger
import h5py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import datasets
import descent.targets.energy
from openff.units import unit


HARTREE_TO_KCAL = (1 * unit.hartree * unit.avogadro_constant).m_as(unit.kilocalories_per_mole)
BOHR_TO_ANGSTROM = (1.0 * unit.bohr).m_as(unit.angstrom)

# This avoids Boron and Silicon as they're in 'SPICE PubChem Boron Silicon v1.0',
SPICE2_SOURCES = {
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
    """Download the SPICE data from the Zenodo."""
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
    """Process the SPICE dataset and save it to disk (without filtering forces)."""

    output_dir = data_dir / "data-raw"

    with h5py.File(data_dir / "SPICE-2.0.1.hdf5") as spice:
        all_data = []
        all_smiles = set()

        for record in tqdm(spice.values(), desc="Extracting dataset", ncols=80):
            smiles = record["smiles"].asstr()[0]
            subset = record["subset"].asstr()[0]

            # Only extract the data if it's of the desired type
            if subset not in SPICE2_SOURCES:
                continue

            # extract the data
            all_smiles.add(smiles)
            n_conformers = record["conformations"].shape[0]
            assert len(record["dft_total_energy"]) == n_conformers
            energies = [
                record["dft_total_energy"][i] * HARTREE_TO_KCAL
                for i in range(n_conformers)
            ]
            coords = [
                record["conformations"][i] * BOHR_TO_ANGSTROM
                for i in range(n_conformers)
            ]
            forces = [
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


def filter_spice2_dataset_by_forces(data_dir: pathlib.Path) -> None:
    """Filter the SPICE dataset by forces and save it to disk."""

    logger.info("Filtering SPICE dataset by forces...")

    input_dir = data_dir / "data-raw"
    output_dir = data_dir / "data-filtered-by-forces"

    dataset = datasets.load_from_disk(input_dir)
    data_df = dataset.to_pandas()

    def get_rms(array: np.ndarray) -> float:
        return np.sqrt(np.mean(array**2))

    data_df["rms_forces"] = data_df["forces"].apply(lambda x: get_rms(np.array(x)))

    # Plot the distribution of the RMS forces
    # Get the percentiles in increments of 5
    percentile_intervals = np.array([85, 90, 95, 97.5, 99])
    percentile_values = np.percentile(data_df["rms_forces"], percentile_intervals)

    # Create a dict of the percentiles
    percentile_dict = {
        interval: value
        for interval, value in zip(percentile_intervals, percentile_values, strict=True)
    }
    logger.info(f"Percentiles: {percentile_dict}")

    # Plot boxplot of the rmse forces
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(x=data_df["rms_forces"], ax=ax)

    for interval, value in percentile_dict.items():
        # Add a vertical line at the percentile
        ax.axvline(x=value, color="red", linestyle="--", alpha=0.5)
        # Write the percentile value
        ax.text(value, 0.4, f"{interval:.2f}", color="red", rotation=90, va="center")

    ax.set_xlabel(r"RMS Forces (kcal mol$^{-1}$ $\mathrm{\AA}^{-1})$")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of RMS Forces")
    fig.savefig(str(output_dir / "rms_forces.png"), dpi=300, bbox_inches="tight")

    # Get the data above the 95th percentile
    df_highest_95 = data_df[data_df["rms_forces"] > percentile_dict[95]]
    logger.info(f"Cutoff: {percentile_dict[95]:.2f} kcal/(mol Angstrom)")
    high_force_smiles = df_highest_95["smiles"].tolist()
    with open(output_dir / "high_force_smiles.json", "w") as file:
        json.dump(high_force_smiles, file)
    logger.info(f"Removed {len(df_highest_95)} entries with high forces")

    # Save a filtered dataset without the high forces
    filtered_dataset = dataset.filter(lambda x: x["smiles"] not in high_force_smiles)
    filtered_dataset.save_to_disk(output_dir)
    logger.info(
        f"Filtered dataset (containing {len(filtered_dataset)} entries) saved to {output_dir}"
    )

    # Save all of the smiles to a json file
    with open(output_dir / "smiles.json", "w") as file:
        json.dump(list(filtered_dataset.unique("smiles")), file)


def get_data_spice2_force_filtered(data_dir: pathlib.Path | str) -> None:
    data_dir = pathlib.Path(data_dir)
    logger.info("Getting data for SPICE...")
    download_spice2_data(data_dir)
    process_dataset_spice2(data_dir)
    filter_spice2_dataset_by_forces(data_dir)
    logger.info("Done getting data for SPICE.")


def main():
    parser = argparse.ArgumentParser(description="Download and process ESPALOMA data.")
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Directory to download and process SPICE2 data.",
    )
    args = parser.parse_args()

    get_data_spice2_force_filtered(args.data_dir)


if __name__ == "__main__":
    main()