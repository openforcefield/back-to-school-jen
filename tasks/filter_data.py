"""Filter HuggingFace datasets by removing high force entries.

This module provides functions to filter molecular datasets by removing entries
with high RMS forces using either percentile-based or Z-score-based methods.
The filtering helps improve dataset quality by removing potentially problematic
high-force conformations.

Command-line Arguments
----------------------
--data-dir : str
    Path to the directory containing HuggingFace formatted molecular dataset.
    This should contain dataset_info.json, state.json, and .arrow files.

Examples
--------
Filter a dataset using Z-score method (removes entries with Z-score > 4):
$ python filter_data.py --data-dir ./data-raw

Filter using percentile method (removes top 5% by RMS forces):
$ python filter_data.py --data-dir ./data-raw
# (requires uncommenting filter_dataset_by_forces_95_percentile in main)

Input Dataset Structure
-----------------------
data-dir/ : dir
├── dataset_info.json          # Dataset metadata and schema
├── state.json                 # Dataset state information
└── data-00000-of-00001.arrow  # Actual data in Apache Arrow format
    ├── smiles (str)           # SMILES molecular representation
    ├── coords (list)          # Flattened 3D coordinates
    ├── energy (list)          # Flattened energy values
    └── forces (list)          # Flattened force vectors

Output Structure
----------------
Creates filtered datasets in new directories with suffixes:
- data-dir_filtered-z-score/     # Z-score filtered dataset (main data)
- data-dir_filtered-95thpercentile/  # Percentile filtered dataset (main data)

Creates auxiliary files in the filtered_results directory:
- filtered_results/[dataset_name]_filtered_z_score/
- filtered_results/[dataset_name]_filtered_95thpercentile/

Each filtered dataset directory contains:
├── dataset_info.json          # Updated dataset metadata
├── state.json                 # Updated dataset state
└── data-00000-of-00001.arrow  # Filtered data in Apache Arrow format

Each filtered_results subdirectory contains:
├── rms_forces.png             # Force distribution visualization
├── high_force_smiles.json     # List of removed high-force SMILES
└── smiles.json                # List of remaining SMILES after filtering
"""

from __future__ import annotations

import json
import pathlib
from typing import Dict, List

import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import numpy.typing as npt
from loguru import logger

import datasets


def get_rms(array: npt.NDArray[np.float64]) -> float:
    """Calculate root mean square of a numpy array.
    Parameters
    ----------
    array : npt.NDArray[np.float64]
        Input array for RMS calculation.
    Returns
    -------
    float
        Root mean square value of the array elements.
    """
    return np.sqrt(np.mean(array**2))


def filter_dataset_by_forces_95_percentile(input_dir: pathlib.Path) -> None:
    """Filter the result collection by removing entries with high RMS forces using percentiles.

    Loads the processed dataset, calculates RMS forces for each entry, and removes
    entries above the 95th percentile. Creates visualizations of the force
    distribution and saves the filtered dataset.

    Taken from https://github.com/fjclark/descent-workflow/blob/main/workflow/get_data.py

    Parameters
    ----------
    input_dir : pathlib.Path
        Path to the directory containing HuggingFace formatted data.
        Filtered data will be saved to input_dir_filtered-95thpercentile.

    Raises
    ------
    FileNotFoundError
        If the input dataset directory doesn't exist.

    Notes
    -----
    This function has the following side effects:
    - Creates input_dir_filtered-95thpercentile directory for filtered dataset
    - Saves filtered dataset to input_dir_filtered-95thpercentile
    - Creates filtered_results/[dataset_name]_filtered_95thpercentile directory
    - Saves force distribution plot as filtered_results/[dataset_name]_filtered_95thpercentile/rms_forces.png
    - Saves high force SMILES list as filtered_results/[dataset_name]_filtered_95thpercentile/high_force_smiles.json
    - Saves filtered SMILES list as filtered_results/[dataset_name]_filtered_95thpercentile/smiles.json
    """

    logger.info("Filtering dataset by forces below 95th percentile...")

    output_dir = input_dir.parent / (input_dir.name + "_filtered-95thpercentile")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create filtered_results directory for auxiliary files
    results_dir = (
        input_dir.parent.parent
        / "filtered_results"
        / (input_dir.name.replace("-", "_") + "_filtered_95thpercentile")
    )
    results_dir.mkdir(parents=True, exist_ok=True)

    dataset = datasets.load_from_disk(input_dir)
    data_df = dataset.to_pandas()

    data_df["rms_forces"] = data_df["forces"].apply(lambda x: get_rms(np.array(x)))

    # Plot the distribution of the RMS forces
    percentile_intervals: npt.NDArray[np.float64] = np.array([85, 95, 97.5, 99])
    percentile_values: npt.NDArray[np.float64] = np.percentile(
        data_df["rms_forces"], percentile_intervals
    )

    # Create a dict of the percentiles
    percentile_dict: Dict[float, float] = {
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
        ax.text(
            value,
            0.4,
            f"Percentile:{interval:.2f}={value:.3e}",
            color="red",
            rotation=90,
            va="center",
        )

    ax.set_xlabel(r"RMS Forces (kcal mol$^{-1}$ $\mathrm{\AA}^{-1})$")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of RMS Forces")
    fig.savefig(str(results_dir / "rms_forces.png"), dpi=300, bbox_inches="tight")

    # Get the data above the 95th percentile
    df_highest_95 = data_df[data_df["rms_forces"] > percentile_dict[95]]
    logger.info(f"Cutoff: {percentile_dict[95]:.2f} kcal/(mol Angstrom)")
    high_force_smiles: List[str] = df_highest_95["smiles"].tolist()
    with open(results_dir / "high_force_smiles.json", "w") as file:
        json.dump(high_force_smiles, file)
    logger.info(f"Removed {len(df_highest_95)} entries with high forces")

    # Save a filtered dataset without the high forces
    filtered_dataset = dataset.filter(lambda x: x["smiles"] not in high_force_smiles)
    filtered_dataset.save_to_disk(output_dir)
    logger.info(
        f"Filtered dataset (containing {len(filtered_dataset)} entries) saved to {output_dir}"
    )

    # Save all of the smiles to a json file in filtered_results
    with open(results_dir / "smiles.json", "w") as file:
        json.dump(list(filtered_dataset.unique("smiles")), file)
    logger.info(f"Auxiliary files saved to {results_dir}")


def filter_dataset_by_forces_z_score(input_dir: pathlib.Path) -> None:
    """Filter the result collection by removing entries with high RMS forces using Z-score.

    Loads the processed dataset, calculates RMS forces for each entry and subsequently the
    Z-score. Entries above a Z-score of 4 are removed.
    A visualization of the force distribution for different Z-scores is saved with the filtered dataset.

    Parameters
    ----------
    input_dir : pathlib.Path
        Path to the directory containing HuggingFace formatted data.
        Filtered data will be saved to input_dir_filtered-z-score.

    Raises
    ------
    FileNotFoundError
        If the input dataset directory doesn't exist.

    Notes
    -----
    This function has the following side effects:
    - Creates input_dir_filtered-z-score directory for filtered dataset
    - Saves filtered dataset to input_dir_filtered-z-score
    - Creates filtered_results/[dataset_name]_filtered_z_score directory
    - Saves force distribution plot as filtered_results/[dataset_name]_filtered_z_score/rms_forces.png
    - Saves high force SMILES list as filtered_results/[dataset_name]_filtered_z_score/high_force_smiles.json
    - Saves filtered SMILES list as filtered_results/[dataset_name]_filtered_z_score/smiles.json
    """

    logger.info("Filtering dataset by forces...")

    output_dir = input_dir.parent / (input_dir.name + "_filtered-z-score")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create filtered_results directory for auxiliary files
    results_dir = (
        input_dir.parent.parent
        / "filtered_results"
        / (input_dir.name.replace("-", "_") + "_filtered_z_score")
    )
    results_dir.mkdir(parents=True, exist_ok=True)

    dataset = datasets.load_from_disk(input_dir)
    data_df = dataset.to_pandas()

    data_df["rms_forces"] = data_df["forces"].apply(lambda x: get_rms(np.array(x)))
    rms_mean = data_df["rms_forces"].mean()
    rms_std = data_df["rms_forces"].std()
    data_df["z-score"] = (data_df["rms_forces"] - rms_mean) / rms_std

    # Create a dict of the Z-scores
    z_score_dict: Dict[float, float] = {
        score: score * rms_std + rms_mean for score in [1, 2, 3, 4, 5, 6]
    }

    # Plot boxplot of the rmse forces
    # Plot boxplot of the rmse forces
    fig, ax = plt.subplots(figsize=(10, 5))
    # Plot histogram of RMS forces overlaid with boxplot
    sns.histplot(
        data_df["rms_forces"],
        ax=ax,
        color="lightgray",
        bins=100,
        kde=False,
        alpha=0.5,
        element="step",
        stat="count",
    )
    # Overlay boxplot (as a horizontal boxplot above the histogram)
    ax_box = ax.inset_axes([0, 1.05, 1, 0.15], sharex=ax)
    sns.boxplot(
        x=data_df["rms_forces"], ax=ax_box, color="skyblue", fliersize=2, linewidth=1
    )
    ax_box.axis("off")

    for interval, value in z_score_dict.items():
        # Add a vertical line at the percentile
        ax.axvline(x=value, color="red", linestyle="--", alpha=0.5)
        # Write the percentile value
        ax.text(
            value,
            0.4,
            f"Z-score:{interval:.2f}={value:.3e}",
            color="red",
            rotation=90,
            va="center",
        )

    ax.set_xlabel(r"RMS Forces (kcal mol$^{-1}$ $\mathrm{\AA}^{-1})$")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of RMS Forces")
    fig.savefig(str(results_dir / "rms_forces.png"), dpi=300, bbox_inches="tight")

    # Get the data above the 95th percentile
    cap_z_score = 4
    df_cap_z_score = data_df[data_df["rms_forces"] > z_score_dict[cap_z_score]]
    logger.info(f"Cutoff: {z_score_dict[cap_z_score]:.2f} kcal/(mol Angstrom)")
    high_force_smiles: List[str] = df_cap_z_score["smiles"].tolist()
    with open(results_dir / "high_force_smiles.json", "w") as file:
        json.dump(high_force_smiles, file, indent=4)
    logger.info(f"Removed {len(df_cap_z_score)} entries with high forces")

    # Save a filtered dataset without the high forces
    filtered_dataset = dataset.filter(lambda x: x["smiles"] not in high_force_smiles)
    filtered_dataset.save_to_disk(output_dir)
    logger.info(
        f"Filtered dataset (containing {len(filtered_dataset)} entries) saved to {output_dir}"
    )

    # Save all of the smiles to a json file in filtered_results
    with open(results_dir / "smiles.json", "w") as file:
        json.dump(list(filtered_dataset.unique("smiles")), file, indent=4)
    logger.info(f"Auxiliary files saved to {results_dir}")


def main(data_dir: pathlib.Path) -> None:
    """Main processing function for filtering molecular datasets by forces.

    Orchestrates the filtering workflow by applying Z-score-based filtering
    to remove entries with high RMS forces from molecular datasets.

    Parameters
    ----------
    data_dir : pathlib.Path
        Path to the directory containing HuggingFace formatted molecular dataset.
        The directory should contain dataset_info.json, state.json, and .arrow files.

    Returns
    -------
    None

    Notes
    -----
    This function performs the following workflow:
    1. Applies Z-score filtering to remove entries with Z-score > 4 for RMS forces
    2. Saves the filtered dataset to a new directory with suffix "_filtered-z-score"
    3. Generates visualization of force distributions
    4. Saves lists of removed and remaining SMILES

    The percentile-based filtering function is commented out but can be enabled
    by uncommenting the filter_dataset_by_forces_95_percentile call.

    This function is typically called from the command-line interface but
    can also be used programmatically when importing the module.

    Examples
    --------
    >>> # Programmatic usage:
    >>> data_dir = pathlib.Path("./data-raw")
    >>> main(data_dir)

    >>> # With string path:
    >>> main(pathlib.Path("/path/to/molecular/dataset"))
    """
    data_dir = pathlib.Path(data_dir)
    filter_dataset_by_forces_95_percentile(data_dir)
    filter_dataset_by_forces_z_score(data_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Filter HuggingFace structured dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python filter.py --data-dir /path/to/data/directory
        """,
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Directory to HuggingFace structured data",
    )
    args = parser.parse_args()
    main(args.data_dir)
