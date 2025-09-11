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
--percentile-cutoff : float, optional
    Percentile cutoff for filtering high force entries (default: 95.0).
    Entries above this percentile will be removed from the dataset.
--z-score-cutoff : float, optional
    Z-score cutoff for filtering high force entries (default: 4.0).
    Entries above this Z-score will be removed from the dataset.

Examples
--------
Filter a dataset using default settings (Z-score > 4 and percentile > 95):
$ cd /desired/output/directory
$ python path/to/tasks/filter_data.py --data-dir /path/to/data-raw

Filter using percentile method with custom 90th percentile cutoff:
$ cd /desired/output/directory
$ python path/to/tasks/filter_data.py --data-dir /path/to/data-raw --percentile-cutoff 90

Filter using Z-score method with custom cutoff of 3:
$ cd /desired/output/directory
$ python path/to/tasks/filter_data.py --data-dir /path/to/data-raw --z-score-cutoff 3

Filter using both custom percentile and Z-score cutoffs:
$ cd /desired/output/directory
$ python path/to/tasks/filter_data.py --data-dir /path/to/data-raw --percentile-cutoff 90 --z-score-cutoff 3

Output will be created in the current working directory regardless of input location.

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
Creates filtered datasets in the current working directory with names based on input directory:
- [input_dir_name]-z-score/     # Z-score filtered dataset (complete HuggingFace dataset)
- [input_dir_name]-{percentile_cutoff}thpercentile/  # Percentile filtered dataset (complete HuggingFace dataset)

Each filtered dataset directory contains:
├── dataset_info.json          # Updated dataset metadata
├── state.json                 # Updated dataset state
├── data-00000-of-00001.arrow  # Filtered data in Apache Arrow format
├── rms_forces.png             # Force distribution visualization
├── high_force_smiles.json     # List of removed high-force SMILES
└── smiles.json                # List of remaining SMILES after filtering
"""

import json
import pathlib

import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import numpy.typing as npt
from loguru import logger

import descent.targets.energy
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


def filter_nonparametrizable_molecules(
    dataset: datasets.Dataset | datasets.DatasetDict
) -> datasets.Dataset:
    """Remove non-parametrizable SMILES string from dataset.

    Filters the dataset to retain only molecules with valid SMILES capable
    of being parametrized by the force field, removing any molecules that
    would cause errors during force field parameter assignment.

    Parameters
    ----------
    dataset : datasets.Dataset | datasets.DatasetDict
        Input HuggingFace dataset containing molecular data. Must be a Dataset
        (not DatasetDict) for compatibility with descent.targets.energy.extract_smiles.

    Returns
    -------
    datasets.Dataset
        Filtered dataset containing only parametrizable molecules.

    Raises
    ------
    TypeError
        If dataset is a DatasetDict instead of Dataset.

    Notes
    -----
    Uses descent.targets.energy.extract_smiles to identify which molecules
    can be successfully parametrized by the force field. This function requires
    a datasets.Dataset object as input.
    """
    # Ensure we have a Dataset object, not DatasetDict
    if isinstance(dataset, datasets.DatasetDict):
        raise TypeError("Function requires datasets.Dataset, not DatasetDict")

    # Ensure molecules are parametrizable
    dataset_size = len(dataset)
    unique_smiles = descent.targets.energy.extract_smiles(dataset)
    dataset = dataset.filter(lambda d: d["smiles"] in unique_smiles)
    logger.info(
        f"Removed non-parametrizable molecules, dataset size change: {dataset_size} -> {len(dataset)}"
    )

    return dataset


def filter_dataset_by_forces_percentile(
    input_dir: pathlib.Path, percentile_cutoff: float = 95.0
) -> None:
    """Filter the result collection by removing entries with high RMS forces using percentiles.

    Loads the processed dataset, calculates RMS forces for each entry, and removes
    entries above the specified percentile. Creates visualizations of the force
    distribution and saves the filtered dataset.

    Taken from https://github.com/fjclark/descent-workflow/blob/main/workflow/get_data.py

    Parameters
    ----------
    input_dir : pathlib.Path
        Path to the directory containing HuggingFace formatted data.
        Filtered data will be saved to current working directory with name [input_dir.name]-{percentile_cutoff}thpercentile.
    percentile_cutoff : float, optional
        The percentile cutoff for filtering (default is 95.0).
        Entries above this percentile will be removed.

    Raises
    ------
    FileNotFoundError
        If the input dataset directory doesn't exist.

    Notes
    -----
    This function has the following side effects:
    - Creates [input_dir.name]-{percentile_cutoff}thpercentile directory in current working directory containing filtered dataset
    - Saves filtered dataset HuggingFace files to [input_dir.name]-{percentile_cutoff}thpercentile directory
    - Saves force distribution plot as [input_dir.name]-{percentile_cutoff}thpercentile/rms_forces.png
    - Saves high force SMILES list as [input_dir.name]-{percentile_cutoff}thpercentile/high_force_smiles.json
    - Saves filtered SMILES list as [input_dir.name]-{percentile_cutoff}thpercentile/smiles.json
    """

    logger.info(
        f"Filtering dataset by forces below {percentile_cutoff}th percentile..."
    )

    # Create output directory in current working directory with same name as input + suffix
    output_dir = pathlib.Path.cwd() / (
        input_dir.name + f"-{percentile_cutoff}thpercentile"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = datasets.load_from_disk(str(input_dir))
    dataset = filter_nonparametrizable_molecules(dataset)
    data_df = dataset.to_pandas()

    data_df["rms_forces"] = data_df["forces"].apply(lambda x: get_rms(np.array(x)))

    # Plot the distribution of the RMS forces
    percentile_intervals: npt.NDArray[np.float64] = np.array(
        [percentile_cutoff - 5, percentile_cutoff, percentile_cutoff + 5]
    )
    percentile_values: npt.NDArray[np.float64] = np.percentile(
        data_df["rms_forces"], percentile_intervals
    )

    # Create a dict of the percentiles
    percentile_dict: dict[float, float] = {
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
    fig.savefig(str(output_dir / "rms_forces.png"), dpi=300, bbox_inches="tight")

    # Get the data above the specified percentile
    df_highest_percentile = data_df[
        data_df["rms_forces"] > percentile_dict[percentile_cutoff]
    ]
    logger.info(f"Cutoff: {percentile_dict[percentile_cutoff]:.2f} kcal/(mol Angstrom)")
    high_force_smiles: list[str] = df_highest_percentile["smiles"].tolist()
    with open(output_dir / "high_force_smiles.json", "w") as file:
        json.dump(high_force_smiles, file)
    logger.info(f"Removed {len(df_highest_percentile)} entries with high forces")

    # Save a filtered dataset without the high forces
    filtered_dataset = dataset.filter(lambda x: x["smiles"] not in high_force_smiles)
    filtered_dataset.save_to_disk(output_dir)
    logger.info(
        f"Filtered dataset (containing {len(filtered_dataset)} entries) saved to {output_dir}"
    )

    # Save all of the smiles to a json file in the output directory
    with open(output_dir / "smiles.json", "w") as file:
        json.dump(list(filtered_dataset.unique("smiles")), file)
    logger.info(f"Auxiliary files saved to {output_dir}")


def filter_dataset_by_forces_z_score(
    input_dir: pathlib.Path, z_score_cutoff: float = 4.0
) -> None:
    """Filter the result collection by removing entries with high RMS forces using Z-score.

    Loads the processed dataset, calculates RMS forces for each entry and subsequently the
    Z-score. Entries above the specified Z-score cutoff are removed.
    A visualization of the force distribution for different Z-scores is saved with the filtered dataset.

    Parameters
    ----------
    input_dir : pathlib.Path
        Path to the directory containing HuggingFace formatted data.
        Filtered data will be saved to current working directory with name [input_dir.name]-z-score.
    z_score_cutoff : float, optional
        The Z-score cutoff for filtering (default is 4.0).
        Entries above this Z-score will be removed.

    Raises
    ------
    FileNotFoundError
        If the input dataset directory doesn't exist.

    Notes
    -----
    This function has the following side effects:
    - Creates [input_dir.name]-z-score directory in current working directory containing filtered dataset
    - Saves filtered dataset HuggingFace files to [input_dir.name]-z-score directory
    - Saves force distribution plot as [input_dir.name]-z-score/rms_forces.png
    - Saves high force SMILES list as [input_dir.name]-z-score/high_force_smiles.json
    - Saves filtered SMILES list as [input_dir.name]-z-score/smiles.json
    """

    logger.info(f"Filtering dataset by forces below a Z-score of {z_score_cutoff}...")

    # Create output directory in current working directory with same name as input + suffix
    output_dir = pathlib.Path.cwd() / (input_dir.name + "-z-score")
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = datasets.load_from_disk(str(input_dir))
    dataset = filter_nonparametrizable_molecules(dataset)
    data_df = dataset.to_pandas()

    data_df["rms_forces"] = data_df["forces"].apply(lambda x: get_rms(np.array(x)))
    rms_mean = data_df["rms_forces"].mean()
    rms_std = data_df["rms_forces"].std()
    data_df["z-score"] = (data_df["rms_forces"] - rms_mean) / rms_std

    # Create a dict of the Z-scores
    z_score_dict: dict[float, float] = {
        score: score * rms_std + rms_mean for score in [1, 2, 3, 4, 5, 6]
    }
    logger.info(f"Z-scores: {z_score_dict}")

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
    fig.savefig(str(output_dir / "rms_forces.png"), dpi=300, bbox_inches="tight")

    # Get the data above the specified Z-score cutoff
    df_cap_z_score = data_df[data_df["rms_forces"] > z_score_dict[z_score_cutoff]]
    logger.info(f"Cutoff: {z_score_dict[z_score_cutoff]:.2f} kcal/(mol Angstrom)")
    high_force_smiles: list[str] = df_cap_z_score["smiles"].tolist()
    with open(output_dir / "high_force_smiles.json", "w") as file:
        json.dump(high_force_smiles, file, indent=4)
    logger.info(f"Removed {len(df_cap_z_score)} entries with high forces")

    # Save a filtered dataset without the high forces
    filtered_dataset = dataset.filter(lambda x: x["smiles"] not in high_force_smiles)
    filtered_dataset.save_to_disk(output_dir)
    logger.info(
        f"Filtered dataset (containing {len(filtered_dataset)} entries) saved to {output_dir}"
    )

    # Save all of the smiles to a json file in the output directory
    with open(output_dir / "smiles.json", "w") as file:
        json.dump(list(filtered_dataset.unique("smiles")), file, indent=4)
    logger.info(f"Auxiliary files saved to {output_dir}")


def main(
    data_dir: pathlib.Path | str,
    percentile_cutoff: float = 95.0,
    z_score_cutoff: float = 4.0,
) -> None:
    """Main processing function for filtering molecular datasets by forces.

    Orchestrates the filtering workflow by applying both percentile-based and
    Z-score-based filtering to remove entries with high RMS forces from molecular datasets.

    Parameters
    ----------
    data_dir : pathlib.Path | str
        Path to the directory containing HuggingFace formatted molecular dataset.
        The directory should contain dataset_info.json, state.json, and .arrow files.
    percentile_cutoff : float, optional
        The percentile cutoff for filtering (default is 95.0).
        Entries above this percentile will be removed.
    z_score_cutoff : float, optional
        The Z-score cutoff for filtering (default is 4.0).
        Entries above this Z-score will be removed.

    Returns
    -------
    None

    Notes
    -----
    This function performs the following workflow:
    1. Applies percentile filtering to remove entries above the specified percentile for RMS forces
    2. Applies Z-score filtering to remove entries above the specified Z-score for RMS forces
    3. Saves the filtered datasets to current working directory with appropriate naming
    4. Generates visualizations of force distributions within the filtered dataset directories
    5. Saves lists of removed and remaining SMILES within the filtered dataset directories

    This function is typically called from the command-line interface but
    can also be used programmatically when importing the module.

    Examples
    --------
    >>> # Programmatic usage with defaults (95th percentile and Z-score 4):
    >>> data_dir = pathlib.Path("./data-raw")
    >>> main(data_dir)

    >>> # With custom cutoffs:
    >>> main(pathlib.Path("/path/to/molecular/dataset"), percentile_cutoff=90.0, z_score_cutoff=3.0)
    """
    data_dir = pathlib.Path(data_dir)
    filter_dataset_by_forces_percentile(data_dir, percentile_cutoff)
    filter_dataset_by_forces_z_score(data_dir, z_score_cutoff)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Filter HuggingFace structured dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python filter.py --data-dir /path/to/data/directory
    python filter.py --data-dir /path/to/data/directory --percentile-cutoff 90
    python filter.py --data-dir /path/to/data/directory --z-score-cutoff 3
    python filter.py --data-dir /path/to/data/directory --percentile-cutoff 90 --z-score-cutoff 3
        """,
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Directory to HuggingFace structured data",
    )
    parser.add_argument(
        "--percentile-cutoff",
        type=float,
        default=95.0,
        help="Percentile cutoff for filtering high force entries (default: 95.0)",
    )
    parser.add_argument(
        "--z-score-cutoff",
        type=float,
        default=4.0,
        help="Z-score cutoff for filtering high force entries (default: 4.0)",
    )
    args = parser.parse_args()
    main(args.data_dir, args.percentile_cutoff, args.z_score_cutoff)
