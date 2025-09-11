"""Split molecular datasets into training and testing sets.

This module splits HuggingFace molecular datasets using DeepChem's MaxMinSplitter
to maximize structural diversity between training and test sets.

Command-line Usage
------------------
python split_train_test_data.py --data-dir DATASET_PATH [OPTIONS]

Required Input
--------------
DATASET_PATH/ containing:
├── dataset_info.json          # Dataset metadata and schema
├── state.json                 # Dataset state information
├── smiles.json                # List of SMILES strings (non-HuggingFace file)
└── data-*.arrow               # Molecular data (coords, energy, forces)

Output
------
Creates in current working directory:
├── data-train/                # Training dataset (default 95%)
├── data-test/                 # Testing dataset (default 5%)
└── smiles_test_train.json     # SMILES split mapping

Examples
--------
Basic usage:
    $ python split_train_test_data.py --data-dir ./filtered_dataset

With options:
    $ python split_train_test_data.py --data-dir ./data --max-mols 10000 --seed 123
    $ python split_train_test_data.py --data-dir ./data --max-mols -1 --seed 123  # Use all molecules
"""

import pathlib
import json

import numpy as np
import argparse
import datasets
from loguru import logger
import deepchem as dc


def split_train_test(
    filename_data: pathlib.Path | str,
    max_mols: int = -1,
    seed: int = 42,
    kwargs_split: dict = {},
) -> None:
    """Split molecular dataset using DeepChem's MaxMinSplitter for structural diversity.

    Parameters
    ----------
    filename_data : pathlib.Path | str
        Path to HuggingFace dataset directory that also contains smiles.json.
    max_mols : int, default=-1
        Maximum number of molecules to include. If -1, uses all molecules.
    seed : int, default=42
        Random seed for reproducibility in sampling and splitting.
    kwargs_split : dict, default={}
        Additional arguments for MaxMinSplitter.train_test_split.
        Defaults include {"frac_train": 0.95}.

    Returns
    -------
    None

    Raises
    ------
    RuntimeError
        If SMILES string not found in training or testing set.
    FileNotFoundError
        If smiles.json file missing from dataset directory.

    Notes
    -----
    Creates files in current working directory:
    - data-train/ : Training dataset (default 95% of data)
    - data-test/ : Testing dataset (default 5% of data)
    - smiles_test_train.json : SMILES strings for each split

    Uses MaxMinSplitter to maximize structural diversity between splits.

    Examples
    --------
    >>> split_train_test("filtered_dataset")
    >>> split_train_test("data", max_mols=5000, seed=123)
    """
    options = {"seed": seed}
    options["frac_train"] = kwargs_split.get("frac_train", 0.95)
    logger.info(f"MinMax Splitting Options: {options}")

    filename_data = pathlib.Path(filename_data)
    logger.info(f"Loading dataset from: {filename_data.resolve()}")
    input_dataset = datasets.load_from_disk(str(filename_data))

    filename = filename_data / "smiles.json"
    logger.info(f"Loading smiles.json from: {filename.resolve()}")
    with open(filename, "r") as file:
        smiles = json.load(file)

    # Optionally restrict the number of datapoints
    if max_mols != -1 and max_mols < len(smiles):
        # Use numpy random number generator to avoid HuggingFace shuffle which loads the dataset into memory
        rng = np.random.default_rng(seed)
        selected_indices = rng.choice(len(smiles), size=max_mols, replace=False)
        discarded_smiles = [
            smiles[i] for i, smi in enumerate(smiles) if smi not in selected_indices
        ]
        smiles = [smiles[i] for i in selected_indices]
        logger.info(f"Total number of points limited to {max_mols}, using seed={seed}")
    else:
        logger.info(f"Total number of points: {len(smiles)}")
        discarded_smiles = []

    output_dirs = {
        "train": pathlib.Path.cwd() / "data-train",
        "test": pathlib.Path.cwd() / "data-test",
    }
    logger.info(
        f"Splitting dataset into training and testing sets, writing to: {pathlib.Path.cwd()}"
    )
    Xs = np.zeros(len(smiles))
    dc_dataset = dc.data.DiskDataset.from_numpy(X=Xs, ids=smiles)
    maxminspliter = dc.splits.MaxMinSplitter()
    train_dataset, test_dataset = maxminspliter.train_test_split(
        dataset=dc_dataset,
        train_dir=output_dirs["train"],
        test_dir=output_dirs["test"],
        **options,
    )
    overlap = set(train_dataset.ids).intersection(set(test_dataset.ids))
    if overlap:
        raise RuntimeError(
            f"Data contamination detected: {len(overlap)} overlapping SMILES between train and test sets."
        )

    train_index, test_index = [], []
    for i, entry in enumerate(input_dataset):
        if entry["smiles"] in train_dataset.ids:
            train_index.append(i)
        elif entry["smiles"] in test_dataset.ids:
            test_index.append(i)
        elif entry["smiles"] not in discarded_smiles:
            raise RuntimeError("The smiles was not in training or testing")

    logger.info(
        f"N_Train: {len(train_index)}, N_Test: {len(test_index)}, N_Total (before max_mols): {len(input_dataset)}"
    )
    train_split = input_dataset.select(indices=train_index)
    train_split.save_to_disk(output_dirs["train"])
    test_split = input_dataset.select(indices=test_index)
    test_split.save_to_disk(output_dirs["test"])
    logger.info("Done splitting dataset into training and testing sets.")

    smiles_train_test_dict = {
        "train": train_split.unique("smiles"),
        "test": test_split.unique("smiles"),
    }

    # Save the smiles to a json file
    filename = pathlib.Path.cwd() / "smiles_test_train.json"
    with open(filename, "w") as file:
        json.dump(smiles_train_test_dict, file)
    logger.info(f"Saved train/test smiles to {filename}")


def main(
    filename_data: pathlib.Path | str,
    max_mols: int = -1,
    seed: int = 42,
    frac_train: float = 0.95,
) -> None:
    """Split molecular dataset into training and test sets.

    Parameters
    ----------
    filename_data : pathlib.Path | str
        Path to HuggingFace dataset directory containing dataset_info.json,
        state.json, and .arrow files.
    max_mols : int, default=-1
        Maximum number of molecules to include. If -1, uses all molecules.
    seed : int, default=42
        Random seed for reproducibility.
    frac_train : float, default=0.95
        Fraction of data for training set (remainder goes to test set).

    Returns
    -------
    None

    Notes
    -----
    Creates the following files in current working directory:
    - data-train/ : Training dataset
    - data-test/ : Testing dataset
    - smiles_test_train.json : SMILES strings for each split

    Examples
    --------
    >>> main("filtered_dataset")
    >>> main("data", max_mols=5000, seed=123, frac_train=0.9)
    """

    filename_data = pathlib.Path(filename_data)
    split_train_test(
        filename_data,
        max_mols=max_mols,
        seed=seed,
        kwargs_split={
            "frac_train": frac_train,
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split HuggingFace dataset into training and test sets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python split_train_test_data.py --data-dir ./filtered_dataset
    python split_train_test_data.py --data-dir ./data --max-mols 10000 --seed 123
    python split_train_test_data.py --data-dir ./data --max-mols -1 --seed 123  # Use all molecules
        """,
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Directory containing HuggingFace structured dataset",
    )
    parser.add_argument(
        "--max-mols",
        type=int,
        default=-1,
        help="Maximum number of molecules to include. A value of -1 includes all molecules (default: -1)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--frac-train",
        type=float,
        default=0.95,
        help="Fraction of data for training (default: 0.95)",
    )
    args = parser.parse_args()
    main(
        args.data_dir,
        max_mols=args.max_mols,
        seed=args.seed,
        frac_train=args.frac_train,
    )
