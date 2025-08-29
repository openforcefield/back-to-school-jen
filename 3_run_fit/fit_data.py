"""Force field parameter optimization using molecular datasets.

This module provides functions to train force field parameters against molecular
energy and force data using gradient-based optimization. The workflow includes
dataset splitting, force field preparation, training with TensorBoard logging,
and output of optimized parameters in OFFXML format.

Command-line Arguments
----------------------
--data-dir : str
    Path to the directory containing HuggingFace formatted molecular dataset.
    This should contain dataset_info.json, state.json, and .arrow files.
--offxml : str
    Path and filename of the starting force field in OFFXML format.
--n-epochs : int, optional
    Number of training epochs (default: 1000).
--learning-rate : float, optional
    Learning rate for Adam optimizer (default: 0.001).
--batch-size : int, optional
    Batch size for training (default: 500).

Examples
--------
Train a force field with default parameters:
$ python fit_data.py --data-dir ./filtered_dataset --offxml openff-2.2.1.offxml

Train with custom hyperparameters:
$ python fit_data.py --data-dir ./filtered_dataset --offxml openff-2.2.1.offxml \\
    --n-epochs 2000 --learning-rate 0.0005 --batch-size 256

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
Creates the following outputs in current working directory:
- data-train/                  # Training dataset split (95% of data)
- data-test/                   # Testing dataset split (5% of data)
- my-smee-fit/                 # Training logs and checkpoints
  ├── events.out.tfevents.*    # TensorBoard event files
  ├── force-field-epoch-*.pt   # Checkpoints every 10 epochs
  └── final-force-field.pt     # Final optimized force field
- final-force-field.offxml     # Optimized force field in OFFXML format
- smiles_test_train.json       # Train/test split SMILES lists
"""

import pathlib
import math
from collections import defaultdict
import json
from typing import Any

import numpy as np
from tqdm import tqdm
import torch
import argparse
import smee.converters
import datasets
import descent
import descent.train
import descent.targets
import descent.targets.energy
from loguru import logger
import deepchem as dc

import tensorboardX
import more_itertools

from openff.toolkit import Molecule, ForceField

parameters = {
    "Bonds": descent.train.ParameterConfig(
        cols=["k", "length"],
        scales={"k": 1e-2, "length": 1.0},  # normalize so roughly equal
        limits={"k": [0.0, None], "length": [0.0, None]},
        # the include/exclude types are Interchange PotentialKey.id's -- typically SMIRKS
        # include=[], <-- bonds to train. Not specifying trains all
        # exclude=[], <-- bonds NOT to train
    ),
    #    "Angles": descent.train.ParameterConfig(
    #        cols=["k", "angle"],
    #        scales={"k": 1e-2, "angle": 1.0},
    #        limits={"k": [0.0, None], "angle": [0.0, math.pi]},
    #    ),
}


def prepare_to_train(
    train_filename_data: pathlib.Path | str, offxml: pathlib.Path | str
) -> tuple[Any, dict[str, Any]]:
    """Prepare molecular dataset and force field objects for training.

    Converts HuggingFace dataset to PyTorch format, creates OpenFF molecular
    objects, generates interchanges, and converts to SMEE format for training.

    Parameters
    ----------
    train_filename_data : pathlib.Path | str
        Path to directory containing training dataset in HuggingFace format.
    offxml : pathlib.Path | str
        Path to the starting force field OFFXML file.

    Returns
    -------
    smee_force_field : Any
        SMEE force field object ready for optimization
    topologies : dict[str, Any]
        Dictionary mapping SMILES strings to SMEE topology objects

    Notes
    -----
    This function performs the following operations:
    1. Loads dataset and sets PyTorch format for energy, coords, forces
    2. Creates OpenFF Molecule objects from SMILES strings
    3. Generates OpenFF Interchange objects for each molecule
    4. Converts to SMEE format for efficient training

    The conversion process handles:
    - Stereochemistry with allow_undefined_stereo=True
    - Automatic topology generation for each unique molecule
    - Force field parameterization via OpenFF Interchange

    Examples
    --------
    >>> smee_ff, topologies = prepare_to_train("data-train", "openff-2.2.1.offxml")
    >>> print(f"Prepared {len(topologies)} molecular topologies")
    """

    train_filename_data = pathlib.Path(train_filename_data)
    offxml = pathlib.Path(offxml)
    logger.info(f"Loading dataset {train_filename_data.resolve()}")
    dataset = datasets.Dataset.load_from_disk(train_filename_data)
    dataset.set_format(
        "torch", columns=["energy", "coords", "forces"], output_all_columns=True
    )

    # Ensure molecules are parametrizable
    dataset_size = len(dataset)
    unique_smiles = descent.targets.energy.extract_smiles(dataset)
    dataset = dataset.filter(lambda d: d["smiles"] in unique_smiles)
    logger.info(
        f"Removed non-parameterisable molecules, dataset size change: {dataset_size} -> {len(dataset)}"
    )

    # Get starting forcefield
    logger.info(f"Loading forcefield: {offxml.resolve()}")
    starting_ff = ForceField(offxml)

    # Get OpenFF Molecules and interchanges
    logger.info("Creating interchanges...")
    all_smiles = []
    interchanges = []
    for entry in tqdm(dataset):
        mol = Molecule.from_mapped_smiles(entry["smiles"], allow_undefined_stereo=True)
        all_smiles.append(entry["smiles"])
        interchange = starting_ff.create_interchange(mol.to_topology())
        interchanges.append(interchange)

    logger.info("Prepare SMEE data structures...")
    smee_force_field, smee_topologies = smee.converters.convert_interchange(
        interchanges
    )
    topologies = dict(zip(all_smiles, smee_topologies))

    return smee_force_field, topologies


def split_train_test(filename_data: pathlib.Path | str) -> None:
    """Split a molecular dataset into training and testing sets using MaxMin splitting.

    Uses DeepChem's MaxMinSplitter to create a diverse training/test split based
    on molecular descriptors, ensuring maximum structural diversity between sets.

    Parameters
    ----------
    filename_data : pathlib.Path | str
        Path to directory containing the molecular dataset in HuggingFace format.
        Must contain smiles.json file in parent directory.

    Returns
    -------
    None

    Raises
    ------
    RuntimeError
        If a SMILES string is found in neither training nor testing set.
    FileNotFoundError
        If smiles.json file is not found in filename_data.parent.

    Notes
    -----
    This function has the following side effects:
    - Creates data-train/ directory with 95% of molecules
    - Creates data-test/ directory with 5% of molecules
    - Saves smiles_test_train.json with SMILES lists for each split
    - Uses MaxMin splitting for maximum structural diversity

    The split is deterministic based on molecular structure, ensuring
    reproducible train/test divisions across runs.

    Examples
    --------
    >>> split_train_test("filtered_dataset")
    # Creates data-train/ and data-test/ directories
    """
    filename_data = pathlib.Path(filename_data)

    output_dirs = {
        "train": pathlib.Path.cwd() / "data-train",
        "test": pathlib.Path.cwd() / "data-test",
    }

    filename = filename_data / "smiles.json"
    logger.info(f"Loading smiles.json from: {filename.resolve()}")
    with open(filename, "r") as file:
        smiles = json.load(file)

    logger.info(f"Loading dataset from: {filename_data.resolve()}")
    input_dataset = datasets.load_from_disk(filename_data)

    logger.info(
        f"Splitting dataset into training and testing sets, writing to: {pathlib.Path.cwd()}"
    )
    Xs = np.zeros(len(smiles))
    dc_dataset = dc.data.DiskDataset.from_numpy(X=Xs, ids=smiles)
    maxminspliter = dc.splits.MaxMinSplitter()
    train_dataset, test_dataset = maxminspliter.train_test_split(
        dataset=dc_dataset,
        frac_train=0.95,
        train_dir=output_dirs["train"],
        test_dir=output_dirs["test"],
    )

    train_index, test_index = [], []
    for i, entry in enumerate(input_dataset):
        if entry["smiles"] in train_dataset.ids:
            train_index.append(i)
        elif entry["smiles"] in test_dataset.ids:
            test_index.append(i)
        else:
            raise RuntimeError("The smiles was not in training or testing")

    logger.info(
        f"Train: {len(train_index)}, Test: {len(test_index)}, Total: {len(input_dataset)}"
    )
    train_split = input_dataset.select(indices=train_index)
    train_split.save_to_disk(output_dirs["train"])
    test_split = input_dataset.select(indices=test_index)
    test_split.save_to_disk(output_dirs["test"])
    logger.info("Done splitting SPICE2 dataset into training and testing sets.")

    smiles_train_test_dict = {
        "train": train_split.unique("smiles"),
        "test": test_split.unique("smiles"),
    }

    # Save the smiles to a json file
    with open(filename_data / "smiles_test_train.json", "w") as file:
        json.dump(smiles_train_test_dict, file)
    logger.info(
        f"Saved train/test smiles to {filename_data / 'smiles_test_train.json'}"
    )


def write_metrics(
    epoch: int,
    loss: torch.Tensor,
    loss_energy: torch.Tensor,
    loss_forces: torch.Tensor,
    writer: tensorboardX.SummaryWriter,
) -> None:
    """Write training metrics to console and TensorBoard.

    Logs training progress including total loss, energy loss, force loss,
    and corresponding RMSE values to both console output and TensorBoard
    for monitoring and visualization.

    Parameters
    ----------
    epoch : int
        Current training epoch number.
    loss : torch.Tensor
        Total loss (energy + force loss) for the epoch.
    loss_energy : torch.Tensor
        Energy-specific loss component for the epoch.
    loss_forces : torch.Tensor
        Force-specific loss component for the epoch.
    writer : tensorboardX.SummaryWriter
        TensorBoard writer object for logging metrics.

    Returns
    -------
    None

    Notes
    -----
    Logs the following metrics to TensorBoard:
    - loss: Total combined loss
    - loss_energy: Energy component loss
    - loss_forces: Force component loss
    - rmse_energy: Square root of energy loss
    - rmse_forces: Square root of force loss

    Console output shows epoch number and total loss value.

    Examples
    --------
    >>> with tensorboardX.SummaryWriter("logs") as writer:
    ...     write_metrics(10, total_loss, energy_loss, force_loss, writer)
    """
    print(f"epoch={epoch} loss={loss.detach().item():.6f}", flush=True)

    writer.add_scalar("loss", loss.detach().item(), epoch)
    writer.add_scalar("loss_energy", loss_energy.detach().item(), epoch)
    writer.add_scalar("loss_forces", loss_forces.detach().item(), epoch)

    writer.add_scalar("rmse_energy", math.sqrt(loss_energy.detach().item()), epoch)
    writer.add_scalar("rmse_forces", math.sqrt(loss_forces.detach().item()), epoch)
    writer.flush()


def train_forcefield(
    train_filename_data: pathlib.Path | str,
    smee_force_field: Any,
    topologies: dict[str, Any],
    n_epochs: int = 1000,
    learning_rate: float = 0.001,
    batch_size: int = 500,
) -> None:
    """Train force field parameters using molecular energy and force data.

    Optimizes force field parameters by minimizing the sum of squared errors
    between predicted and reference energies and forces using gradient descent
    with the Adam optimizer.

    Parameters
    ----------
    train_filename_data : pathlib.Path | str
        Path to directory containing training dataset in HuggingFace format.
    smee_force_field : smee.ForceField
        SMEE force field object with parameters to optimize.
    topologies : dict
        Dictionary mapping SMILES strings to SMEE topology objects.
    n_epochs : int, optional
        Number of training epochs, by default 1000.
    learning_rate : float, optional
        Learning rate for Adam optimizer, by default 0.001.
    batch_size : int, optional
        Number of molecular configurations per batch, by default 500.

    Returns
    -------
    None

    Notes
    -----
    This function has the following side effects:
    - Creates my-smee-fit/ directory with TensorBoard logs
    - Saves force field checkpoints every 10 epochs as .pt files
    - Saves final optimized force field as final-force-field.pt
    - Logs training metrics (loss, RMSE) to TensorBoard

    The loss function is: L = Σ(E_pred - E_ref)² + Σ(F_pred - F_ref)²
    where energies and forces are weighted equally.

    Examples
    --------
    >>> train_forcefield(
    ...     "data-train",
    ...     smee_ff,
    ...     topologies,
    ...     n_epochs=500,
    ...     learning_rate=0.0005
    ... )
    """

    train_filename_data = pathlib.Path(train_filename_data)
    logger.info(f"Loading dataset from: {train_filename_data.resolve()}")
    dataset = datasets.Dataset.load_from_disk(train_filename_data)

    trainable = descent.train.Trainable(
        force_field=smee_force_field, parameters=parameters, attributes={}
    )

    directory = pathlib.Path("my-smee-fit")
    directory.mkdir(exist_ok=True, parents=True)

    trainable_parameters = trainable.to_values()
    device = trainable_parameters.device.type

    logger.info("Start training...")
    with tensorboardX.SummaryWriter(str(directory)) as writer:
        optimizer = torch.optim.Adam(
            [trainable_parameters], lr=learning_rate, amsgrad=True
        )
        dataset_indices = list(range(len(dataset)))

        for i in range(n_epochs):
            ff = trainable.to_force_field(trainable_parameters)
            total_loss = torch.zeros(size=(1,), device=device)
            energy_loss = torch.zeros(size=(1,), device=device)
            force_loss = torch.zeros(size=(1,), device=device)
            grad = None

            for batch_ids in tqdm(
                more_itertools.batched(dataset_indices, batch_size),
                desc="Calculating energies",
                ncols=80,
                total=math.ceil(len(dataset) / batch_size),
            ):
                batch = dataset.select(indices=batch_ids)
                true_batch_size = len(dataset)
                e_ref, e_pred, f_ref, f_pred = descent.targets.energy.predict(
                    batch, ff, topologies, "mean"
                )
                # L2 loss
                batch_loss_energy = ((e_pred - e_ref) ** 2).sum() / true_batch_size
                batch_loss_force = ((f_pred - f_ref) ** 2).sum() / true_batch_size

                # Equal sum of L2 loss on energies and forces
                batch_loss = batch_loss_energy + batch_loss_force

                (batch_grad,) = torch.autograd.grad(
                    batch_loss, trainable_parameters, create_graph=True
                )
                batch_grad = batch_grad.detach()
                if grad is None:
                    grad = batch_grad
                else:
                    grad += batch_grad

                # keep sum of squares to report MSE at the end
                total_loss += batch_loss.detach()
                energy_loss += batch_loss_energy.detach()
                force_loss += batch_loss_force.detach()

            trainable_parameters.grad = grad

            write_metrics(
                epoch=i,
                loss=total_loss,
                loss_energy=energy_loss,
                loss_forces=force_loss,
                writer=writer,
            )

            optimizer.step()
            optimizer.zero_grad()

            if i % 10 == 0:
                torch.save(
                    trainable.to_force_field(trainable_parameters),
                    directory / f"force-field-epoch-{i}.pt",
                )

        logger.info(f"Saving {directory / "final-force-field.pt"}")
        torch.save(
            trainable.to_force_field(trainable_parameters),
            directory / "final-force-field.pt",
        )


def write_new_offxml(smee_force_field: Any, offxml: pathlib.Path | str) -> None:
    """Convert optimized SMEE force field parameters to OFFXML format.

    Takes the optimized parameters from a SMEE force field and writes them
    back to an OpenFF OFFXML file, preserving the original force field
    structure while updating the fitted parameters.

    Parameters
    ----------
    smee_force_field : Any
        Optimized SMEE force field containing fitted parameters.
    offxml : pathlib.Path | str
        Path to the original OFFXML file (used for reference structure).

    Returns
    -------
    None

    Notes
    -----
    This function has the following side effects:
    - Creates final-force-field.offxml in the current working directory
    - Updates parameters for Bonds, Angles, ProperTorsions, and ImproperTorsions
    - Preserves original force field structure and non-fitted parameters

    The function handles different parameter types:
    - Bonds/Angles: Updates both k (force constant) and equilibrium values
    - ProperTorsions: Collects k values by periodicity for each SMIRKS pattern
    - ImproperTorsions: Updates only the k values (v2 terms)

    Examples
    --------
    >>> write_new_offxml(optimized_smee_ff, "openff-2.2.1.offxml")
    """

    offxml = pathlib.Path(offxml)
    logger.info("Writing out new forcefield...")
    starting_ff = ForceField(offxml)

    for potential in smee_force_field.potentials:
        handler_name = potential.parameter_keys[0].associated_handler

        parameter_attrs = potential.parameter_cols
        parameter_units = potential.parameter_units

        if handler_name in ["Bonds", "Angles"]:
            handler = starting_ff.get_parameter_handler(handler_name)
            for i, opt_parameters in enumerate(potential.parameters):
                smirks = potential.parameter_keys[i].id
                ff_parameter = handler[smirks]
                opt_parameters = opt_parameters.detach().cpu().numpy()
                for j, (p, unit) in enumerate(zip(parameter_attrs, parameter_units)):
                    setattr(ff_parameter, p, opt_parameters[j] * unit)

        elif handler_name in ["ProperTorsions"]:
            handler = starting_ff.get_parameter_handler(handler_name)
            k_index = parameter_attrs.index("k")
            p_index = parameter_attrs.index("periodicity")
            # we need to collect the k values into a list across the entries
            collection_data: dict[str, dict[int, float]] = defaultdict(dict)
            for i, opt_parameters in enumerate(potential.parameters):
                smirks = potential.parameter_keys[i].id
                ff_parameter = handler[smirks]
                opt_parameters = opt_parameters.detach().cpu().numpy()
                # find k and the periodicity
                k = opt_parameters[k_index] * parameter_units[k_index]
                p = int(opt_parameters[p_index])
                collection_data[smirks][p] = k
            # now update the force field
            for smirks, k_s in collection_data.items():
                ff_parameter = handler[smirks]
                k_mapped_to_p = [k_s[p] for p in ff_parameter.periodicity]
                ff_parameter.k = k_mapped_to_p

        elif handler_name in ["ImproperTorsions"]:
            k_index = parameter_attrs.index("k")
            handler = starting_ff.get_parameter_handler(handler_name)
            # we only fit the v2 terms for improper torsions so convert to list and set
            for i, opt_parameters in enumerate(potential.parameters):
                smirks = potential.parameter_keys[i].id
                ff_parameter = handler[smirks]
                opt_parameters = opt_parameters.detach().cpu().numpy()
                ff_parameter.k = [opt_parameters[k_index] * parameter_units[k_index]]

    filename = "final-force-field.offxml"
    logger.info(f"Saving new forcefield to: {filename}")
    starting_ff.to_file(filename)


def main(
    filename_data: pathlib.Path | str,
    offxml: pathlib.Path | str,
    n_epochs: int = 1000,
    learning_rate: float = 0.001,
    batch_size: int = 500,
) -> None:
    """Main workflow for force field parameter optimization.

    Orchestrates the complete force field training pipeline: dataset splitting,
    preparation, training, and output generation.

    Parameters
    ----------
    filename_data : pathlib.Path | str
        Path to directory containing HuggingFace formatted molecular dataset.
        Must contain dataset_info.json, state.json, and .arrow files.
    offxml : pathlib.Path | str
        Path to the starting force field in OFFXML format.
    n_epochs : int, optional
        Number of training epochs, by default 1000.
    learning_rate : float, optional
        Learning rate for Adam optimizer, by default 0.001.
    batch_size : int, optional
        Batch size for training, by default 500.

    Returns
    -------
    None

    Notes
    -----
    This function performs the following workflow:
    1. Splits dataset into 95% training / 5% testing using MaxMin splitting
    2. Prepares force field and molecular topologies for training
    3. Trains force field parameters using gradient descent optimization
    4. Converts optimized parameters back to OFFXML format

    Output files created:
    - data-train/: Training dataset
    - data-test/: Testing dataset
    - my-smee-fit/: Training logs and checkpoints
    - final-force-field.offxml: Optimized force field
    - smiles_test_train.json: Train/test SMILES split

    Examples
    --------
    >>> # Basic usage with default parameters
    >>> main("filtered_dataset", "openff-2.2.1.offxml")

    >>> # Custom training parameters
    >>> main(
    ...     "filtered_dataset",
    ...     "openff-2.2.1.offxml",
    ...     n_epochs=2000,
    ...     learning_rate=0.0005,
    ...     batch_size=256
    ... )
    """
    filename_data = pathlib.Path(filename_data)
    offxml = pathlib.Path(offxml)
    split_train_test(filename_data)
    smee_force_field, topologies = prepare_to_train(
        pathlib.Path.cwd() / "data-train", offxml
    )
    train_forcefield(
        pathlib.Path.cwd() / "data-train",
        smee_force_field,
        topologies,
        n_epochs=n_epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
    )
    write_new_offxml(smee_force_field, offxml)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fit HuggingFace structured dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python fit_data.py --data-dir /path/to/data/directory
        """,
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Directory to HuggingFace structured data",
    )
    parser.add_argument(
        "--offxml",
        type=str,
        required=True,
        help="Path and filename of .offxml file",
    )
    parser.add_argument(
        "--n-epochs",
        type=int,
        default=1000,
        help="Number of epochs to undergo in the fitting procedure",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        help="Learning rate ...",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=500,
        help="Batch size ...",
    )
    args = parser.parse_args()
    main(
        args.data_dir,
        args.offxml,
        n_epochs=args.n_epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
    )
