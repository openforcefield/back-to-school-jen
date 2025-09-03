"""Force field parameter optimization using molecular datasets.

This module provides functions to train force field parameters against molecular
energy and force data using gradient-based optimization. The workflow includes
loading pre-prepared SMEE force fields and topologies, training with TensorBoard
logging, and output of optimized parameters in OFFXML format.

Command-line Arguments
----------------------
--data-dir : str
    Path to directory containing HuggingFace formatted molecular dataset.
    Must contain dataset_info.json, state.json, and .arrow files used as
    an imput for the creation of SMEE force field and topology objects.
--filename-forcefield : str
    Path to saved SMEE force field file (.pkl or .json format).
--filename-topo-dict : str
    Path to saved SMEE topologies dictionary file (.pkl or .json format).
--offxml : str
    Path to reference OFFXML force field file for output structure.
    Must be the same as that used for constructing SMEE force field and topology
    objects.
--n-epochs : int, optional
    Number of training epochs (default: 1000).
--learning-rate : float, optional
    Learning rate for Adam optimizer (default: 0.001).
--batch-size : int, optional
    Batch size for training (default: 500).

Examples
--------
Train a force field with default parameters:
$ python fit_data.py --data-dir ./data-train --filename-forcefield smee_force_field.pkl \\
    --filename-topo-dict smee_topology_dict.pkl --offxml openff-2.2.1.offxml

Train with custom hyperparameters:
$ python fit_data.py --data-dir ./data-train --filename-forcefield smee_force_field.json \\
    --filename-topo-dict smee_topology_dict.json --offxml openff-2.2.1.offxml \\
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
- my-smee-fit/                 # Training logs and checkpoints
  ├── events.out.tfevents.*    # TensorBoard event files
  ├── force-field-epoch-*.pt   # Checkpoints every 10 epochs
  └── final-force-field.pt     # Final optimized force field
- final-force-field.offxml     # Optimized force field in OFFXML format
"""

import pathlib
import math
from collections import defaultdict
import json
import pickle

from tqdm import tqdm
import torch
import argparse
import smee
import datasets
import descent
import descent.train
import descent.targets
import descent.targets.energy
from loguru import logger

import tensorboardX
import more_itertools

from openff.toolkit import ForceField

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


def load_smee_outputs(
    filename_ff: pathlib.Path | str,
    filename_topo: pathlib.Path | str,
) -> tuple[smee.TensorForceField, dict[str, smee.TensorTopology]]:
    """Load SMEE force field and topologies from disk.

    Loads SMEE objects from either pickle or JSON files. Automatically
    detects file format based on extension.

    Parameters
    ----------
    filename_ff : pathlib.Path | str
        Path to saved SMEE force field file(s) (.pkl or .json).
    filename_topo : pathlib.Path | str
        Path to saved SMEE topologies file(s) (.pkl or .json).

    Returns
    -------
    smee_force_field : smee.TensorForceField
        SMEE force field tensor object with parameters and potentials.
    topologies : dict[str, smee.TensorTopology]
        Dictionary mapping SMILES strings to SMEE topology tensor objects.

    Raises
    ------
    FileNotFoundError
        If fileis not found.
    ValueError
        If file format is not supported or loading fails.

    Examples
    --------
    >>> # Load single files
    >>> ff, topologies = load_smee_outputs(
    ...     "smee_force_field.pkl",
    ...     "smee_topology_dict.pkl"
    ... )
    """

    def convert_to_tensors(obj):
        """Helper function to convert lists back to tensors in JSON data."""
        if isinstance(obj, dict):
            return {k: convert_to_tensors(v) for k, v in obj.items()}
        elif (
            isinstance(obj, list) and len(obj) > 0 and isinstance(obj[0], (int, float))
        ):
            return torch.tensor(obj)
        return obj

    filename_ff = pathlib.Path(filename_ff)
    if not pathlib.Path(filename_ff).exists():
        raise FileNotFoundError(f"Force field file not found: {filename_ff}")

    filename_topo = pathlib.Path(filename_topo)
    if not pathlib.Path(filename_topo).exists():
        raise FileNotFoundError(f"Topology file not found: {filename_topo}")

    logger.info(f"Loading SMEE force field from: {filename_ff}")

    if filename_ff.suffix.lower() == ".pkl":
        with open(filename_ff, "rb") as f_pkl:
            smee_ff = pickle.load(f_pkl)
    elif filename_ff.suffix.lower() == ".json":
        with open(filename_ff, "r") as f_json:
            ff_dict = json.load(f_json)
        try:
            ff_dict = convert_to_tensors(ff_dict)
            smee_ff = smee.TensorForceField(**ff_dict)
        except Exception as e:
            raise ValueError(
                f"Failed to reconstruct force field from JSON {filename_ff}: {e}"
            )
    else:
        raise ValueError(
            f"Unsupported file format for force field: {filename_ff.suffix}"
        )

    logger.info(f"Loading dict with SMILES and SMEE topologies from: {filename_topo}")
    if filename_topo.suffix.lower() == ".pkl":
        with open(filename_topo, "rb") as f_pkl:
            topologies = pickle.load(f_pkl)
    elif filename_topo.suffix.lower() == ".json":
        with open(filename_topo, "r") as f_json:
            topo_dict = json.load(f_json)
        topologies = {}
        try:
            for smiles, topo_data in topo_dict.items():
                # Convert lists back to tensors if needed
                for key, value in topo_data.items():
                    if isinstance(value, list):
                        topo_data[key] = torch.tensor(value)
                topologies[smiles] = smee.TensorTopology(**topo_data)
        except Exception as e:
            raise ValueError(
                f"Failed to reconstruct topologies from JSON {filename_topo}: {e}"
            )
    else:
        raise ValueError(
            f"Unsupported file format for topologies: {filename_topo.suffix}"
        )

    return smee_ff, topologies


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
    TensorBoard metrics logged:
    - loss: Total combined loss
    - loss_energy: Energy component loss
    - loss_forces: Force component loss
    - rmse_energy: Square root of energy loss
    - rmse_forces: Square root of force loss

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
    smee_force_field: smee.TensorForceField,
    topologies: dict[str, smee.TensorTopology],
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
        Must contain dataset_info.json, state.json, and .arrow files.
    smee_force_field : smee.TensorForceField
        SMEE force field tensor object with parameters to optimize.
    topologies : dict[str, smee.TensorTopology]
        Dictionary mapping SMILES strings to SMEE topology tensor objects.
    n_epochs : int, optional
        Number of training epochs (default: 1000).
    learning_rate : float, optional
        Learning rate for Adam optimizer (default: 0.001).
    batch_size : int, optional
        Number of molecular configurations per batch (default: 500).

    Returns
    -------
    None

    Notes
    -----
    Side effects:
    - Creates my-smee-fit/ directory with TensorBoard logs
    - Saves force field checkpoints every 10 epochs as .pt files
    - Saves final optimized force field as final-force-field.pt
    - Logs training metrics (loss, RMSE) to TensorBoard

    Loss function: L = Σ(E_pred - E_ref)² + Σ(F_pred - F_ref)²
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


def write_new_offxml(
    smee_force_field: smee.TensorForceField, offxml: pathlib.Path | str
) -> None:
    """Convert optimized SMEE force field parameters to OFFXML format.

    Takes the optimized parameters from a SMEE force field and writes them
    back to an OpenFF OFFXML file, preserving the original force field
    structure while updating the fitted parameters.

    Parameters
    ----------
    smee_force_field : smee.TensorForceField
        Optimized SMEE force field tensor object containing fitted parameters.
    offxml : pathlib.Path | str
        Path to the reference OFFXML file used for output structure.

    Returns
    -------
    None

    Notes
    -----
    Side effects:
    - Creates final-force-field.offxml in current working directory
    - Updates parameters for Bonds, Angles, ProperTorsions, and ImproperTorsions
    - Preserves original force field structure and non-fitted parameters

    Parameter handling by type:
    - Bonds/Angles: Updates k (force constant) and equilibrium values
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
    filename_ff: pathlib.Path | str,
    filename_topo: pathlib.Path | str,
    offxml: pathlib.Path | str,
    n_epochs: int = 1000,
    learning_rate: float = 0.001,
    batch_size: int = 500,
) -> None:
    """Main workflow for force field parameter optimization.

    Loads pre-prepared SMEE force field and topologies, then orchestrates
    the training pipeline and output generation.

    Parameters
    ----------
    filename_data : pathlib.Path | str
        Path to directory containing training dataset in HuggingFace format.
        Must contain dataset_info.json, state.json, and .arrow files.
    filename_ff : pathlib.Path | str
        Path to saved SMEE force field file (.pkl or .json format).
    filename_topo : pathlib.Path | str
        Path to saved SMEE topologies dictionary file (.pkl or .json format).
    offxml : pathlib.Path | str
        Path to reference OFFXML force field file for output structure.
    n_epochs : int, optional
        Number of training epochs (default: 1000).
    learning_rate : float, optional
        Learning rate for Adam optimizer (default: 0.001).
    batch_size : int, optional
        Batch size for training (default: 500).

    Returns
    -------
    None

    Notes
    -----
    Workflow:
    1. Loads SMEE force field and topologies from disk
    2. Trains force field parameters using gradient descent optimization
    3. Converts optimized parameters back to OFFXML format

    Output files created:
    - my-smee-fit/: Training logs and checkpoints
    - final-force-field.offxml: Optimized force field

    Examples
    --------
    >>> main(
    ...     "data-train",
    ...     "smee_force_field.pkl",
    ...     "smee_topology_dict.pkl",
    ...     "openff-2.2.1.offxml"
    ... )

    >>> # Custom training parameters
    >>> main(
    ...     "data-train",
    ...     "smee_force_field.json",
    ...     "smee_topology_dict.json",
    ...     "openff-2.2.1.offxml",
    ...     n_epochs=2000,
    ...     learning_rate=0.0005,
    ...     batch_size=256
    ... )
    """
    filename_data = pathlib.Path(filename_data)
    offxml = pathlib.Path(offxml)
    smee_force_field, topologies = load_smee_outputs(filename_ff, filename_topo)
    train_forcefield(
        filename_data,
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
        "--filename-forcefield",
        type=str,
        required=True,
        help="Filename for SMEE forcefield .pkl or .json file",
    )
    parser.add_argument(
        "--filename-topo-dict",
        type=str,
        required=True,
        help="Filename for dictionary of SMILES and SMEE topologies .pkl or .json file",
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
        args.filename_forcefield,
        args.filename_topo_dict,
        args.offxml,
        n_epochs=args.n_epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
    )
