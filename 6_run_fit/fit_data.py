"""Force field parameter optimization using molecular datasets.

This module prepares data for use in parametrization with SMEE. The workflow includes
loading pre-prepared SMEE force fields and topologies that have been exported as .pkl files.

Command-line Arguments
----------------------
--data-dir : str
    Path to directory containing HuggingFace formatted molecular dataset.
    Must contain dataset_info.json, state.json, and .arrow files used as
    an input for the creation of SMEE force field and topology objects.
--filename-forcefield : str
    Path to saved SMEE force field file (.pkl format).
--filename-topo-dict : str
    Path to saved SMEE topologies dictionary file (.pkl format).
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
--to_cuda : bool, optional
    If true, the pytorch objects for the force field and topology objects are
    converted to be GPU compatible (default: False).

Examples
--------
Train a force field with default parameters:
$ python fit_data.py --data-dir ./data-train --filename-forcefield smee_force_field.pkl \\
    --filename-topo-dict smee_topology_dict.pkl --offxml openff-2.2.1.offxml

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

Required .pkl Files
-------------------
filename-forcefield.pkl : file
    Serialized SMEE TensorForceField object containing force field parameters
    and potentials ready for optimization. Generated from OFFXML force field.

filename-topo-dict.pkl : file
    Dictionary mapping SMILES strings to SMEE TensorTopology objects.
    Contains molecular topology information (bonds, angles, dihedrals) for
    each unique molecule in the dataset.

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
import random
from collections import defaultdict
import pickle

import torch
import argparse
import smee
import datasets
import descent
import descent.train
import descent.targets
import descent.targets.energy
from loguru import logger
import numpy as np

import tensorboardX
import more_itertools

from openff.toolkit import ForceField
from openff.units import unit


def get_parameter_col_values(
    ff: ForceField, parameter_type: str, parameter_cols: list[str]
) -> dict:
    """Return column-wise numeric values for a given parameter handler in an OFF force field.

    Parameters
    ----------
    ff : ForceField
        OpenFF ForceField instance to inspect.
    parameter_type : str
        Name of the parameter handler (e.g. "Bonds", "Angles", "ProperTorsions").
    parameter_cols : list[str]
        Column/attribute names to extract from each parameter (e.g. ["k", "length"]).

    Returns
    -------
    dict[str, list[float]]
        Mapping from each requested column name to a list of numeric values found in the
        force field. Missing handlers or missing values result in empty lists for the
        corresponding keys.

    Notes
    -----
    - Scalars and single-element lists are converted to floats. Multi-valued attributes
      (e.g. lists of k values for torsions) will have each entry appended separately.
    """
    values: dict = {k: [] for k in parameter_cols}
    for parameter in ff.get_parameter_handler(parameter_type).parameters:
        for parameter_col in parameter_cols:
            val = getattr(parameter, parameter_col)
            if val is not None:
                values[parameter_col].append(val)
    return values


def _mean_vals(vals):
    nums = []
    for x in vals:
        # pint.Quantity
        if hasattr(x, "m"):
            nums.append(x.m)
        elif hasattr(x, "magnitude"):
            nums.append(x.magnitude)
        # iterable (ValidatedList, list, tuple, ...)
        elif hasattr(x, "__iter__") and not isinstance(x, (str, bytes)):
            for y in x:
                nums.append(getattr(y, "m", getattr(y, "magnitude", y)))
        # plain numeric
        else:
            nums.append(x)
    return float(np.mean(nums))


def get_parameter_scales(offxml: str) -> dict[str, dict[str, float]]:
    """Compute scaling factors for force field parameters using SMEE's internal units.

    Extracts parameter values from an OFFXML force field and converts them to
    SMEE's internal unit system before computing scaling factors (1/mean).
    This ensures that scaling factors match the units used during optimization.

    Parameters
    ----------
    offxml : pathlib.Path | str
        Path to an OFFXML file or a force field identifier that can be loaded by
        openff.toolkit.ForceField.

    Returns
    -------
    dict[str, dict[str, float]]
        Mapping from parameter handler name to a dict of scaling factors.
        Example: {"Bonds": {"k": 0.0018, "length": 0.64}, "Angles": {"k": 0.0086, "angle": 0.497}}

    Notes
    -----
    SMEE internal units (from smee/converters/openff/valence.py):
    - Bonds: k in kcal/(mol·Å²), length in Å
    - Angles: k in kcal/(mol·rad²), angle in radians
    - ProperTorsions: k in kcal/mol
    - ImproperTorsions: k in kcal/mol
    """
    # SMEE default units from smee/converters/openff/valence.py
    # _ANGSTROM = openff.units.unit.angstrom
    # _RADIANS = openff.units.unit.radians
    # _KCAL_PER_MOL = openff.units.unit.kilocalories / openff.units.unit.mole
    SMEE_UNITS = {
        "Bonds": {
            "k": unit.kilocalorie_per_mole / unit.angstrom**2,
            "length": unit.angstrom,
        },
        #        "Angles": {
        #            "k": unit.kilocalorie_per_mole / unit.radian**2,
        #            "angle": unit.radian,
        #        },
    }

    PARAMETER_COLS = {
        "Bonds": ["k", "length"],
        #        "Angles": ["k", "angle"],
    }
    logger.info(f"Loading force field from OFFXML: {offxml}")
    ff = ForceField(offxml)

    scales = {}
    for parameter_type, parameter_cols in PARAMETER_COLS.items():
        logger.info(f"Processing scales for parameter type: {parameter_type}")
        values = get_parameter_col_values(ff, parameter_type, parameter_cols)

        # Convert all parameters to SMEE's internal units
        if parameter_type in SMEE_UNITS:
            for param_name, target_unit in SMEE_UNITS[parameter_type].items():
                if param_name in values:
                    values[param_name] = [
                        val.to(target_unit) if hasattr(val, "to") else val
                        for val in values[param_name]
                    ]
        else:
            raise ValueError(f"Parameter type, {parameter_type}, is unknown.")

        scales[parameter_type] = {
            param: 1 / _mean_vals(vals) for param, vals in values.items()
        }
        logger.info(f"Scaling values for {parameter_type}: {scales[parameter_type]}\n")
    return scales


def get_parameter_limits() -> dict[str, dict[str, tuple[float | None, float | None]]]:
    """Get parameter limits in SMEE's internal unit system.

    Returns physically reasonable bounds for force field parameters during optimization.
    All limits are specified in SMEE's internal units to match the optimization space.

    Returns
    -------
    dict[str, dict[str, tuple[float, float | None]]]
        Mapping from parameter handler name to parameter limits.
        Tuple format: (lower_bound, upper_bound), where None means no upper bound.

    Notes
    -----
    SMEE internal units and typical ranges:
    - Bonds:
        - k: [0, ∞) kcal/(mol·Å²) - force constant must be positive
        - length: [0, ∞) Å - bond length must be positive
    - Angles:
        - k: [0, ∞) kcal/(mol·rad²) - force constant must be positive
        - angle: [0, π] radians - equilibrium angle range
    - ProperTorsions:
        - k: (-∞, ∞) kcal/mol - can be positive or negative
    - ImproperTorsions:
        - k: [0, ∞) kcal/mol - typically positive

    Examples
    --------
    >>> limits = get_parameter_limits()
    >>> limits["Angles"]["angle"]
    (0.0, 3.141592653589793)
    """
    return {
        "Bonds": {
            "k": (0.0, None),  # kcal/(mol·Å²), must be positive
            "length": (0.0, None),  # Å, must be positive
        },
        "Angles": {
            "k": (0.0, None),  # kcal/(mol·rad²), must be positive
            "angle": (0.0, math.pi),  # radians, 0 to π
        },
        "ProperTorsions": {
            "k": (None, None),  # kcal/mol, can be any value
        },
        "ImproperTorsions": {
            "k": (0.0, None),  # kcal/mol, typically positive
        },
    }


def prepare_batch_for_device(batch: datasets.Dataset, device_str: str) -> list:
    """Prepare a batch for the target device (CPU or CUDA).

    Converts coordinate, energy, and force tensors to the specified device
    before passing to descent.targets.energy.predict().

    Parameters
    ----------
    batch : datasets.Dataset
        Batch of molecular data from HuggingFace dataset.
    device_str : str
        Target device string ("cpu" or "cuda").

    Returns
    -------
    list
        List of entries with tensors moved to the target device.
    """
    device_batch = []
    for entry in batch:
        entry_copy = {}
        for key, value in entry.items():
            if key in ["coords", "energy", "forces"]:
                if isinstance(value, torch.Tensor):
                    entry_copy[key] = value.to(device_str)
                else:
                    entry_copy[key] = torch.tensor(value, device=device_str)
            else:
                entry_copy[key] = value
        device_batch.append(entry_copy)
    return device_batch


def load_smee_outputs(
    filename_ff: pathlib.Path | str,
    filename_topo: pathlib.Path | str,
    to_cuda: bool = False,
) -> tuple[smee.TensorForceField, dict[str, smee.TensorTopology]]:
    """Load SMEE force field and topologies from pickle files.

    Parameters
    ----------
    filename_ff : pathlib.Path | str
        Path to saved SMEE force field .pkl file.
    filename_topo : pathlib.Path | str
        Path to saved SMEE topologies .pkl file.
    to_cuda : bool
        If true, the pytorch objects for the force field and topology objects are
        converted to be GPU compatible (default: False).

    Returns
    -------
    smee_force_field : smee.TensorForceField
        SMEE force field tensor object with parameters and potentials.
    topologies : dict[str, smee.TensorTopology]
        Dictionary mapping SMILES strings to SMEE topology tensor objects.

    Raises
    ------
    FileNotFoundError
        If file is not found.
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
    else:
        raise ValueError(
            f"Unsupported file format for force field: {filename_ff.suffix}"
        )

    logger.info(f"Loading dict with SMILES and SMEE topologies from: {filename_topo}")
    if filename_topo.suffix.lower() == ".pkl":
        with open(filename_topo, "rb") as f_pkl:
            topologies = pickle.load(f_pkl)
    else:
        raise ValueError(
            f"Unsupported file format for topologies: {filename_topo.suffix}"
        )

    # Always explicitly move to the target device to ensure consistency
    # This handles cases where .pkl files were saved with tensors on a different device
    target_device = "cuda" if to_cuda else "cpu"
    smee_ff = smee_ff.to(target_device)
    topologies = {
        smiles: topology.to(target_device) for smiles, topology in topologies.items()
    }

    return smee_ff, topologies


def write_metrics(
    epoch: int,
    loss: torch.Tensor,
    loss_energy: torch.Tensor,
    loss_forces: torch.Tensor,
    loss_val: torch.Tensor,
    loss_val_energy: torch.Tensor,
    loss_val_forces: torch.Tensor,
    writer: tensorboardX.SummaryWriter,
) -> None:
    """Write training and validation metrics to console and TensorBoard.

    Logs training progress including total loss, energy loss, force loss,
    and corresponding RMSE values for both training and validation sets
    to both console output and TensorBoard for monitoring and visualization.

    Parameters
    ----------
    epoch : int
        Current training step (minibatch counter).
    loss : torch.Tensor
        Total training loss (energy + force loss) for the epoch.
    loss_energy : torch.Tensor
        Energy-specific training loss component for the epoch.
    loss_forces : torch.Tensor
        Force-specific training loss component for the epoch.
    loss_val : torch.Tensor
        Total validation loss (energy + force loss) for the epoch.
    loss_val_energy : torch.Tensor
        Energy-specific validation loss component for the epoch.
    loss_val_forces : torch.Tensor
        Force-specific validation loss component for the epoch.
    writer : tensorboardX.SummaryWriter
        TensorBoard writer object for logging metrics.

    Returns
    -------
    None

    Notes
    -----
    TensorBoard metrics logged:
    - loss: Total combined training loss
    - loss_energy: Energy component training loss
    - loss_forces: Force component training loss
    - loss_val: Total combined validation loss
    - loss_val_energy: Energy component validation loss
    - loss_val_forces: Force component validation loss
    - rmse_energy: Square root of training energy loss
    - rmse_forces: Square root of training force loss
    - rmse_val_energy: Square root of validation energy loss
    - rmse_val_forces: Square root of validation force loss

    Examples
    --------
    >>> with tensorboardX.SummaryWriter("logs") as writer:
    ...     write_metrics(10, train_loss, train_energy, train_force,
    ...                   val_loss, val_energy, val_force, writer)
    """
    logger.info(
        f"epoch={epoch} loss_train={loss.detach().item():.6f}, "
        f"loss_val={loss_val.detach().item():.6f}",
        flush=True,
    )

    writer.add_scalar("loss", loss.detach().item(), epoch)
    writer.add_scalar("loss_energy", loss_energy.detach().item(), epoch)
    writer.add_scalar("loss_forces", loss_forces.detach().item(), epoch)

    writer.add_scalar("loss_val", loss_val.detach().item(), epoch)
    writer.add_scalar("loss_val_energy", loss_val_energy.detach().item(), epoch)
    writer.add_scalar("loss_val_forces", loss_val_forces.detach().item(), epoch)

    writer.add_scalar("rmse_energy", math.sqrt(loss_energy.detach().item()), epoch)
    writer.add_scalar("rmse_forces", math.sqrt(loss_forces.detach().item()), epoch)

    writer.add_scalar(
        "rmse_val_energy", math.sqrt(loss_val_energy.detach().item()), epoch
    )
    writer.add_scalar(
        "rmse_val_forces", math.sqrt(loss_val_forces.detach().item()), epoch
    )

    writer.flush()


def train_forcefield(
    train_filename_data: pathlib.Path | str,
    offxml: pathlib.Path | str,
    smee_force_field: smee.TensorForceField,
    topologies: dict[str, smee.TensorTopology],
    n_epochs: int = 1000,
    learning_rate: float = 0.001,
    minibatch_size: int = 256,
    val_filename_data: pathlib.Path | str | None = None,
    to_cuda: bool = False,
    output_dir: pathlib.Path | str = pathlib.Path("my-smee-fit"),
) -> pathlib.Path:
    """Train force field parameters using molecular energy and force data.

    Optimizes force field parameters by minimizing the mean squared errors
    between predicted and reference energies and forces using gradient descent
    with the
    `Adam optimizer<https://docs.pytorch.org/docs/stable/generated/torch.optim.Adam.html>`_.
    Uses minibatch gradient descent where the optimizer updates parameters after
    each minibatch within an epoch.

    Parameters
    ----------
    train_filename_data : pathlib.Path | str
        Path to directory containing training dataset in HuggingFace format.
        Must contain dataset_info.json, state.json, and .arrow files.
    offxml : pathlib.Path | str
        Path to the force field file.
    smee_force_field : smee.TensorForceField
        SMEE force field tensor object with parameters to optimize.
    topologies : dict[str, smee.TensorTopology]
        Dictionary mapping SMILES strings to SMEE topology tensor objects.
    n_epochs : int, optional
        Number of training epochs (default: 1000).
    learning_rate : float, optional
        Learning rate for Adam optimizer (default: 0.001).
    minibatch_size : int, optional
        Number of training examples per minibatch. The optimizer updates
        parameters after processing each minibatch (default: 256).
    val_filename_data : pathlib.Path | str | None, optional
        Path to directory containing validation dataset in HuggingFace format.
        If provided, validation loss will be tracked but will not influence
        parameter updates (default: None).
    to_cuda : bool, optional
        If True, run training on GPU. If False, use CPU (default: False).
    output_dir : pathlib.Path | str, optional
        Directory for saving training outputs (default: "my-smee-fit").

    Returns
    -------
    pathlib.Path
        Path to the final optimized force field file (final-force-field.pt).

    Notes
    -----
    Side effects:
    - Creates output_dir/ directory with TensorBoard logs
    - Saves force field checkpoints every 10 epochs as .pt files
    - Saves final optimized force field as final-force-field.pt
    - Logs training metrics (loss, RMSE) to TensorBoard

    Loss function: L = MSE(E_pred, E_ref) + MSE(F_pred, F_ref)
    where MSE is the mean squared error and energies and forces are weighted equally.

    Examples
    --------
    >>> ff_path = train_forcefield(
    ...     "data-train",
    ...     "openff-2.2.1.offxml"
    ...     smee_ff,
    ...     topologies,
    ...     n_epochs=500,
    ...     learning_rate=0.0005
    ... )
    """

    train_filename_data = pathlib.Path(train_filename_data)
    logger.info(f"Loading training dataset from: {train_filename_data.resolve()}")
    dataset_train = datasets.Dataset.load_from_disk(train_filename_data)

    # Load validation dataset if provided
    dataset_val = None
    if val_filename_data is not None:
        val_filename_data = pathlib.Path(val_filename_data)
        logger.info(f"Loading validation dataset from: {val_filename_data.resolve()}")
        dataset_val = datasets.Dataset.load_from_disk(val_filename_data)

    # Validate that all SMILES in the training dataset have corresponding topologies
    dataset_train_smiles = set(entry["smiles"] for entry in dataset_train)
    topology_smiles = set(topologies.keys())
    missing_smiles = dataset_train_smiles - topology_smiles

    if missing_smiles:
        logger.warning(
            f"Found {len(missing_smiles)} SMILES in training dataset without matching topologies. "
            f"These molecules will be excluded from training."
        )
        for smiles in list(missing_smiles)[:5]:  # Log first 5 missing
            logger.debug(f"Missing topology for SMILES: {smiles}")
        if len(missing_smiles) > 5:
            logger.debug(f"... and {len(missing_smiles) - 5} more missing SMILES")

        # Filter training dataset to only include molecules with topologies
        original_size = len(dataset_train)
        dataset_train = dataset_train.filter(
            lambda example: example["smiles"] in topology_smiles,
            desc="Filtering training molecules with topologies",
        )
        logger.info(
            f"Training dataset filtered: {original_size} -> {len(dataset_train)} molecules "
            f"({len(dataset_train)/original_size*100:.1f}% retained)"
        )

        if len(dataset_train) == 0:
            raise ValueError(
                "No molecules remaining in training dataset after filtering! Check that the topology "
                "dictionary was generated from the same dataset."
            )

    # Filter validation dataset if provided
    if dataset_val is not None:
        dataset_val_smiles = set(entry["smiles"] for entry in dataset_val)
        missing_val_smiles = dataset_val_smiles - topology_smiles

        if missing_val_smiles:
            logger.warning(
                f"Found {len(missing_val_smiles)} SMILES in validation dataset without matching topologies. "
                f"These molecules will be excluded from validation."
            )

            # Filter validation dataset to only include molecules with topologies
            original_val_size = len(dataset_val)
            dataset_val = dataset_val.filter(
                lambda example: example["smiles"] in topology_smiles,
                desc="Filtering validation molecules with topologies",
            )
            logger.info(
                f"Validation dataset filtered: {original_val_size} -> {len(dataset_val)} molecules "
                f"({len(dataset_val)/original_val_size*100:.1f}% retained)"
            )

    # Determine target device - use CPU by default for consistency
    # The trainable will create parameters on the same device as the force field
    # Use string device specifier for compatibility with smee's .to() method
    device_str = "cuda" if to_cuda else "cpu"
    device = torch.device(device_str)

    # Ensure force field and topologies are on the target device before creating Trainable
    smee_force_field = smee_force_field.to(device_str)
    topologies = {
        smiles: topology.to(device_str) for smiles, topology in topologies.items()
    }

    logger.info(f"Using device: {device_str}")

    # Verify topologies are on the correct device
    for smiles, topo in list(topologies.items())[:1]:  # Check first topology
        for potential in smee_force_field.potentials:
            handler = potential.parameter_keys[0].associated_handler
            if hasattr(topo, "parameters") and handler in topo.parameters:
                param_map = topo.parameters[handler]
                if (
                    hasattr(param_map, "particle_idxs")
                    and param_map.particle_idxs is not None
                ):
                    logger.info(
                        f"Topology {handler} particle_idxs device: {param_map.particle_idxs.device}"
                    )

    scale_mean_parameter_values = get_parameter_scales(str(offxml))
    parameter_limits = get_parameter_limits()

    PARAMETERS = {
        "Bonds": descent.train.ParameterConfig(
            cols=["k", "length"],
            scales=scale_mean_parameter_values["Bonds"],
            limits=parameter_limits["Bonds"],
            include=[],
            exclude=[],
        ),
    }

    # Conditionally include angle parameters if present in the parameter configuration.
    # This avoids relying on commented-out code and automatically enables angle training
    # when the force field and scaling information contain angle terms.
    if "Angles" in scale_mean_parameter_values and "Angles" in parameter_limits:
        PARAMETERS["Angles"] = descent.train.ParameterConfig(
            cols=["k", "angle"],
            scales=scale_mean_parameter_values["Angles"],
            limits=parameter_limits["Angles"],
            include=[],
            exclude=[],
        )
    trainable = descent.train.Trainable(
        force_field=smee_force_field, parameters=PARAMETERS, attributes={}
    )

    directory = pathlib.Path(output_dir)
    directory.mkdir(exist_ok=True, parents=True)

    trainable_parameters = trainable.to_values()

    logger.info("Start training...")
    logger.info(f"Training dataset size: {len(dataset_train)}")
    if dataset_val is not None:
        logger.info(f"Validation dataset size: {len(dataset_val)}")

    with tensorboardX.SummaryWriter(str(directory)) as writer:
        optimizer = torch.optim.Adam(
            [trainable_parameters], lr=learning_rate, amsgrad=True
        )
        dataset_train_indices = list(range(len(dataset_train)))

        # Seed the Python RNG to ensure reproducible minibatch ordering across runs
        random.seed(42)
        for i in range(n_epochs):
            # Shuffle indices at the start of each epoch to randomize minibatch order
            random.shuffle(dataset_train_indices)

            # Calculate number of minibatches per epoch
            n_minibatches = math.ceil(len(dataset_train) / minibatch_size)

            # Iterate through minibatches within the epoch
            for j, minibatch_ids in enumerate(
                more_itertools.batched(dataset_train_indices, minibatch_size)
            ):
                logger.info(f"Epoch {i}, minibatch {j+1} of {n_minibatches}")

                # Select minibatch from training data
                minibatch = dataset_train.select(indices=list(minibatch_ids))

                # Convert to force field for this minibatch
                ff = trainable.to_force_field(trainable_parameters).to(device_str)

                # Prepare minibatch for the target device (CPU or CUDA)
                device_minibatch = prepare_batch_for_device(minibatch, device_str)

                # Compute predictions and losses for entire minibatch
                e_ref, e_pred, f_ref, f_pred = descent.targets.energy.predict(
                    device_minibatch, ff, topologies, "mean"
                )

                # L2 loss normalized by minibatch size
                minibatch_size_actual = len(minibatch)
                minibatch_energy_loss = (
                    (e_pred - e_ref) ** 2
                ).sum() / minibatch_size_actual
                minibatch_force_loss = (
                    (f_pred - f_ref) ** 2
                ).sum() / minibatch_size_actual

                # Equal sum of L2 loss on energies and forces
                minibatch_loss = minibatch_energy_loss + minibatch_force_loss

                # Compute gradient for this minibatch
                (grad,) = torch.autograd.grad(
                    minibatch_loss, trainable_parameters, create_graph=True
                )
                grad = grad.detach()

                # Set gradient and update parameters
                trainable_parameters.grad = grad
                optimizer.step()
                optimizer.zero_grad()

                # Compute validation loss if validation data provided
                val_loss = torch.zeros(size=(1,), device=device)
                val_energy_loss = torch.zeros(size=(1,), device=device)
                val_force_loss = torch.zeros(size=(1,), device=device)

                if dataset_val is not None:
                    # Recompute force field with updated parameters for validation
                    ff_val = trainable.to_force_field(trainable_parameters).to(
                        device_str
                    )

                    # Process entire validation dataset at once
                    device_val_batch = prepare_batch_for_device(dataset_val, device_str)
                    val_dataset_size = len(dataset_val)

                    (
                        e_ref_val,
                        e_pred_val,
                        f_ref_val,
                        f_pred_val,
                    ) = descent.targets.energy.predict(
                        device_val_batch, ff_val, topologies, "mean"
                    )

                    # Compute validation losses (no gradient computation)
                    val_energy_loss = (
                        (e_pred_val - e_ref_val) ** 2
                    ).sum() / val_dataset_size
                    val_force_loss = (
                        (f_pred_val - f_ref_val) ** 2
                    ).sum() / val_dataset_size
                    val_loss = val_energy_loss + val_force_loss

                # Write metrics for this minibatch
                # Use step counter to track minibatch progress
                epoch_step = i * n_minibatches + j
                write_metrics(
                    epoch=epoch_step,
                    loss=minibatch_loss,
                    loss_energy=minibatch_energy_loss,
                    loss_forces=minibatch_force_loss,
                    loss_val=val_loss,
                    loss_val_energy=val_energy_loss,
                    loss_val_forces=val_force_loss,
                    writer=writer,
                )

            # Save checkpoint at the end of each epoch
            if i % 10 == 0:
                torch.save(
                    trainable.to_force_field(trainable_parameters),
                    directory / f"force-field-epoch-{i}.pt",
                )

        final_ff_path = directory / "final-force-field.pt"
        logger.info(f"Saving {final_ff_path}")
        torch.save(
            trainable.to_force_field(trainable_parameters),
            final_ff_path,
        )

        return final_ff_path


def write_new_offxml(
    offxml: pathlib.Path | str, optimized_ff_path: pathlib.Path | str
) -> None:
    """Convert optimized SMEE force field parameters to OFFXML format.

    Loads the optimized force field from the specified path and writes the fitted
    parameters back to an OpenFF OFFXML file, preserving the original force field
    structure while updating only the fitted parameters.

    Parameters
    ----------
    offxml : pathlib.Path | str
        Path to the reference OFFXML file used for output structure.
        Must be the same OFFXML used during training.
    optimized_ff_path : pathlib.Path | str
        Path to the optimized SMEE force field .pt file.

    Returns
    -------
    None

    Notes
    -----
    Side effects:
    - Loads optimized force field from optimized_ff_path (must exist)
    - Creates final-force-field.offxml in current working directory
    - Updates parameters for Bonds, Angles, ProperTorsions, and ImproperTorsions
    - Preserves original force field structure and non-fitted parameters

    Parameter handling by type:
    - Bonds/Angles: Updates k (force constant) and equilibrium values
    - ProperTorsions: Collects k values by periodicity for each SMIRKS pattern
    - ImproperTorsions: Updates only the k values (v2 terms)

    Examples
    --------
    >>> write_new_offxml("openff-2.2.1.offxml")
    >>> write_new_offxml("openff-2.2.1.offxml", "custom-dir/optimized-ff.pt")
    """

    # Load the optimized force field that was saved during training
    optimized_ff_path = pathlib.Path(optimized_ff_path)
    logger.info(f"Loading optimized force field from: {optimized_ff_path}")

    if not optimized_ff_path.exists():
        raise FileNotFoundError(
            f"Optimized force field not found at {optimized_ff_path}. "
            "Ensure training has completed successfully."
        )

    smee_force_field = torch.load(optimized_ff_path)

    offxml = pathlib.Path(offxml)
    logger.info("Writing optimized parameters to new OFFXML force field...")
    starting_ff = ForceField(str(offxml))

    for potential in smee_force_field.potentials:
        handler_name = potential.parameter_keys[0].associated_handler
        if handler_name is None:
            logger.warning("Skipping potential with no associated handler")
            continue

        try:
            handler = starting_ff.get_parameter_handler(handler_name)
        except Exception:
            logger.warning(f"Handler {handler_name} not found in force field, skipping")
            continue

        parameter_attrs = potential.parameter_cols
        parameter_units = potential.parameter_units

        logger.info(f"Updating {len(potential.parameters)} {handler_name} parameters")

        if handler_name in ["Bonds", "Angles"]:
            # Update force constants and equilibrium values directly
            for param_key, opt_parameters in zip(
                potential.parameter_keys, potential.parameters
            ):
                ff_parameter = handler[param_key.id]
                opt_parameters = opt_parameters.detach().cpu().numpy()

                for param_name, param_value, param_unit in zip(
                    parameter_attrs, opt_parameters, parameter_units
                ):
                    setattr(ff_parameter, param_name, param_value * param_unit)

        elif handler_name == "ProperTorsions":
            # Collect k values by periodicity for each SMIRKS pattern
            k_index = parameter_attrs.index("k")
            p_index = parameter_attrs.index("periodicity")
            collection_data: dict[str, dict[int, float]] = defaultdict(dict)

            for param_key, opt_parameters in zip(
                potential.parameter_keys, potential.parameters
            ):
                opt_parameters = opt_parameters.detach().cpu().numpy()
                k = opt_parameters[k_index] * parameter_units[k_index]
                periodicity = int(opt_parameters[p_index])
                collection_data[param_key.id][periodicity] = k

            # Update force field with collected k values
            for smirks, k_by_periodicity in collection_data.items():
                ff_parameter = handler[smirks]
                ff_parameter.k = [k_by_periodicity[p] for p in ff_parameter.periodicity]

        elif handler_name == "ImproperTorsions":
            # Only fit v2 terms for improper torsions
            k_index = parameter_attrs.index("k")
            for param_key, opt_parameters in zip(
                potential.parameter_keys, potential.parameters
            ):
                ff_parameter = handler[param_key.id]
                opt_parameters = opt_parameters.detach().cpu().numpy()
                ff_parameter.k = [opt_parameters[k_index] * parameter_units[k_index]]

    filename = "final-force-field.offxml"
    logger.info(f"Saving optimized force field to: {filename}")
    starting_ff.to_file(filename)


def main(
    filename_data: pathlib.Path | str,
    filename_ff: pathlib.Path | str,
    filename_topo: pathlib.Path | str,
    offxml: pathlib.Path | str,
    val_filename_data: pathlib.Path | str | None = None,
    n_epochs: int = 1000,
    learning_rate: float = 0.001,
    minibatch_size: int = 500,
    to_cuda: bool = False,
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
        Path to saved SMEE force field .pkl file.
    filename_topo : pathlib.Path | str
        Path to saved SMEE topologies dictionary .pkl file.
    offxml : pathlib.Path | str
        Path to reference OFFXML force field file for output structure.
    val_filename_data : pathlib.Path | str | None = None
        Path to directory containing validation dataset in HuggingFace format.
        Must contain dataset_info.json, state.json, and .arrow files.
    n_epochs : int, optional
        Number of training epochs (default: 1000).
    learning_rate : float, optional
        Learning rate for
        `Adam optimizer<https://docs.pytorch.org/docs/stable/generated/torch.optim.Adam.html>`_
        (default: 0.001).
    minibatch_size : int, optional
        Batch size for training (default: 500).
    to_cuda : bool
        If true, the pytorch objects for the force field and topology objects are
        converted to be GPU compatible (default: False).

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
    ...     minibatch_size=256
    ... )
    """
    filename_data = pathlib.Path(filename_data)
    offxml = pathlib.Path(offxml)
    smee_force_field, topologies = load_smee_outputs(
        filename_ff, filename_topo, to_cuda=to_cuda
    )

    # Train force field and get path to optimized parameters
    optimized_ff_path = train_forcefield(
        filename_data,
        offxml,
        smee_force_field,
        topologies,
        n_epochs=n_epochs,
        learning_rate=learning_rate,
        minibatch_size=minibatch_size,
        val_filename_data=val_filename_data,
        to_cuda=to_cuda,
    )

    # Convert optimized force field to OFFXML format
    write_new_offxml(offxml, optimized_ff_path)


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
        help="Filename for SMEE forcefield .pkl file",
    )
    parser.add_argument(
        "--filename-topo-dict",
        type=str,
        required=True,
        help="Filename for dictionary of SMILES and SMEE topologies .pkl file",
    )
    parser.add_argument(
        "--offxml",
        type=str,
        required=True,
        help="Path and filename of .offxml file",
    )
    parser.add_argument(
        "--val-data-dir",
        type=str,
        default=None,
        help="Directory to HuggingFace structured data",
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
        help="Learning rate input for PyTorch Adam optimizer",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=500,
        help="Batch size",
    )
    parser.add_argument(
        "--to-cuda",
        type=bool,
        default=False,
        help="Whether to convert pytorch data to be formatted for GPUs",
    )
    args = parser.parse_args()
    main(
        args.data_dir,
        args.filename_forcefield,
        args.filename_topo_dict,
        args.offxml,
        val_filename_data=args.val_data_dir,
        n_epochs=args.n_epochs,
        learning_rate=args.learning_rate,
        minibatch_size=args.batch_size,
        to_cuda=args.to_cuda,
    )
