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
from collections import defaultdict
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
        "Angles": {
            "k": unit.kilocalorie_per_mole / unit.radian**2,
            "angle": unit.radian,
        },
    }

    PARAMETER_COLS = {
        "Bonds": ["k", "length"],
        "Angles": ["k", "angle"],
        "ProperTorsions": ["k"],
        "ImproperTorsions": ["k"],
    }
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
        logger.info(f"Mean values for {parameter_type}: {scales[parameter_type]}\n")
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
    ...     write_metrics(10, epoch_loss, energy_loss, force_loss, writer)
    """
    logger.info(f"epoch={epoch} loss={loss.detach().item():.6f}", flush=True)

    writer.add_scalar("loss", loss.detach().item(), epoch)
    writer.add_scalar("loss_energy", loss_energy.detach().item(), epoch)
    writer.add_scalar("loss_forces", loss_forces.detach().item(), epoch)

    writer.add_scalar("rmse_energy", math.sqrt(loss_energy.detach().item()), epoch)
    writer.add_scalar("rmse_forces", math.sqrt(loss_forces.detach().item()), epoch)
    writer.flush()


def train_forcefield(
    train_filename_data: pathlib.Path | str,
    offxml: pathlib.Path | str,
    smee_force_field: smee.TensorForceField,
    topologies: dict[str, smee.TensorTopology],
    n_epochs: int = 1000,
    learning_rate: float = 0.001,
    batch_size: int = 500,
    to_cuda: bool = False,
) -> None:
    """Train force field parameters using molecular energy and force data.

    Optimizes force field parameters by minimizing the mean squared errors
    between predicted and reference energies and forces using gradient descent
    with the
    `Adam optimizer<https://docs.pytorch.org/docs/stable/generated/torch.optim.Adam.html>`_.

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
    batch_size : int, optional
        Number of molecular configurations per batch (default: 500).
    to_cuda : bool, optional
        If True, run training on GPU. If False, use CPU (default: False).

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

    Loss function: L = MSE(E_pred, E_ref) + MSE(F_pred, F_ref)
    where MSE is the mean squared error and energies and forces are weighted equally.

    Examples
    --------
    >>> train_forcefield(
    ...     "data-train",
    ...     "openff-2.2.1.offxml"
    ...     smee_ff,
    ...     topologies,
    ...     n_epochs=500,
    ...     learning_rate=0.0005
    ... )
    """

    train_filename_data = pathlib.Path(train_filename_data)
    logger.info(f"Loading dataset from: {train_filename_data.resolve()}")
    dataset = datasets.Dataset.load_from_disk(train_filename_data)

    # Validate that all SMILES in the dataset have corresponding topologies
    dataset_smiles = set(entry["smiles"] for entry in dataset)
    topology_smiles = set(topologies.keys())
    missing_smiles = dataset_smiles - topology_smiles

    if missing_smiles:
        logger.warning(
            f"Found {len(missing_smiles)} SMILES in dataset without matching topologies. "
            f"These molecules will be excluded from training."
        )
        for smiles in list(missing_smiles)[:5]:  # Log first 5 missing
            logger.debug(f"Missing topology for SMILES: {smiles}")
        if len(missing_smiles) > 5:
            logger.debug(f"... and {len(missing_smiles) - 5} more missing SMILES")

        # Filter dataset to only include molecules with topologies
        original_size = len(dataset)
        dataset = dataset.filter(
            lambda example: example["smiles"] in topology_smiles,
            desc="Filtering molecules with topologies",
        )
        logger.info(
            f"Dataset filtered: {original_size} -> {len(dataset)} molecules "
            f"({len(dataset)/original_size*100:.1f}% retained)"
        )

        if len(dataset) == 0:
            raise ValueError(
                "No molecules remaining after filtering! Check that the topology "
                "dictionary was generated from the same dataset."
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
        "Angles": descent.train.ParameterConfig(
            cols=["k", "angle"],
            scales=scale_mean_parameter_values["Angles"],
            limits=parameter_limits["Angles"],
            include=[],
            exclude=[],
        ),
    }
    trainable = descent.train.Trainable(
        force_field=smee_force_field, parameters=PARAMETERS, attributes={}
    )

    directory = pathlib.Path("my-smee-fit")
    directory.mkdir(exist_ok=True, parents=True)

    trainable_parameters = trainable.to_values()

    logger.info("Start training...")
    with tensorboardX.SummaryWriter(str(directory)) as writer:
        optimizer = torch.optim.Adam(
            [trainable_parameters], lr=learning_rate, amsgrad=True
        )
        dataset_indices = list(range(len(dataset)))

        for i in range(n_epochs):
            ff = trainable.to_force_field(trainable_parameters).to(device_str)

            epoch_loss = torch.zeros(size=(1,), device=device)
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
                # Prepare batch for the target device (CPU or CUDA)
                device_batch = prepare_batch_for_device(batch, device_str)
                true_batch_size = len(
                    dataset
                )  # because loss between batches are combined
                e_ref, e_pred, f_ref, f_pred = descent.targets.energy.predict(
                    device_batch, ff, topologies, "mean"
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
                epoch_loss += batch_loss.detach()
                energy_loss += batch_loss_energy.detach()
                force_loss += batch_loss_force.detach()

            trainable_parameters.grad = grad

            write_metrics(
                epoch=i,
                loss=epoch_loss,
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

        logger.info(f'Saving {directory / "final-force-field.pt"}')
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
                for j, (p, param_unit) in enumerate(
                    zip(parameter_attrs, parameter_units)
                ):
                    setattr(ff_parameter, p, opt_parameters[j] * param_unit)

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
    n_epochs : int, optional
        Number of training epochs (default: 1000).
    learning_rate : float, optional
        Learning rate for
        `Adam optimizer<https://docs.pytorch.org/docs/stable/generated/torch.optim.Adam.html>`_
        (default: 0.001).
    batch_size : int, optional
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
    ...     batch_size=256
    ... )
    """
    filename_data = pathlib.Path(filename_data)
    offxml = pathlib.Path(offxml)
    smee_force_field, topologies = load_smee_outputs(
        filename_ff, filename_topo, to_cuda=to_cuda
    )
    train_forcefield(
        filename_data,
        offxml,
        smee_force_field,
        topologies,
        n_epochs=n_epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        to_cuda=to_cuda,
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
        n_epochs=args.n_epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        to_cuda=args.to_cuda,
    )
