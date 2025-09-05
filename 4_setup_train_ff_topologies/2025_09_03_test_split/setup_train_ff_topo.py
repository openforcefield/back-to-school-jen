"""Prepare force field and topology information for training.

This script converts molecular datasets and force fields into SMEE format for
efficient machine learning optimization workflows.

Note that outputs may be saved as .pkl or .json files, where the latter is only
for debugging purposes. During the serialization process in creating a .json file
the objects are saved as strings which is not a comprehensive record. This script
should be updated with a more thorough writer for a human readable output if the
json output is meant to be used for force field fitting.

Input Requirements
------------------
Dataset Directory (--data-dir):
    HuggingFace datasets format with:
    - dataset_info.json : Dataset metadata and column schemas
    - state.json : Dataset state information
    - data-*.arrow : Apache Arrow files containing:
        - smiles (string) : SMILES molecular representations
        - coords (float) : Flattened 3D coordinates [x1,y1,z1,x2,y2,z2,...]
        - energy (float) : Total molecular energies
        - forces (float) : Force vectors [fx1,fy1,fz1,fx2,fy2,fz2,...]

Force Field (--offxml):
    OpenFF force field in XML format (.offxml) with parameter definitions

Output Files
------------
Creates in output directory:
- smee_force_field.{pkl|json} : SMEE TensorForceField for optimization
- smee_topology_dict.{pkl|json} : Dict[SMILES, TensorTopology] mappings
- smee_metadata.json : Processing metadata and molecule counts

Examples
--------
Basic usage:
$ python setup_train_ff_topo.py --data-dir ./data-train --offxml openff-2.2.1.offxml

With custom parameters:
$ python setup_train_ff_topo.py --data-dir ./data-train --offxml openff-2.2.1.offxml \\
    --device cuda --file-format json
"""

import pathlib
import dataclasses
import json
import pickle
from typing import Any, Literal
from datetime import datetime

import argparse
from loguru import logger
from tqdm import tqdm
import smee
import smee.converters
import datasets
import torch

from openff.toolkit import Molecule, ForceField


def validate_molecular_dataset(
    dataset: datasets.Dataset, allow_undefined_stereo: bool = True
) -> datasets.Dataset:
    """Validate and filter molecules in a HuggingFace dataset.

    Validates each molecule in the dataset by attempting to create OpenFF Molecule
    objects from SMILES strings. Invalid molecules that cannot be parsed or have
    zero atoms are filtered out.

    Parameters
    ----------
    dataset : datasets.Dataset
        HuggingFace dataset with 'smiles' column.
    allow_undefined_stereo : bool, optional
        Whether to allow undefined stereochemistry. Default is True.

    Returns
    -------
    datasets.Dataset
        Filtered dataset containing only valid molecules.

    Examples
    --------
    >>> dataset = datasets.Dataset.from_dict({'smiles': ['CCO', 'invalid', 'C1=CC=CC=C1']})
    >>> valid_dataset = validate_molecular_dataset(dataset)
    """
    logger.info("Validating molecules...")
    valid_indices = []
    invalid_count = 0

    for i, entry in enumerate(dataset):
        try:
            smiles: str = entry["smiles"]  # type: ignore[index]
            mol = Molecule.from_mapped_smiles(
                smiles, allow_undefined_stereo=allow_undefined_stereo
            )

            # Basic validation - check if molecule can be created and has atoms
            if mol.n_atoms > 0:
                valid_indices.append(i)
            else:
                logger.warning(f"Molecule at index {i} has zero atoms: {smiles}")
                invalid_count += 1

        except Exception as e:
            logger.warning(f"Invalid molecule at index {i}: {e}")
            invalid_count += 1

    if invalid_count > 0:
        logger.info(
            f"Filtered {invalid_count} invalid molecules out of {len(dataset)} total"
        )
        filtered_dataset = dataset.select(valid_indices)
    else:
        logger.info("All molecules passed validation")
        filtered_dataset = dataset

    return filtered_dataset


def prepare_to_train(
    dataset: datasets.Dataset,
    offxml: pathlib.Path | str,
    device: str | None = None,
    precision: Literal["single", "double"] = "single",
    validate_molecules: bool = True,
) -> tuple[smee.TensorForceField, dict[str, smee.TensorTopology]]:
    """Convert molecular dataset and force field to SMEE format for training.

    Processes a dataset and returns SMEE objects for training workflows.

    Parameters
    ----------
    dataset : datasets.Dataset
        HuggingFace dataset with molecular data columns:
        - 'smiles' : str, SMILES molecular representations
        - 'energy' : torch.Tensor, molecular energies
        - 'coords' : torch.Tensor, 3D coordinates
        - 'forces' : torch.Tensor, force vectors
    offxml : pathlib.Path or str
        Path to OpenFF force field XML file (.offxml format).
    device : str or None, optional
        Target device for tensors ('cpu', 'cuda', or None for auto-detect).
        Default is None.
    precision : {'single', 'double'}, optional
        Tensor precision ('single' for float32, 'double' for float64).
        Default is 'single'.
    validate_molecules : bool, optional
        Whether to validate molecular structures before processing.
        Default is True.

    Returns
    -------
    smee_force_field : smee.TensorForceField
        SMEE force field tensor object ready for optimization.
    topologies : dict[str, smee.TensorTopology]
        Dictionary mapping SMILES to SMEE topology tensors.

    Notes
    -----
    Failed molecules are logged and skipped. Uses `smee.converters.convert_interchange` for tensor conversion.

    Examples
    --------
    >>> dataset = load_dataset("./data-train")
    >>> smee_ff, topologies = prepare_to_train(dataset, "openff-2.2.1.offxml")
    >>> len(topologies)
    1000

    >>> # With device and precision
    >>> smee_ff, topologies = prepare_to_train(
    ...     dataset, "openff-2.2.1.offxml", device="cuda", precision="double"
    ... )
    """

    # Get starting forcefield
    offxml = pathlib.Path(offxml)
    logger.info(f"Loading force field: {offxml.resolve()}")
    starting_ff = ForceField(offxml)

    # Optional molecule validation
    if validate_molecules:
        dataset = validate_molecular_dataset(dataset, allow_undefined_stereo=True)

    total_molecules = len(dataset)
    logger.info(f"Processing {total_molecules} molecules.")

    # Process molecules and create interchanges
    logger.info("Creating interchanges...")
    all_smiles = []
    all_interchanges = []
    failed_molecules = 0

    for entry in tqdm(dataset, desc="Creating interchanges"):
        smiles: str = ""
        try:
            smiles = entry["smiles"]  # type: ignore[index]
            mol = Molecule.from_mapped_smiles(smiles, allow_undefined_stereo=True)
            interchange = starting_ff.create_interchange(mol.to_topology())

            all_smiles.append(smiles)
            all_interchanges.append(interchange)

        except Exception as e:
            failed_molecules += 1
            logger.warning(f"Failed to process molecule '{smiles}': {e}")
            continue

    if failed_molecules > 0:
        logger.warning(f"Failed to process {failed_molecules} molecules")

    logger.info("Prepare SMEE data structures...")
    smee_force_field, smee_topologies = smee.converters.convert_interchange(
        all_interchanges
    )

    # Apply device and precision settings
    logger.info(f"Converting to device={device}, precision={precision}")
    options = {}
    if device is not None:
        options["device"] = device
    if precision is not None:
        options["precision"] = precision
    if options:
        smee_force_field = smee_force_field.to(**options)
        smee_topologies = [topo.to(**options) for topo in smee_topologies]

    topologies = dict(zip(all_smiles, smee_topologies))

    return smee_force_field, topologies


def save_smee_output(
    smee_force_field: smee.TensorForceField,
    topologies: dict[str, smee.TensorTopology],
    file_format: Literal["pkl", "json"] = "pkl",
    output_dir: pathlib.Path | str | None = None,
    overwrite: bool = True,
) -> None:
    """Save SMEE objects to disk for training pipelines.

    Serializes SMEE force field and topology objects. Note that .json output
    is meant for debugging purposes as objects are lost during serialization.

    Parameters
    ----------
    smee_force_field : smee.TensorForceField
        SMEE force field tensor object containing parameters and potentials
        ready for optimization workflows.
    topologies : dict[str, smee.TensorTopology]
        Dictionary mapping SMILES to SMEE topology tensors.
    file_format : {'pkl', 'json'}, optional
        Serialization format ('pkl' for binary, 'json' for text).
        Default is 'pkl'. JSON is for debugging only as information is lost
        during serialization.
    output_dir : pathlib.Path, str, or None, optional
        Output directory (None uses current directory). Default is None.
    overwrite : bool, optional
        Whether to overwrite existing files. Default is True.

    Returns
    -------
    None

    Notes
    -----
    Creates files:
    - smee_force_field.{pkl|json}
    - smee_topology_dict.{pkl|json}
    - smee_metadata.json

    JSON format converts tensors to lists. Pickle preserves exact state.

    Examples
    --------
    >>> save_smee_output(smee_ff, topologies, output_dir="./models")

    >>> # JSON format for inspection and debugging
    >>> save_smee_output(
    ...     smee_ff, topologies,
    ...     file_format="json",
    ...     output_dir="./debug"
    ... )
    # Creates human-readable JSON files

    >>> # Load saved pickle files in training code
    >>> import pickle
    >>> with open("./models/smee_force_field.pkl", "rb") as f:
    ...     loaded_ff = pickle.load(f)
    >>> with open("./models/smee_topology_dict.pkl", "rb") as f:
    ...     loaded_topologies = pickle.load(f)
    """

    # Set up output directory
    if output_dir is None:
        output_dir = pathlib.Path.cwd()
    else:
        output_dir = pathlib.Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    def serialize_to_list_or_str(obj: Any) -> Any:
        """Custom JSON encoder for torch tensors and other non-serializable objects."""
        if isinstance(obj, torch.Tensor):
            return obj.tolist()
        # For other non-JSON serializable objects, convert to string
        return str(obj)

    def get_filename(name: str, extension: str) -> pathlib.Path | None:
        """Generate filename and check for overwrites."""
        filename = output_dir / f"smee_{name}.{extension}"
        if not overwrite and filename.exists():
            logger.warning(f"File {filename} exists and overwrite=False, skipping")
            return None
        return filename

    # Save force field
    logger.info("Saving SMEE force field...")
    if file_format == "pkl":
        filename = get_filename("force_field", "pkl")
        if filename:
            with open(filename, "wb") as pkl_file:
                pickle.dump(smee_force_field, pkl_file)
    elif file_format == "json":
        filename = get_filename("force_field", "json")
        if filename:
            try:
                ff_dict = dataclasses.asdict(smee_force_field)
                with open(filename, "w") as json_file:
                    json.dump(
                        ff_dict, json_file, indent=2, default=serialize_to_list_or_str
                    )
            except Exception as e:
                logger.warning(f"Could not save JSON format for force field: {e}")
                filename = None
    else:
        raise ValueError(
            f"Unsupported file format: {file_format}, must be 'pkl' or 'json'"
        )

    if filename:
        logger.info(f"Saved {filename}")

    # Save topologies
    logger.info("Saving SMEE topologies...")
    if file_format == "pkl":
        filename = get_filename("topology_dict", "pkl")
        if filename:
            with open(filename, "wb") as pkl_file:
                pickle.dump(topologies, pkl_file)
    elif file_format == "json":
        filename = get_filename("topology_dict", "json")
        if filename:
            try:
                # Convert all topologies to dict format
                topologies_dict = {}
                for smiles, topology in topologies.items():
                    topologies_dict[smiles] = dataclasses.asdict(topology)

                with open(filename, "w") as json_file:
                    json.dump(
                        topologies_dict,
                        json_file,
                        indent=2,
                        default=serialize_to_list_or_str,
                    )
            except Exception as e:
                logger.warning(f"Could not save JSON format for topologies: {e}")
                filename = None

    if filename:
        logger.info(f"Saved {filename}")

    metadata = {
        "smiles_list": list(topologies.keys()),
        "topology_count": len(topologies),
        "force_field_potentials": len(smee_force_field.potentials)
        if hasattr(smee_force_field, "potentials")
        else 0,
        "file_format": file_format,
        "creation_timestamp": datetime.now().isoformat(),
    }

    metadata_filename = get_filename("metadata", "json")
    if metadata_filename:
        with open(metadata_filename, "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Saved {metadata_filename}")

    logger.info(f"Successfully saved SMEE objects for {len(topologies)} molecules")


def load_dataset(data_dir: pathlib.Path | str) -> datasets.Dataset:
    """Load HuggingFace dataset with torch format enabled.

    Parameters
    ----------
    data_dir : pathlib.Path or str
        Path to directory containing HuggingFace dataset files.

    Returns
    -------
    datasets.Dataset
        Dataset with torch format for 'energy', 'coords', 'forces' columns.

    Notes
    -----
    Enables PyTorch tensor access for numerical columns while preserving
    other column types.

    Examples
    --------
    >>> dataset = load_dataset("./data-train")
    >>> print(f"Loaded {len(dataset)} molecules")
    Loaded 1000 molecules

    >>> # Access torch tensors directly
    >>> energy = dataset[0]['energy']  # Returns torch.Tensor
    >>> coords = dataset[0]['coords']  # Returns torch.Tensor
    >>> smiles = dataset[0]['smiles']  # Returns string
    """
    # Get Dataset
    train_filename_data = pathlib.Path(data_dir)
    logger.info(f"Loading dataset: {train_filename_data.resolve()}")
    dataset = datasets.Dataset.load_from_disk(train_filename_data)
    dataset.set_format(
        "torch", columns=["energy", "coords", "forces"], output_all_columns=True
    )

    return dataset


def main(
    filename_data: pathlib.Path | str,
    offxml: pathlib.Path | str,
    validate_molecules: bool = True,
    precision: Literal["single", "double"] = "single",
    device: Literal["cpu", "cuda"] | None = None,
    file_format: Literal["pkl", "json"] = "pkl",
    output_dir: pathlib.Path | str | None = None,
) -> None:
    """Main pipeline for SMEE force field preparation.

    Parameters
    ----------
    filename_data : pathlib.Path or str
        Path to HuggingFace dataset directory.
    offxml : pathlib.Path or str
        Path to OpenFF force field XML file.
    validate_molecules : bool, optional
        Whether to validate molecules before processing. Default is True.
    precision : {'single', 'double'}, optional
        Tensor precision. Default is 'single'.
    device : {'cpu', 'cuda'} or None, optional
        Target device for tensors. Default is None (auto-detect).
    file_format : {'pkl', 'json'}, optional
        Output serialization format. Default is 'pkl'.
    output_dir : pathlib.Path, str, or None, optional
        Output directory. Default is None (current directory).

    Returns
    -------
    None

    Notes
    -----
    Converts molecular dataset and force field to SMEE format.

    Examples
    --------
    >>> main("./data-train", "openff-2.2.1.offxml")

    >>> main("./data-train", "openff-2.2.1.offxml", device="cuda")
    """
    filename_data = pathlib.Path(filename_data)
    offxml = pathlib.Path(offxml)

    # Load dataset
    logger.info(f"Loading dataset: {filename_data.resolve()}")
    dataset = load_dataset(filename_data)
    total_molecules = len(dataset)

    logger.info(f"Processing all {total_molecules} molecules")

    smee_force_field, topologies = prepare_to_train(
        dataset,
        offxml,
        device=device,
        precision=precision,
        validate_molecules=validate_molecules,
    )

    save_smee_output(
        smee_force_field,
        topologies,
        file_format=file_format,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare force field and topology information for training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage
    python setup_train_ff_topo.py --data-dir /path/to/data --offxml openff-2.2.1.offxml

    # With custom parameters
    python setup_train_ff_topo.py --data-dir /path/to/data --offxml openff-2.2.1.offxml \\
        --precision double --device cuda --file-format json --output-dir ./results
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
        "--validate-molecules",
        action="store_true",
        default=True,
        help="Validate molecule structures before processing (default: True)",
    )
    parser.add_argument(
        "--no-validate-molecules",
        dest="validate_molecules",
        action="store_false",
        help="Skip molecule validation",
    )
    parser.add_argument(
        "--precision",
        type=str,
        choices=["single", "double"],
        default="single",
        help="Tensor precision (default: single)",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda"],
        default=None,
        help="Target device for tensors (default: auto-detect)",
    )
    parser.add_argument(
        "--file-format",
        type=str,
        choices=["pkl", "json"],
        default="pkl",
        help="Output file format (default: pkl). JSON format is for debugging only, as information is lost during serialization.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save output files (default: current directory)",
    )
    args = parser.parse_args()
    main(
        args.data_dir,
        args.offxml,
        validate_molecules=args.validate_molecules,
        precision=args.precision,
        device=args.device,
        file_format=args.file_format,
        output_dir=args.output_dir,
    )
