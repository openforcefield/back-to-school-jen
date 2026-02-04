"""
Force field parameter generation and coverage validation pipeline.

Generates SMIRNOFF force field parameters from molecular datasets and validates
coverage across training, testing, and external datasets. Processes Bond and
Angle components with configurable SMIRKS specificity levels.

Functions
---------
load_specificity_config
    Load SMIRKS factory configuration from JSON.
get_components_by_type
    Extract and organize molecular components from dataset.
write_forcefield_file
    Generate enhanced force field with new parameters.
get_qca_smiles_dict
    Retrieve molecular SMILES from QCArchive datasets.
get_train_test_smiles_dict
    Load training/testing molecular splits.
test_coverage
    Validate force field coverage across datasets.
main
    Execute complete pipeline.

Input Data Structure
--------------------
- SMILES JSON: {"train": [smiles], "test": [smiles]}
- HuggingFace Dataset: data-*.arrow files with smiles, coords, energy, forces
- Template force field: .offxml SMIRNOFF format
- Specificity config: JSON with bond_specificities, angle_specificities

Examples
--------
>>> # Command line usage
>>> python make_offxml.py --data-dir ./dataset/ --specificity-json config.json
...     --filename-offxml-out enhanced.offxml --filename-offxml-in template.offxml
...     --filename-test-train-smiles splits.json -vv
"""

import json
import pathlib
import sys
from collections import defaultdict
from typing import Any

import argparse
from loguru import logger
from datasets import load_from_disk

from qcportal import PortalClient

from openff.toolkit import ForceField

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from from_finlay.molecular_classes import (  # noqa: E402
    MMComponent,
    Bond,
    Angle,
)
from from_finlay import process_SMIRKS as ffps  # noqa: E402
from from_finlay import process_mmcomponents as ffpmm  # noqa: E402
from from_finlay.coverage import check_all_components_fully_covered_parallel_chunks  # noqa: E402

logger.remove()


def load_specificity_config(config_file: str | pathlib.Path) -> dict:
    """
    Load specificity configuration from JSON file.

    Parameters
    ----------
    config_file : str | pathlib.Path
        Path to JSON configuration file.

    Returns
    -------
    dict
        Configuration with *_specificities for keys defining the specificity for bonds, angles, etc..

    Examples
    --------
    >>> config = load_specificity_config("config.json")
    >>> "bond_specificities" in config
    True
    """
    logger.info(f"Configuration file contents from {config_file}:")
    with open(config_file, "r") as f:
        content = f.read()
        logger.info(content)
    with open(config_file, "r") as f:
        config = json.load(f)

    return config


def summarize_all_types(
    mm_component_types: dict[int, dict[str, list[MMComponent]]],
) -> None:
    """
    Log statistical summary of component types across specificity levels.

    Parameters
    ----------
    mm_component_types : dict[int, dict[str, list[MMComponent]]]
        Component organization: {specificity_level: {smirks_pattern: [components]}}.

    Examples
    --------
    >>> types = {0: {"[#6:1]-[#6:2]": [bond1, bond2]}}
    >>> summarize_all_types(types)
    # Logs: "Total unique component types at specificity level 0: 1"
    """
    for specificity_num, components_by_type in mm_component_types.items():
        component_type_counts = ffpmm.get_mm_component_type_num(components_by_type)
        logger.info(
            f"Total unique component types at specificity level {specificity_num}: {len(component_type_counts)}"
        )
        logger.info("Most common component types:")
        for smirks, count in sorted(
            component_type_counts.items(), key=lambda item: item[1], reverse=True
        )[:10]:
            logger.info(f"{smirks}: {count}")

    # Flatten the component types for overall statistics
    all_component_type_counts = ffpmm.flatten_mm_component_types(mm_component_types)
    logger.info(
        f"% of component types with count < 5: {sum(1 for count in all_component_type_counts.values() if count < 5) / len(all_component_type_counts) * 100:.2f}%"
    )
    logger.info(
        f"% of component types with count = 1: {sum(1 for count in all_component_type_counts.values() if count == 1) / len(all_component_type_counts) * 100:.2f}%"
    )

    # Print the total number of unique component types across all specificity levels
    total_unique_component_types = len(all_component_type_counts)
    first_component = next(
        iter(next(iter(next(iter(mm_component_types.values())).values())))
    )
    component_type = first_component.__class__.__name__
    logger.info(
        f"\n{'=' * 50}\n"
        f"Total unique component types, {component_type}, across all specificity levels: {total_unique_component_types}"
        f"\n{'=' * 50}"
    )


def get_components_by_type(
    data_dir: str,
    specificity_json: str,
    n_workers: int | None = None,
    cutoff_population: int = 10,
) -> dict[type[MMComponent], dict[int, dict[str, list[MMComponent]]]]:
    """
    Extract and organize molecular components from dataset.

    Parameters
    ----------
    data_dir : str
        Path to HuggingFace dataset directory.
    specificity_json : str
        Path to JSON configuration file.
    n_workers : int
        Number of workers to parallelize processing
    cutoff_population : int, optional
        Minimum number of bonds in that category to keep that value to fit

    Returns
    -------
    dict[type[MMComponent], dict[int, dict[str, list[MMComponent]]]]
        Component organization: {ComponentClass: {level: {smirks: [components]}}}.

    Examples
    --------
    >>> components = get_components_by_type("./dataset/", "config.json")
    >>> len(components[Bond])  # Number of specificity levels
    2
    """
    config = load_specificity_config(specificity_json)
    bond_specs, angle_specs = ffps.create_specificity_factories(config)
    specificities: dict[type[MMComponent], dict[str, Any]] = {}
    if bond_specs is not None:
        specificities[Bond] = bond_specs
    if angle_specs is not None:
        specificities[Angle] = angle_specs
    if not specificities:
        raise ValueError("No specificities have been defined for a new force field")
    specificity_levels_by_component = ffps.create_specificity_levels(specificities)

    logger.info("Getting components by type:")
    dataset = load_from_disk(data_dir)
    components_by_type: dict[
        type[MMComponent], dict[int, dict[str, list[MMComponent]]]
    ] = {}
    component_classes: list[type[MMComponent]] = [Bond, Angle]
    for component_class in component_classes:
        # Skip processing if no specificities are defined for this component
        if not specificity_levels_by_component.get(component_class):
            logger.info(
                f"Skipping {component_class.__name__} - no specificities defined"
            )
            continue

        logger.info(f"\n{'=' * 20}\nProcessing {component_class.__name__}\n{'=' * 20}")

        components = ffpmm.get_all_mm_components(
            dataset,  # type: ignore[type-abstract]
            component_class,  # type: ignore[type-abstract]
            n_workers=n_workers,
        )
        logger.info(f"Found {len(components)} {component_class.__name__}s.")

        class_components_by_type = ffpmm.get_mm_components_by_specificity_by_type(
            components,
            specificity_levels_by_component[component_class],  # type: ignore[type-abstract]
            cutoff_population=cutoff_population,
            n_workers=n_workers,
        )

        if len(components) != sum(
            [
                len(comps)
                for _, smirks_dict in class_components_by_type.items()
                for _, comps in smirks_dict.items()
            ]
        ):
            raise ValueError(
                "Number of components before sorting does not equal the number of components after sorting."
            )

        summarize_all_types(class_components_by_type)
        components_by_type[component_class] = class_components_by_type  # type: ignore[type-abstract]

    return components_by_type


def write_forcefield_file(
    components_by_type: dict[
        type[MMComponent], dict[int, dict[str, list[MMComponent]]]
    ],
    filename_offxml_out: pathlib.Path | str,
    filename_offxml_in: pathlib.Path | str,
    n_workers: int | None = None,
) -> None:
    """
    Generate enhanced force field with additional parameters.

    Parameters
    ----------
    components_by_type : dict[type[MMComponent], dict[int, dict[str, list[MMComponent]]]]
        Output from get_components_by_type().
    filename_offxml_out : pathlib.Path | str
        Output path for enhanced force field (.offxml).
    filename_offxml_in : pathlib.Path | str
        Input template force field path (.offxml).
    n_workers : int, optional
        Number of worker processes. If None, uses all CPU cores.

    Examples
    --------
    >>> components = get_components_by_type("dataset/", "config.json")
    >>> write_forcefield_file(components, "out.offxml", "template.offxml")
    """

    filename_offxml_out = pathlib.Path(filename_offxml_out)
    filename_offxml_in = pathlib.Path(filename_offxml_in)
    logger.info(f"Reading template force field: {filename_offxml_in.resolve()}")
    new_ff = ForceField(str(filename_offxml_in))

    if not components_by_type:
        logger.info(
            "No new component types to add. Using original force field parameters."
        )
    else:
        for component_class, valence_by_type in components_by_type.items():
            logger.info(
                f"\nAdding {component_class.__name__} parameters to force field..."
            )
            new_ff = ffps.add_types_to_ff(
                new_ff,
                valence_by_type,
                component_class,
                None,  # None for bonds and angles
                n_workers=n_workers,
            )

    logger.info(f"Writing new force field: {filename_offxml_out.resolve()}")
    new_ff.to_file(str(filename_offxml_out.resolve()))


def get_qca_smiles_dict(datasets: list[str], dataset_type: str) -> dict[str, list[str]]:
    """
    Retrieve molecular SMILES from QCArchive datasets.

    Parameters
    ----------
    datasets : list[str]
        QCArchive dataset names.
    dataset_type : str
        Dataset type ("optimization", "torsiondrive", "singlepoint").

    Returns
    -------
    dict[str, list[str]]
        Mapping: {dataset_name: [mapped_smiles]}.

    Examples
    --------
    >>> smiles_dict = get_qca_smiles_dict(["OpenFF v1.0"], "optimization")
    >>> len(smiles_dict["OpenFF v1.0"])
    1000
    """

    client = PortalClient("https://api.qcarchive.molssi.org:443/", cache_dir=".")
    smiles_dict: dict[str, list[str]] = {}
    for dataset_name in datasets:
        ds = client.get_dataset(dataset_type, dataset_name)
        smiles_dict[dataset_name] = list(
            set(
                [
                    entry.initial_molecule.extras[
                        "canonical_isomeric_explicit_hydrogen_mapped_smiles"
                    ]
                    for entry in ds.iterate_entries()
                ]
            )
        )
    return smiles_dict


def get_train_test_smiles_dict(
    smiles_file_path: pathlib.Path | str,
) -> dict[str, list[str]]:
    """
    Load train/test split molecular SMILES from JSON file.

    Parameters
    ----------
    smiles_file_path : str
        Path to JSON file with train/test SMILES split.

    Returns
    -------
    dict[str, list[str]]
        Mapping: {"train": [smiles], "test": [smiles]}.

    Examples
    --------
    >>> smiles_dict = get_train_test_smiles_dict("split.json")
    >>> len(smiles_dict["train"])
    8000
    """

    logger.info(f"Reading test/train smiles from {str(smiles_file_path)}")
    with open(smiles_file_path, "r") as f:
        smiles_data = json.load(f)
    logger.info(
        f"In the training and test sets there are {len(smiles_data['train'])} and {len(smiles_data['test'])} SMILES strings respectively."
    )
    return {"train": smiles_data["train"], "test": smiles_data["test"]}


def test_coverage(
    filename_offxml: str, smiles_dict: dict[str, list[str]]
) -> dict[str, dict[str, dict[str, list]]]:
    """
    Test force field coverage on molecular datasets.

    Parameters
    ----------
    filename_offxml : str
        Path to force field XML file.
    smiles_dict : dict[str, list[str]]
        Mapping: {dataset_name: [smiles]}.

    Returns
    -------
    dict[str, dict[str, dict[str, list]]]
        Nested mapping of uncovered components:
        {dataset_name: {component_type: {smiles: [uncovered_indices]}}}.
        - dataset_name: Name of the dataset (e.g., "train", "test")
        - component_type: Type of component (e.g., "Bonds", "Angles")
        - smiles: SMILES string of the molecule with uncovered components
        - uncovered_indices: List of atom indices for uncovered components

    Examples
    --------
    >>> smiles = {"train": ["CCO", "CCC"], "test": ["CNN"]}
    >>> uncovered = test_coverage("ff.offxml", smiles)
    >>> uncovered["train"]["Bonds"]  # Get uncovered bonds in training set
    {"CCO": [(0, 1)]}
    """

    new_ff = ForceField(filename_offxml)
    uncovered_components: dict[str, dict[str, dict[str, list]]] = defaultdict(
        lambda: defaultdict()
    )
    for dataset_name, smiles in smiles_dict.items():
        logger.info(
            f"\nChecking coverage for {dataset_name} dataset with {len(smiles)} SMILES strings..."
        )
        uncovered = check_all_components_fully_covered_parallel_chunks(smiles, new_ff)
        if uncovered:
            component_types = [
                "Bonds",
                "Angles",
            ]  # , "ImproperTorsions", "ProperTorsions"]
            for comp_typ in component_types:
                comp_uncovered = {
                    mol: x[comp_typ]
                    for mol, x in uncovered.items()
                    if comp_typ in x.keys()
                }  # likely redundancy in component types
                logger.info(
                    f"Uncovered {comp_typ}: {sum([len(x) for x in comp_uncovered.values()])} in {len(comp_uncovered)} molecules"
                )
                if comp_uncovered:
                    logger.debug(f"    Uncovered {comp_typ}")
                    for mol, indices in comp_uncovered.items():
                        logger.debug(f"    {mol}: {indices}")

                uncovered_components[dataset_name][comp_typ] = comp_uncovered

    return dict(uncovered_components)


def main(
    data_dir: str,
    specificity_json: str,
    filename_offxml_out: str,
    filename_offxml_in: pathlib.Path | str,
    filename_test_train_smiles: pathlib.Path | str,
    datasets: list[str],
    dataset_type: str,
    n_workers: int | None = None,
    cutoff_population: int = 10,
) -> None:
    """
    Generate force field with custom parameters and test coverage.

    Parameters
    ----------
    data_dir : str
        HuggingFace dataset directory path.
    specificity_json : str
        JSON file with SMIRKS specificity configuration.
    filename_offxml_out : str
        Output force field XML file path.
    filename_offxml_in : pathlib.Path | str
        Template force field file path.
    filename_test_train_smiles : pathlib.Path | str
        Train/test SMILES split JSON file.
    datasets : list[str]
        QCArchive dataset names for testing.
    dataset_type : str
        QCArchive dataset type.
    n_workers : int, optional
        Number of worker processes.
    cutoff_population : int, optional
        Minimum number of bonds in that category to keep that value to fit

    Examples
    --------
    >>> main("data/", "config.json", "out.offxml", "in.offxml",
    ...      "split.json", ["OpenFF v1.0"], "optimization")
    """

    smiles_dict = get_train_test_smiles_dict(filename_test_train_smiles)
    components_by_type = get_components_by_type(
        data_dir,
        specificity_json,
        n_workers=n_workers,
        cutoff_population=cutoff_population,
    )
    write_forcefield_file(
        components_by_type, filename_offxml_out, filename_offxml_in, n_workers=n_workers
    )

    qca_dict = get_qca_smiles_dict(datasets, dataset_type)
    smiles_dict.update(qca_dict)
    _ = test_coverage(filename_offxml_out, smiles_dict)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate force field with custom parameters and test coverage",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    Generate enhanced force field with coverage testing:
        python make_offxml.py \
            --data-dir ./dataset/ \
            --specificity-json config.json \
            --filename-offxml-out enhanced.offxml \
            --filename-offxml-in template.offxml \
            --filename-test-train-smiles splits.json \
            --datasets "OpenFF v1.0" \
            --datasets-type optimization \
            -vv

    Verbosity levels:
        -v    : Warnings only
        -vv   : Info and warnings (recommended)
        -vvv  : Debug, info, and warnings

Pipeline:
    1. Load molecular dataset and SMIRKS configuration
    2. Extract Bond/Angle components with custom specificity
    3. Generate enhanced force field with new parameters
    4. Test coverage on train/test/external datasets
    5. Report coverage statistics and gaps

Output:
    - Enhanced force field (.offxml)
    - Coverage analysis logs
        """,
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="HuggingFace dataset directory path",
    )
    parser.add_argument(
        "--specificity-json",
        type=str,
        required=True,
        help="JSON file with SMIRKS specificity configuration",
    )
    parser.add_argument(
        "--filename-offxml-out",
        type=str,
        required=True,
        help="Output force field XML file path",
    )
    parser.add_argument(
        "--filename-offxml-in",
        type=str,
        required=True,
        help="Template force field file path",
    )
    parser.add_argument(
        "--filename-test-train-smiles",
        type=str,
        required=True,
        help="Train/test SMILES split JSON file",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="*",
        default=[],
        help="QCArchive dataset names for coverage testing",
    )
    parser.add_argument(
        "--datasets-type",
        type=str,
        help="QCArchive dataset type (optimization, singlepoint, etc.)",
    )
    parser.add_argument(
        "-n",
        "--n-workers",
        type=int,
        default=None,
        help="Number of worker processes",
    )
    parser.add_argument(
        "--cutoff-population",
        type=int,
        default=10,
        help="Minimum number of bonds in that category to keep that value to fit",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase verbosity level: -v for WARNING, -vv for INFO, -vvv for DEBUG",
    )
    args = parser.parse_args()

    # Configure logging based on verbosity level
    # Logger was already cleared of default handlers at import time

    if args.verbose == 0:
        # No logging output - keep logger silent
        pass
    elif args.verbose == 1:
        logger.add(sys.stdout, level="WARNING")
    elif args.verbose == 2:
        logger.add(sys.stdout, level="INFO")
    elif args.verbose >= 3:
        logger.add(sys.stdout, level="DEBUG")

    main(
        args.data_dir,
        args.specificity_json,
        args.filename_offxml_out,
        args.filename_offxml_in,
        args.filename_test_train_smiles,
        args.datasets,
        args.datasets_type,
        args.n_workers,
        args.cutoff_population,
    )
