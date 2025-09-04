"""
Force Field Parameter Generation and Coverage Analysis Pipeline.

This module provides a comprehensive pipeline for generating SMIRNOFF force field
parameters from molecular datasets and validating parameter coverage. It processes
molecular mechanics components (bonds and angles) at multiple specificity
levels and creates exhaustive parameter definitions for OpenFF force fields.

Key Functions
-------------
Component Processing:
    - get_components_by_type: Extract and categorize molecular mechanics components
    - summarize_all_types: Statistical analysis of component type distributions

Force Field Generation:
    - write_forcefield_file: Generate new force field with additional parameters
    - main: Complete pipeline from data processing to coverage validation

Data Management:
    - get_qca_smiles_dict: Retrieve molecular data from QCArchive datasets
    - get_train_test_smiles_dict: Load train/test molecular splits

Coverage Analysis:
    - test_coverage: Validate force field parameter coverage across datasets

Workflow Overview
-----------------
1. Load molecular dataset from HuggingFace format
2. Extract molecular mechanics components (bonds, angles)
3. Generate SMIRKS patterns at multiple specificity levels
4. Create new force field with additional parameters
5. Test coverage against training, testing, and external datasets

Examples
--------
Basic usage from command line:

>>> python make_offxml.py \\
...     --filename-offxml-out generated_ff.offxml \\
...     --filename-offxml-in openff-2.0.0.offxml \\
...     --data-dir ./processed_data \\
...     --filename-test-train-smiles splits.json \\
...     --datasets dataset1 dataset2 \\
...     --datasets-type OptimizationDataset

Programmatic usage:

>>> import pathlib
>>> from make_offxml import main
>>> main(
...     filename_offxml_out="output.offxml",
...     filename_offxml_in="openff-2.0.0.offxml",
...     data_dir="./data",
...     filename_test_train_smiles="splits.json",
...     datasets=["spice"],
...     dataset_type="OptimizationDataset"
... )

Notes
-----
This module requires the OpenFF toolkit, QCPortal for dataset access, and
the from_finlay submodules for molecular component processing. The pipeline
uses parallel processing for efficient analysis of large molecular datasets.

The specificity levels are defined in SPECIFICITY_LEVELS_BY_COMPONENT within this
module to determine how general or specific the generated SMIRKS patterns will be.
"""

from typing import cast
import json
import pathlib
from collections import defaultdict

from datasets import Dataset
import argparse
from loguru import logger
from qcportal import PortalClient

from openff.toolkit import ForceField

from from_finlay.molecular_classes import (
    SpecificityLevel,
    MMComponent,
    Bond,
    Angle,
)
from from_finlay import process_SMIRKS as ffps
from from_finlay import process_mmcomponents as ffpmm
from from_finlay.coverage import check_all_components_fully_covered_parallel_chunks

SPECIFICITY_LEVELS_BY_COMPONENT: dict[
    type[MMComponent], dict[int, SpecificityLevel]
] = {
    Bond: {
        0: SpecificityLevel(
            name="Standard",
            get_atom_smirks=ffps.get_atom_smirks_standard,
            get_bond_smirks=ffps.get_bond_smirks_standard,
        ),
    },
    Angle: {
        0: SpecificityLevel(
            name="TerminalWildcard",
            get_atom_smirks=ffps.get_atom_smirks_terminal_wildcard,
            get_bond_smirks=ffps.get_bond_smirks_non_central_bonds_generalised,
        ),
        1: SpecificityLevel(
            name="TerminalHnoH",
            get_atom_smirks=ffps.get_atom_smirks_terminal_h_no_h,
            get_bond_smirks=ffps.get_bond_smirks_non_central_bonds_generalised,
        ),
        2: SpecificityLevel(
            name="Standard",
            get_atom_smirks=ffps.get_atom_smirks_standard,
            get_bond_smirks=ffps.get_bond_smirks_standard,
        ),
    },
}


def summarize_all_types(
    mm_component_types: dict[int, dict[str, list[MMComponent]]],
) -> None:
    """
    Print statistical summary of molecular mechanics component types.

    Analyzes the distribution of component types across specificity levels
    and provides detailed statistics about component frequency, including
    identification of rare and common component types.

    Parameters
    ----------
    mm_component_types : dict[int, dict[str, list[MMComponent]]]
        Hierarchical dictionary containing molecular mechanics components organized
        by specificity level and component type. Structure:
        {specificity_level: {smirks_pattern: [component_instances]}}

    Returns
    -------
    None
        Prints summary statistics to the logger.

    Notes
    -----
    The function provides several key statistics:
    - Total unique component types per specificity level
    - Most common component types (top 10)
    - Percentage of component types with low frequency (< 5 occurrences)
    - Percentage of singleton component types (exactly 1 occurrence)
    - Overall total across all specificity levels

    These statistics help assess the diversity and distribution of molecular
    mechanics components in the dataset, which is crucial for force field
    parameter generation decisions.

    Examples
    --------
    >>> from molecular_classes import Bond, Angle
    >>> # Sample hierarchical component data
    >>> components = {
    ...     0: {
    ...         "[#6:1]-[#6:2]": [Bond(...), Bond(...)],
    ...         "[#6:1]-[#8:2]": [Bond(...)]
    ...     },
    ...     1: {
    ...         "[*:1]-[*:2]": [Bond(...) for _ in range(100)]
    ...     }
    ... }
    >>> summarize_all_types(components)
    INFO - Total unique component types at specificity level 0: 2
    INFO - Most common component types:
    INFO - [#6:1]-[#6:2]: 2
    INFO - [#6:1]-[#8:2]: 1
    ...
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
    logger.info(
        f"Total unique component types across all specificity levels: {total_unique_component_types}"
    )


def get_components_by_type(data_dir: pathlib.Path | str):
    """
    Extract and categorize molecular mechanics components from dataset.

    Processes molecular data to extract bonds, angles, and other molecular
    mechanics components, organizing them by component type and specificity
    level for force field parameter generation.

    Parameters
    ----------
    data_dir : pathlib.Path or str
        Path to directory containing HuggingFace dataset with molecular data.
        Dataset must contain a 'smiles' column with molecular SMILES strings.

    Returns
    -------
    dict[type[MMComponent], dict[int, dict[str, list[MMComponent]]]]
        Nested dictionary with structure:
        {ComponentClass: {specificity_level: {smirks_pattern: [components]}}}
        where ComponentClass is Bond, Angle, etc.

    Notes
    -----
    This function processes each component class (Bond, Angle) sequentially:
    1. Extracts all components of that type from the dataset
    2. Categorizes them by specificity levels defined in SPECIFICITY_LEVELS_BY_COMPONENT
    3. Applies population cutoff filtering (minimum 10 occurrences)
    4. Provides detailed logging of component statistics

    The specificity levels determine how general or specific the SMIRKS patterns
    will be, with higher levels typically being more specific.

    Examples
    --------
    >>> import pathlib
    >>> data_path = pathlib.Path("./molecular_dataset")
    >>> components = get_components_by_type(data_path)
    >>> print(f"Found {len(components)} component types")
    Found 2 component types

    >>> # Access specific component data
    >>> bond_components = components[Bond]
    >>> print(f"Bond specificity levels: {list(bond_components.keys())}")
    Bond specificity levels: [0]

    >>> # Get components at specific level
    >>> level_0_bonds = bond_components[0]
    >>> print(f"Found {len(level_0_bonds)} bond types at level 0")
    Found 150 bond types at level 0
    """
    # Also get the SMILES after filtering to make sure all types are trained

    data_dir = pathlib.Path(data_dir)
    logger.info(
        f"Retrieving SMILES strings from HuggingFace dataset: {data_dir.resolve()}"
    )
    dataset_smiles = list(Dataset.load_from_disk(data_dir)["smiles"])

    logger.info("Getting components by type:")
    components_by_type: dict[
        type[MMComponent], dict[int, dict[str, list[MMComponent]]]
    ] = {}
    for component_class in [Bond, Angle]:
        logger.info(f"\n{'=' * 20}\nProcessing {component_class.__name__}\n{'=' * 20}")

        components = ffpmm.get_all_mm_components(dataset_smiles, component_class)  # type: ignore[type-abstract]
        logger.info(f"Found {len(components)} {component_class.__name__}s.")

        class_components_by_type = ffpmm.get_mm_components_by_specificity_by_type(
            components,
            SPECIFICITY_LEVELS_BY_COMPONENT[component_class],  # type: ignore[type-abstract]
            cutoff_population=10,
        )

        summarize_all_types(class_components_by_type)
        components_by_type[component_class] = class_components_by_type  # type: ignore[type-abstract]

    return components_by_type


def write_forcefield_file(
    components_by_type,
    filename_offxml_out: pathlib.Path | str,
    filename_offxml_in: pathlib.Path | str,
) -> None:
    """
    Generate new force field file with additional molecular mechanics parameters.

    Takes a template force field and adds new bond and angle parameters derived
    from molecular dataset analysis. The new parameters are generated from the
    component analysis and SMIRKS pattern generation pipeline.

    Parameters
    ----------
    components_by_type : dict
        Dictionary containing molecular mechanics components organized by type
        and specificity level, as returned by get_components_by_type().
    filename_offxml_out : pathlib.Path or str
        Output path for the new force field file. Will be created or overwritten.
    filename_offxml_in : pathlib.Path or str
        Input template force field file path. Should be a valid SMIRNOFF force field.

    Returns
    -------
    None
        Writes the new force field to the specified output file.

    """

    filename_offxml_out = pathlib.Path(filename_offxml_out)
    filename_offxml_in = pathlib.Path(filename_offxml_in)
    logger.info(f"Reading template force field: {filename_offxml_in.resolve()}")
    new_ff = ForceField(filename_offxml_in)
    for component_class, angles_by_type in components_by_type.items():
        logger.info(f"\nAdding {component_class.__name__} parameters to force field...")
        new_ff = ffps.add_types_to_ff(
            new_ff,
            angles_by_type,
            component_class,
            None,  # None for bonds and angles
        )

    logger.info(f"Writing new force field: {filename_offxml_out.resolve()}")
    new_ff.to_file(str(filename_offxml_out.resolve()))


def get_qca_smiles_dict(datasets: list[str], dataset_type: str) -> dict[str, set[str]]:
    """
    Retrieve molecular SMILES from QCArchive datasets.

    Connects to the QCArchive portal and downloads molecular data from specified
    datasets, extracting mapped SMILES strings for coverage analysis.

    Parameters
    ----------
    datasets : list[str]
        List of dataset names to retrieve from QCArchive.
    dataset_type : str
        Type of QCArchive dataset (e.g., "OptimizationDataset", "ReactionDataset").

    Returns
    -------
    dict[str, set[str]]
        Dictionary mapping dataset names to sets of mapped SMILES strings.
        Structure: {dataset_name: {smiles1, smiles2, ...}}

    Notes
    -----
    The function specifically extracts "mapped_smiles" from the entry extras,
    which provides atom-mapped molecular representations needed for force
    field parameter assignment and coverage analysis.

    """

    client = PortalClient("https://api.qcarchive.molssi.org:443/", cache_dir=".")
    smiles_dict: defaultdict = defaultdict()
    for dataset_name in datasets:
        ds = client.get_dataset(dataset_type, dataset_name)
        smiles_dict[dataset_name] = set(
            [entry["extras"]["mapped_smiles"] for entry in ds.get_entries]
        )
    return smiles_dict


def get_train_test_smiles_dict(filename: pathlib.Path | str) -> dict[str, list[str]]:
    """
    Load training and testing molecular data splits from JSON file.

    Reads a JSON file containing pre-defined train/test splits of molecular
    SMILES strings for consistent dataset partitioning across experiments.

    Parameters
    ----------
    filename : pathlib.Path or str
        Path to JSON file containing train/test splits. Expected format:
        {"train": [smiles_list], "test": [smiles_list]}

    Returns
    -------
    dict[str, list[str]]
        Dictionary with "Train" and "Test" keys mapping to lists of SMILES strings.
        Structure: {"Train": [smiles1, smiles2, ...], "Test": [smiles1, smiles2, ...]}

    """
    with open(filename, "r") as f:
        smiles_data = json.load(f)
    return {"Train": smiles_data["train"], "Test": smiles_data["test"]}


def test_coverage(
    filename_offxml: str, smiles_dict: dict[str, list[str] | set[str]]
) -> None:
    """
    Test force field parameter coverage across multiple molecular datasets.

    Validates that the generated force field can parameterize molecules from
    training, testing, and external datasets, identifying any molecules or
    components that lack necessary parameters.

    Parameters
    ----------
    filename_offxml : str
        Path to the force field file to test.
    smiles_dict : dict[str, list[str] | set[str]]
        Dictionary mapping dataset names to collections of SMILES strings.
        Each collection represents molecules to test for coverage.

    Returns
    -------
    None
        Logs coverage analysis results for each dataset.

    Notes
    -----
    This function performs comprehensive coverage analysis by:
    1. Loading the specified force field
    2. Testing parameterization for each molecule in each dataset
    3. Identifying uncovered molecular mechanics components
    4. Logging detailed statistics about coverage gaps

    Uncovered components indicate molecular environments not present in the
    force field, which may require additional parameter development or
    suggest limitations in the training dataset.

    Examples
    --------
    >>> # Test coverage on multiple datasets
    >>> datasets = {
    ...     "Train": ["CC", "CCO", "CCC"],
    ...     "Test": ["CCCC", "CCN"],
    ...     "External": {"[C:1][O:2]", "[N:1][C:2]"}
    ... }
    >>> test_coverage("enhanced_ff.offxml", datasets)
    INFO - Checking coverage for Train dataset...
    INFO - Found 0 uncovered components in 0 component types
    INFO - Checking coverage for Test dataset...
    INFO - Found 2 uncovered components in 1 component types
    ...

    >>> # Interpret results
    >>> # - Train: Full coverage (expected for training data)
    >>> # - Test: Some gaps (identifies generalization limitations)
    >>> # - External: Coverage assessment for independent validation
    """

    new_ff = ForceField(filename_offxml)

    for dataset_name, smiles in smiles_dict.items():
        logger.info(f"\nChecking coverage for {dataset_name} dataset...")
        # Convert to list if it's a set
        smiles_list = list(smiles) if isinstance(smiles, set) else smiles
        uncovered = check_all_components_fully_covered_parallel_chunks(
            smiles_list, new_ff
        )
        if uncovered:
            total_uncovered = sum(len(v) for v in uncovered.values())
            logger.debug([v.keys() for v in uncovered.values()])
            logger.debug(uncovered)
            logger.info(
                f"Found {total_uncovered} uncovered components in {len(uncovered)} component types:"
            )


def main(
    filename_offxml_out: str,
    filename_offxml_in: pathlib.Path | str,
    data_dir: pathlib.Path | str,
    filename_test_train_smiles: pathlib.Path | str,
    datasets: list[str],
    dataset_type: str,
) -> None:
    """
    Execute complete force field parameter generation and validation pipeline.

    Orchestrates the entire workflow from molecular dataset processing through
    force field generation to coverage validation across multiple datasets.
    This is the main entry point for the force field enhancement pipeline.

    Parameters
    ----------
    filename_offxml_out : str
        Output path for the enhanced force field file (.offxml format).
    filename_offxml_in : pathlib.Path or str
        Input template force field file path. Should be a valid SMIRNOFF force field.
    data_dir : pathlib.Path or str
        Directory containing HuggingFace dataset with molecular training data.
    filename_test_train_smiles : pathlib.Path or str
        JSON file containing train/test molecular splits for validation.
    datasets : list[str]
        List of QCArchive dataset names for additional coverage testing.
    dataset_type : str
        Type of QCArchive datasets (e.g., "OptimizationDataset").

    Returns
    -------
    None
        Generates force field file and logs comprehensive analysis results.

    Examples
    --------
    >>> # Complete pipeline execution
    >>> main(
    ...     filename_offxml_out="enhanced_sage.offxml",
    ...     filename_offxml_in="openff-2.0.0.offxml",
    ...     data_dir="./processed_spice_data",
    ...     filename_test_train_smiles="./splits/train_test.json",
    ...     datasets=["SPICE-1.1.4", "OpenFF-benchmark-ligand-fragments-v1.0"],
    ...     dataset_type="OptimizationDataset"
    ... )

    Expected log output:
    >>> # INFO - Processing Bond
    >>> # INFO - Found 1250 Bonds.
    >>> # INFO - Total unique component types at specificity level 0: 85
    >>> # INFO - Processing Angle
    >>> # INFO - Found 2100 Angles.
    >>> # INFO - Writing new force field: enhanced_sage.offxml
    >>> # INFO - Checking coverage for Train dataset...
    >>> # INFO - Checking coverage for SPICE-1.1.4 dataset...

    Pipeline Integration:
    >>> # This function can be called from command line via:
    >>> # python make_offxml.py --filename-offxml-out enhanced.offxml \\
    >>> #                       --filename-offxml-in sage.offxml \\
    >>> #                       --data-dir ./data \\
    >>> #                       --filename-test-train-smiles splits.json \\
    >>> #                       --datasets dataset1 dataset2 \\
    >>> #                       --datasets-type OptimizationDataset
    """

    components_by_type = get_components_by_type(data_dir)
    write_forcefield_file(components_by_type, filename_offxml_out, filename_offxml_in)

    smiles_dict = get_train_test_smiles_dict(filename_test_train_smiles)
    qca_dict = get_qca_smiles_dict(datasets, dataset_type)
    # Convert sets to lists for consistency
    for key, value in qca_dict.items():
        smiles_dict[key] = list(value)
    test_coverage(
        filename_offxml_out, cast(dict[str, list[str] | set[str]], smiles_dict)
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare force field and topology information for training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    Basic force field enhancement from local dataset:
        python make_offxml.py \
            --filename-offxml-out comprehensive_ff.offxml \
            --filename-offxml-in sage-2.1.0.offxml \
            --data-dir /path/to/molecular_dataset \
            --filename-test-train-smiles molecular_splits.json \
            --datasets "OpenFF Additional Generated ChEMBL Optimizations v4.0" "OpenFF Additional Generated ChEMBL Optimizations v4.0" \
            --datasets-type singlepoint

Pipeline Overview:
    1. Loads molecular dataset from HuggingFace format
    2. Extracts bonds and angles with SMIRKS pattern generation
    3. Creates enhanced force field with new parameters
    4. Validates coverage against train/test/external datasets
    5. Reports statistics and missing parameter analysis

Output:
    - Enhanced force field file (.offxml format)
    - Comprehensive logging of component analysis and coverage statistics
        """,
    )
    parser.add_argument(
        "--filename-offxml-out",
        type=str,
        required=True,
        help="Output filename *.offxml",
    )
    parser.add_argument(
        "--filename-offxml-in",
        type=str,
        required=True,
        help="Input filename *.offxml",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Directory to HuggingFace structured data from which to create new bond and angle types.",
    )
    parser.add_argument(
        "--filename-test-train-smiles",
        type=str,
        required=True,
        help="Path and filename to test/train smiles .json file",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="*",
        default=[],
        help="List of QCArchive dataset names for additional coverage validation (optional)",
    )
    parser.add_argument(
        "--datasets-type",
        type=str,
        help="Type of QCArchive dataset (e.g., 'OptimizationDataset', 'ReactionDataset'). Required if --datasets is provided.",
    )
    args = parser.parse_args()
    main(
        args.filename_offxml_out,
        args.filename_offxml_in,
        args.data_dir,
        args.filename_test_train_smiles,
        args.datasets,
        args.datasets_type,
    )
