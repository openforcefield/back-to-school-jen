"""
Force field parameter generation and coverage analysis pipeline.

This module generates SMIRNOFF force field parameters from molecular datasets
and validates parameter coverage. It processes Bond and Angle components according to
types that are defined by SMIRKS strings. These types can range in
specificity level, creating comprehensive parameter definitions.

Input Requirements
------------------
- JSON file with train/test SMILES splits: {"train": [smiles], "test": [smiles]}
- Template OFFXML force field file
- QCArchive dataset names for external validation

Output Structure
----------------
- Enhanced OFFXML force field with additional bond/angle parameters
- Coverage analysis logs for train/test/external datasets

Workflow
--------
1. Load molecular SMILES from train/test splits
2. Extract molecular mechanics components (bonds, angles)
3. Generate SMIRKS patterns at specified levels
4. Create enhanced force field with new parameters
5. Test coverage against training, testing, and external datasets

Examples
--------
Command line usage:
    python make_offxml.py \\
        --filename-offxml-out enhanced.offxml \\
        --filename-offxml-in openff-2.0.0.offxml \\
        --filename-test-train-smiles splits.json \\
        --datasets dataset1 dataset2 --datasets-type optimization

Programmatic usage:
    from make_offxml import main
    main("output.offxml", "input.offxml", "splits.json", ["spice"], "optimization")
"""

import json
import pathlib

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

    Analyzes component type distributions across specificity levels, providing
    statistics on frequency, rare types, and overall diversity.

    Parameters
    ----------
    mm_component_types : dict[int, dict[str, list[MMComponent]]]
        Components organized by specificity level and SMIRKS pattern.
        Structure: {specificity_level: {smirks_pattern: [component_instances]}}
        where components are Bond or Angle.

    Returns
    -------
    None
        Logs statistics including total types, most common patterns,
        and percentage of low-frequency types.

    Notes
    -----
    "Types" are molecular mechanics component categories defined by SMIRKS
    strings that can range in specificity. Provides key statistics for
    force field parameter generation decisions:
    - Total unique types per specificity level
    - Top 10 most common component types
    - Percentage with count < 5 and singleton types
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


def get_components_by_type(
    smiles_list: list[str],
) -> dict[type[MMComponent], dict[int, dict[str, list[MMComponent]]]]:
    """
    Extract and categorize molecular mechanics components from SMILES list.

    Processes molecular data to extract bonds and angles, organizing them by
    component type and specificity level for force field parameter generation.

    Parameters
    ----------
    smiles_list : list[str]
        List of SMILES strings to process.

    Returns
    -------
    dict[type[MMComponent], dict[int, dict[str, list[MMComponent]]]]
        Components organized by class, specificity level, and SMIRKS pattern.
        Structure: {ComponentClass: {level: {smirks: [components]}}}
        where ComponentClass is Bond or Angle.

    Notes
    -----
    Components are molecular mechanics elements; this function processes Bond
    and Angle components. "Types" are categories defined by SMIRKS strings that
    can range in specificity. Processes with population cutoff of 10 occurrences.
    Specificity levels determine SMIRKS pattern generality (higher = more specific).
    """

    logger.info("Getting components by type:")
    components_by_type: dict[
        type[MMComponent], dict[int, dict[str, list[MMComponent]]]
    ] = {}
    for component_class in [Bond, Angle]:
        logger.info(f"\n{'=' * 20}\nProcessing {component_class.__name__}\n{'=' * 20}")

        components = ffpmm.get_all_mm_components(smiles_list, component_class)  # type: ignore[type-abstract]
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
    components_by_type: dict[
        type[MMComponent], dict[int, dict[str, list[MMComponent]]]
    ],
    filename_offxml_out: pathlib.Path | str,
    filename_offxml_in: pathlib.Path | str,
) -> None:
    """
    Generate new force field file with additional parameters.

    Takes a template force field and adds new bond and angle parameters derived
    from molecular dataset analysis.

    Parameters
    ----------
    components_by_type : dict[type[MMComponent], dict[int, dict[str, list[MMComponent]]]]
        Components organized by class, specificity level, and SMIRKS pattern.
        Output from get_components_by_type().
    filename_offxml_out : pathlib.Path | str
        Output path for the new force field file (.offxml format).
    filename_offxml_in : pathlib.Path | str
        Input template force field file path (valid SMIRNOFF format).

    Returns
    -------
    None
        Writes enhanced force field to specified output file.
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


def get_qca_smiles_dict(datasets: list[str], dataset_type: str) -> dict[str, list[str]]:
    """
    Retrieve molecular SMILES from QCArchive datasets.

    Connects to QCArchive portal and downloads molecular data from specified
    datasets, extracting mapped SMILES strings for coverage analysis.

    Parameters
    ----------
    datasets : list[str]
        List of dataset names to retrieve from QCArchive.
    dataset_type : str
        Type of QCArchive dataset (e.g., "optimization", "torsiondrive").

    Returns
    -------
    dict[str, list[str]]
        Dictionary mapping dataset names to lists of mapped SMILES strings.
        Structure: {dataset_name: [mapped_smiles1, mapped_smiles2, ...]}

    Notes
    -----
    Extracts "mapped_smiles" from entry extras, providing atom-mapped
    molecular representations needed for parameter assignment.
    """

    client = PortalClient("https://api.qcarchive.molssi.org:443/", cache_dir=".")
    smiles_dict: dict[str, list[str]] = {}
    for dataset_name in datasets:
        ds = client.get_dataset(dataset_type, dataset_name)
        smiles_dict[dataset_name] = list(
            set([entry["extras"]["mapped_smiles"] for entry in ds.get_entries])
        )
    return smiles_dict


def get_train_test_smiles_dict(filename: pathlib.Path | str) -> dict[str, list[str]]:
    """
    Load training and testing molecular data splits from JSON file.

    Reads pre-defined train/test splits for consistent dataset partitioning
    across experiments.

    Parameters
    ----------
    filename : pathlib.Path | str
        Path to JSON file with format: {"train": [smiles], "test": [smiles]}

    Returns
    -------
    dict[str, list[str]]
        Dictionary with "Train" and "Test" keys mapping to SMILES lists.
        Structure: {"Train": [smiles1, ...], "Test": [smiles1, ...]}
    """
    with open(filename, "r") as f:
        smiles_data = json.load(f)
    return {"Train": smiles_data["train"], "Test": smiles_data["test"]}


def test_coverage(filename_offxml: str, smiles_dict: dict[str, list[str]]) -> None:
    """
    Test force field parameter coverage across molecular datasets.

    Validates that the generated force field can parameterize molecules from
    training, testing, and external datasets, identifying any molecules or
    components that lack necessary parameters.

    Parameters
    ----------
    filename_offxml : str
        Path to the force field file to test.
    smiles_dict : dict[str, list[str]]
        Dictionary mapping dataset names to SMILES string lists.
        Structure: {dataset_name: [smiles1, smiles2, ...]}

    Returns
    -------
    None
        Logs coverage analysis results for each dataset, including
        uncovered component counts and types.

    Notes
    -----
    Components are molecular mechanics elements; this function tests Bond and
    Angle coverage. Uncovered components indicate molecular environments
    not present in the force field, suggesting potential parameter gaps
    or training limitations.
    """

    new_ff = ForceField(filename_offxml)

    for dataset_name, smiles in smiles_dict.items():
        logger.info(f"\nChecking coverage for {dataset_name} dataset...")
        uncovered = check_all_components_fully_covered_parallel_chunks(smiles, new_ff)
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
    filename_test_train_smiles: pathlib.Path | str,
    datasets: list[str],
    dataset_type: str,
) -> None:
    """
    Execute complete force field parameter generation and validation pipeline.

    Parameters
    ----------
    filename_offxml_out : str
        Output path for the enhanced force field file (.offxml format).
    filename_offxml_in : pathlib.Path | str
        Input template force field file path. Should be a valid SMIRNOFF force field.
    filename_test_train_smiles : pathlib.Path | str
        JSON file containing train/test molecular splits for validation.
    datasets : list[str]
        List of QCArchive dataset names for coverage testing.
    dataset_type : str
        Type of QCArchive datasets (e.g., "optimization").

    Returns
    -------
    None
    """

    smiles_dict = get_train_test_smiles_dict(filename_test_train_smiles)
    components_by_type = get_components_by_type(smiles_dict["Train"])
    write_forcefield_file(components_by_type, filename_offxml_out, filename_offxml_in)

    qca_dict = get_qca_smiles_dict(datasets, dataset_type)
    smiles_dict.update(qca_dict)
    test_coverage(filename_offxml_out, smiles_dict)


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
            --filename-test-train-smiles molecular_splits.json \
            --datasets "OpenFF Additional Generated ChEMBL Optimizations v4.0" "OpenFF Additional Generated ChEMBL Optimizations v4.0" \
            --datasets-type singlepoint

Pipeline Overview:
    1. Loads smiles set from a JSON file containing two lists under the keywords "Test" and "Train"
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
        help="Type of QCArchive dataset (e.g., 'optimization', 'singlepoint'). Required if --datasets is provided.",
    )
    args = parser.parse_args()
    main(
        args.filename_offxml_out,
        args.filename_offxml_in,
        args.filename_test_train_smiles,
        args.datasets,
        args.datasets_type,
    )
