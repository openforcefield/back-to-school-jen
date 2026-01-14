"""Force field benchmarking using YAMMBS.

This module benchmarks force field performance against QM reference data using
the YAMMBS (Yet Another Molecular Mechanics Benchmarking Suite) package. It
computes metrics including DDE, RMSD, TFD, and internal coordinate RMSDs,
with optional breakdown by type and plotting.

Command-line Arguments
----------------------
--offxml : str
    Path to OFFXML force field file to benchmark.
--input-dataset : str
    Path to YAMMBS QCArchiveDataset JSON file with QM reference data.
--database : str
    Path to SQLite database file for storing results (default: benchmark.sqlite).
--n-processes : int, optional
    Number of parallel processes for MM optimization (default: 4).
--force-field-name : str, optional
    Custom name/tag for the force field (default: filename without extension).
--skip-type-analysis : bool, optional
    Skip bond/angle/dihedral/improper breakdown analysis (default: False).
--skip-plots : bool, optional
    Skip generating plots (default: False).
--output-dir : str, optional
    Directory for output files (default: current directory).

Examples
--------
Basic benchmarking:
$ python benchmarking.py --offxml my-forcefield.offxml \\
    --input-dataset qcarchive-dataset.json

With custom database and multiple processes:
$ python benchmarking.py --offxml openff-2.2.1.offxml \\
    --input-dataset dataset.json --database results.sqlite --n-processes 8

Output Structure
----------------
Creates the following outputs:
- {database}.sqlite         # SQLite database with all results
- metrics.json              # Summary metrics for all force fields
- metrics_by_type.json    # Metrics broken down by parameter type
- minimized.json            # MM-minimized geometries and energies
- plots/                    # Directory containing benchmark plots
    - dde_histogram.png
    - rmsd_cdf.png
    - tfd_cdf.png
    - icrmsd_comparison.png
    - icrmsd_by_type.png

Metrics Computed
----------------
- DDE: Deformation-driven energy difference (kcal/mol)
- RMSD: Root-mean-square deviation of coordinates (Angstrom)
- TFD: Torsion fingerprint deviation
- ICRMSD: Internal coordinate RMSDs (Bond, Angle, Dihedral, Improper)
"""

import argparse
import json
import pathlib
import sys
from collections import defaultdict
from multiprocessing import freeze_support, Pool

import numpy as np
from loguru import logger

from openff.toolkit import Molecule, ForceField
from openff.qcsubmit.results import OptimizationResultCollection
from yammbs import MoleculeStore
from yammbs.inputs import QCArchiveDataset
from yammbs.outputs import MetricCollection
from yammbs.analysis import get_internal_coordinate_differences

# Import plotting functions from plot_benchmark.py
from plot_benchmark import (
    generate_basic_plots,
    generate_parameter_type_plots,
)

logger.remove()


def load_qcarchive_dataset(input_path: pathlib.Path | str) -> QCArchiveDataset:
    """Load QCArchive dataset from JSON file.

    Supports both native yammbs QCArchiveDataset format and openff-qcsubmit
    OptimizationResultCollection format.

    Parameters
    ----------
    input_path : pathlib.Path | str
        Path to JSON file containing QCArchiveDataset or OptimizationResultCollection.

    Returns
    -------
    QCArchiveDataset
        Loaded dataset with QM reference molecules.

    Raises
    ------
    FileNotFoundError
        If input file does not exist.
    ValueError
        If JSON file cannot be parsed.

    Examples
    --------
    >>> dataset = load_qcarchive_dataset("qcarchive-data.json")
    >>> len(dataset.qm_molecules)
    1500
    """
    input_path = pathlib.Path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input dataset not found: {input_path}")

    logger.info(f"Loading QCArchive dataset from: {input_path}")

    # First, try to load as native QCArchiveDataset
    with open(input_path, "r") as f:
        content = f.read()

    try:
        dataset = QCArchiveDataset.model_validate_json(content)
        if len(dataset.qm_molecules) > 0:
            logger.info(
                f"Loaded {len(dataset.qm_molecules)} QM molecules (QCArchiveDataset format)"
            )
            return dataset
    except Exception:
        pass

    # Try to load as OptimizationResultCollection (qcsubmit format)
    try:
        logger.info("Attempting to load as OptimizationResultCollection...")
        collection = OptimizationResultCollection.parse_file(input_path)
        dataset = QCArchiveDataset.from_qcsubmit_collection(collection)
        logger.info(
            f"Loaded {len(dataset.qm_molecules)} QM molecules (OptimizationResultCollection format)"
        )
        return dataset
    except ImportError:
        raise ValueError(
            "File appears to be OptimizationResultCollection format but openff-qcsubmit "
            "is not installed. Install with: pip install openff-qcsubmit"
        )
    except Exception as e:
        raise ValueError(
            f"Failed to parse input file as QCArchiveDataset or OptimizationResultCollection: {e}"
        )


def get_or_create_store(
    database_path: pathlib.Path | str,
    dataset: QCArchiveDataset | None = None,
) -> MoleculeStore:
    """Get existing or create new MoleculeStore database.

    Parameters
    ----------
    database_path : pathlib.Path | str
        Path to SQLite database file.
    dataset : QCArchiveDataset, optional
        Dataset to populate new database. Required if database doesn't exist.

    Returns
    -------
    MoleculeStore
        Database store for molecules and benchmark results.

    Raises
    ------
    ValueError
        If database doesn't exist and no dataset provided.

    Examples
    --------
    >>> store = get_or_create_store("benchmark.sqlite", dataset)
    """
    database_path = pathlib.Path(database_path)

    if database_path.exists():
        logger.info(f"Loading existing database: {database_path}")
        return MoleculeStore(str(database_path))
    else:
        if dataset is None:
            raise ValueError(
                f"Database {database_path} does not exist and no dataset provided"
            )
        logger.info(f"Creating new database: {database_path}")
        return MoleculeStore.from_qm_dataset(dataset, str(database_path))


def run_benchmark(
    store: MoleculeStore,
    force_field: str,
    n_processes: int = 4,
) -> None:
    """Run MM optimizations for a force field.

    Parameters
    ----------
    store : MoleculeStore
        Database store containing QM reference data.
    force_field : str
        Path to OFFXML force field.
    n_processes : int, optional
        Number of parallel processes (default: 4).

    Examples
    --------
    >>> run_benchmark(store, "openff-2.2.1.offxml", n_processes=8)
    """
    logger.info(f"Running MM optimizations with force field: {force_field}")
    logger.info(f"Using {n_processes} parallel processes")

    store.optimize_mm(force_field=force_field, n_processes=n_processes)

    logger.info("MM optimizations complete")


def get_metrics(store: MoleculeStore) -> MetricCollection:
    """Compute benchmark metrics for all force fields in store.

    Parameters
    ----------
    store : MoleculeStore
        Database store with completed MM optimizations.

    Returns
    -------
    MetricCollection
        Collection of DDE, RMSD, TFD, and ICRMSD metrics.

    Examples
    --------
    >>> metrics = get_metrics(store)
    >>> metrics.metrics["openff-2.2.1"]["12345"].dde
    0.523
    """
    logger.info("Computing benchmark metrics...")
    return store.get_metrics()


def get_parameter_labels_for_molecule(
    molecule: Molecule,
    force_field: ForceField,
) -> dict[str, dict[tuple[int, ...], str]]:
    """Get SMIRKS parameter labels for all internal coordinates in a molecule.

    Parameters
    ----------
    molecule : Molecule
        OpenFF Molecule object.
    force_field : ForceField
        OpenFF ForceField to use for labeling.

    Returns
    -------
    dict[str, dict[tuple[int, ...], str]]
        Mapping: {ic_type: {atom_indices: smirks_id}}.
    """
    labels = force_field.label_molecules(molecule.to_topology())[0]

    result: dict[str, dict[tuple[int, ...], str]] = {
        "Bond": {},
        "Angle": {},
        "Dihedral": {},
        "Improper": {},
    }

    # Bonds
    if "Bonds" in labels:
        for indices, param in labels["Bonds"].items():
            # Normalize bond indices (sorted tuple)
            norm_indices = tuple(sorted(indices))
            result["Bond"][norm_indices] = param.id

    # Angles
    if "Angles" in labels:
        for indices, param in labels["Angles"].items():
            result["Angle"][indices] = param.id

    # Proper Torsions
    if "ProperTorsions" in labels:
        for indices, param in labels["ProperTorsions"].items():
            result["Dihedral"][indices] = param.id

    # Improper Torsions
    if "ImproperTorsions" in labels:
        for indices, param in labels["ImproperTorsions"].items():
            result["Improper"][indices] = param.id

    return result


def _process_single_molecule_for_param_analysis(args):
    """Process a single molecule for parameter type analysis.

    This is a worker function for parallel processing.

    Parameters
    ----------
    args : tuple
        (mol_id, smiles, qm_conformers, mm_conformers, force_field_path)

    Returns
    -------
    dict or None
        Parameter deviations for this molecule, or None if processing failed.
    """
    mol_id, smiles, qm_conformers, mm_conformers, force_field_path = args

    try:
        molecule = Molecule.from_mapped_smiles(smiles, allow_undefined_stereo=True)
        ff = ForceField(force_field_path)

        # Get parameter labels
        labels = ff.label_molecules(molecule.to_topology())[0]
        param_labels = {
            "Bond": {},
            "Angle": {},
            "Dihedral": {},
            "Improper": {},
        }

        if "Bonds" in labels:
            for indices, param in labels["Bonds"].items():
                norm_indices = tuple(sorted(indices))
                param_labels["Bond"][norm_indices] = param.id
        if "Angles" in labels:
            for indices, param in labels["Angles"].items():
                param_labels["Angle"][indices] = param.id
        if "ProperTorsions" in labels:
            for indices, param in labels["ProperTorsions"].items():
                param_labels["Dihedral"][indices] = param.id
        if "ImproperTorsions" in labels:
            for indices, param in labels["ImproperTorsions"].items():
                param_labels["Improper"][indices] = param.id

        if len(qm_conformers) != len(mm_conformers):
            return None

        # Collect deviations for this molecule
        mol_deviations = {
            "Bond": defaultdict(list),
            "Angle": defaultdict(list),
            "Dihedral": defaultdict(list),
            "Improper": defaultdict(list),
        }

        for qm_conf, mm_conf in zip(qm_conformers, mm_conformers):
            try:
                ic_diffs = get_internal_coordinate_differences(
                    molecule, qm_conf, mm_conf
                )

                for ic_type in ["Bond", "Angle", "Dihedral", "Improper"]:
                    if ic_type not in ic_diffs:
                        continue

                    for indices, diff in ic_diffs[ic_type].items():
                        if ic_type == "Bond":
                            indices = tuple(sorted(indices))

                        smirks_id = param_labels.get(ic_type, {}).get(indices)
                        if smirks_id is None:
                            smirks_id = "unmatched"

                        if ic_type in ["Angle", "Dihedral", "Improper"]:
                            diff = np.degrees(diff)

                        mol_deviations[ic_type][smirks_id].append(abs(diff))

            except Exception:
                continue

        # Convert defaultdicts to regular dicts
        return {
            ic_type: dict(smirks_dict)
            for ic_type, smirks_dict in mol_deviations.items()
        }

    except Exception:
        return None


def analyze_by_parameter_type(
    store: MoleculeStore,
    force_field_path: str,
    n_processes: int = 1,
) -> dict[str, dict[str, list[float]]]:
    """Analyze internal coordinate deviations by SMIRKS parameter type.

    For each molecule, computes the difference between QM and MM internal
    coordinates, then groups these differences by the SMIRKS pattern that
    matched each internal coordinate.

    Parameters
    ----------
    store : MoleculeStore
        Database store with molecule and conformer information.
    force_field_path : str
        Path to the force field file used for benchmarking.
    n_processes : int, optional
        Number of parallel processes (default: 1).

    Returns
    -------
    dict[str, dict[str, list[float]]]
        Nested dict: {ic_type: {smirks_id: [deviations]}}.
        ic_type is one of: "Bond", "Angle", "Dihedral", "Improper".
        Deviations are in Angstroms (bonds) or degrees (angles/dihedrals).

    Examples
    --------
    >>> param_deviations = analyze_by_parameter_type(store, "openff-2.2.1.offxml")
    >>> np.mean(param_deviations["Bond"]["[#6X4:1]-[#6X4:2]"])
    0.012
    """
    print("Analyzing internal coordinates by parameter type...")
    print(f"Using {n_processes} parallel processes")

    molecule_ids = store.get_molecule_ids()
    force_fields_in_store = store.get_force_fields()
    total_molecules = len(molecule_ids)

    print(f"Total molecules to process: {total_molecules}")

    # Find matching force field in store
    ff_tag = None
    for ff_name in force_fields_in_store:
        if force_field_path in ff_name or ff_name in force_field_path:
            ff_tag = ff_name
            break
    if ff_tag is None and force_fields_in_store:
        ff_tag = force_fields_in_store[0]
        print(f"Warning: Could not match force field path, using: {ff_tag}")

    if ff_tag is None:
        print("Error: No force field results found in store")
        return {"Bond": {}, "Angle": {}, "Dihedral": {}, "Improper": {}}

    # Prepare data for parallel processing
    print("Preparing molecule data for parallel processing...")
    work_items = []
    for mol_id in molecule_ids:
        try:
            smiles = store.get_smiles_by_molecule_id(mol_id)
            qm_conformers = store.get_qm_conformers_by_molecule_id(mol_id)
            mm_conformers = store.get_mm_conformers_by_molecule_id(mol_id, ff_tag)
            work_items.append(
                (mol_id, smiles, qm_conformers, mm_conformers, force_field_path)
            )
        except Exception as e:
            logger.debug(f"Error preparing molecule {mol_id}: {e}")
            continue

    print(f"Prepared {len(work_items)} molecules for processing")

    # Initialize result structure
    param_deviations: dict[str, dict[str, list[float]]] = {
        "Bond": defaultdict(list),
        "Angle": defaultdict(list),
        "Dihedral": defaultdict(list),
        "Improper": defaultdict(list),
    }

    # Process molecules
    processed = 0
    last_percent_reported = -5  # Start at -5 so we report 0%

    if n_processes == 1:
        # Sequential processing
        print("Processing molecules sequentially...")
        for work_item in work_items:
            result = _process_single_molecule_for_param_analysis(work_item)
            if result is not None:
                # Merge results
                for ic_type in ["Bond", "Angle", "Dihedral", "Improper"]:
                    for smirks_id, values in result[ic_type].items():
                        param_deviations[ic_type][smirks_id].extend(values)

            processed += 1
            percent_complete = int(100 * processed / len(work_items))
            if percent_complete >= last_percent_reported + 5:
                print(
                    f"Progress: {percent_complete}% complete ({processed}/{len(work_items)} molecules)"
                )
                last_percent_reported = percent_complete
    else:
        # Parallel processing
        print(f"Processing molecules in parallel with {n_processes} workers...")

        with Pool(processes=n_processes) as pool:
            # Use imap for ordered results with progress tracking
            for result in pool.imap(
                _process_single_molecule_for_param_analysis, work_items, chunksize=50
            ):
                if result is not None:
                    # Merge results
                    for ic_type in ["Bond", "Angle", "Dihedral", "Improper"]:
                        for smirks_id, values in result[ic_type].items():
                            param_deviations[ic_type][smirks_id].extend(values)

                processed += 1
                percent_complete = int(100 * processed / len(work_items))
                if percent_complete >= last_percent_reported + 5:
                    print(
                        f"Progress: {percent_complete}% complete ({processed}/{len(work_items)} molecules)"
                    )
                    last_percent_reported = percent_complete

    print(f"Analysis complete! Processed {processed} molecules")

    # Count total deviations collected
    total_deviations = sum(
        len(values)
        for ic_dict in param_deviations.values()
        for values in ic_dict.values()
    )
    print(f"Total deviations collected: {total_deviations}")

    # Convert defaultdicts to regular dicts
    result = {}
    for ic_type, smirks_dict in param_deviations.items():
        result[ic_type] = dict(smirks_dict)

    return result


def compute_parameter_type_summary(
    param_deviations: dict[str, dict[str, list[float]]],
) -> dict[str, dict[str, dict[str, float]]]:
    """Compute summary statistics for parameter type deviations.

    Parameters
    ----------
    param_deviations : dict
        Raw deviations from analyze_by_parameter_type().

    Returns
    -------
    dict
        Summary statistics: {ic_type: {smirks_id: {stat: value}}}.
        Statistics include mean, std, median, count, min, max.

    Examples
    --------
    >>> summary = compute_parameter_type_summary(param_deviations)
    >>> summary["Bond"]["[#6X4:1]-[#6X4:2]"]["mean"]
    0.012
    """
    logger.info("Computing parameter type summary statistics...")

    summary: dict[str, dict[str, dict[str, float]]] = {}

    for ic_type, smirks_dict in param_deviations.items():
        summary[ic_type] = {}

        for smirks_id, values in smirks_dict.items():
            if not values:
                continue

            arr = np.array(values)
            arr = arr[np.isfinite(arr)]

            if len(arr) == 0:
                continue

            summary[ic_type][smirks_id] = {
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "median": float(np.median(arr)),
                "count": int(len(arr)),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
            }

    return summary


def save_results(
    output_dir: pathlib.Path,
    metrics: MetricCollection,
    param_summary: dict | None = None,
    store: MoleculeStore | None = None,
) -> None:
    """Save benchmark results to files.

    Parameters
    ----------
    output_dir : pathlib.Path
        Directory to save output files.
    metrics : MetricCollection
        Computed benchmark metrics.
    param_summary : dict, optional
        Parameter type breakdown summary statistics.
    store : MoleculeStore, optional
        Database store for saving minimized outputs.

    Examples
    --------
    >>> save_results(Path("./results"), metrics, param_summary, store)
    """
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save metrics
    metrics_path = output_dir / "metrics.json"
    logger.info(f"Saving metrics to: {metrics_path}")
    with open(metrics_path, "w") as f:
        f.write(metrics.model_dump_json(indent=2))

    # Save parameter type summary
    if param_summary is not None:
        param_path = output_dir / "metrics_by_parameter_type.json"
        logger.info(f"Saving parameter type analysis to: {param_path}")
        with open(param_path, "w") as f:
            json.dump(param_summary, f, indent=2)

    # Save minimized outputs
    if store is not None:
        outputs_path = output_dir / "minimized.json"
        logger.info(f"Saving minimized outputs to: {outputs_path}")
        with open(outputs_path, "w") as f:
            f.write(store.get_outputs().model_dump_json(indent=2))


def print_summary(
    metrics: MetricCollection,
) -> None:
    """Print summary of benchmark results to console.

    Parameters
    ----------
    metrics : MetricCollection
        Computed benchmark metrics.
    """
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)

    for force_field, ff_metrics in metrics.metrics.items():
        print(f"\nForce Field: {force_field}")
        print("-" * 40)

        ddes = [m.dde for m in ff_metrics.values() if m.dde is not None]
        rmsds = [m.rmsd for m in ff_metrics.values() if m.rmsd is not None]
        tfds = [m.tfd for m in ff_metrics.values() if m.tfd is not None]

        if ddes:
            dde_arr = np.array(ddes)
            dde_arr = dde_arr[np.isfinite(dde_arr)]
            if len(dde_arr) > 0:
                print(
                    f"  DDE:  mean={np.mean(dde_arr):.3f}, std={np.std(dde_arr):.3f}, n={len(dde_arr)}"
                )

        if rmsds:
            rmsd_arr = np.array(rmsds)
            print(
                f"  RMSD: mean={np.mean(rmsd_arr):.3f}, std={np.std(rmsd_arr):.3f}, n={len(rmsd_arr)}"
            )

        if tfds:
            tfd_arr = np.array(tfds)
            print(
                f"  TFD:  mean={np.mean(tfd_arr):.4f}, std={np.std(tfd_arr):.4f}, n={len(tfd_arr)}"
            )


def main(
    offxml: pathlib.Path | str,
    input_dataset: pathlib.Path | str,
    database: pathlib.Path | str = "benchmark.sqlite",
    n_processes: int = 4,
    force_field_name: str | None = None,
    skip_parameter_analysis: bool = False,
    skip_plots: bool = False,
    output_dir: pathlib.Path | str = ".",
) -> None:
    """Run force field benchmarking pipeline.

    Parameters
    ----------
    offxml : pathlib.Path | str
        Path to OFFXML force field file.
    input_dataset : pathlib.Path | str
        Path to QCArchiveDataset JSON file.
    database : pathlib.Path | str, optional
        Path to SQLite database (default: benchmark.sqlite).
    n_processes : int, optional
        Number of parallel processes (default: 4).
    force_field_name : str, optional
        Custom name for force field (default: filename).
    skip_parameter_analysis : bool, optional
        Skip SMIRKS parameter type breakdown (default: False).
    skip_plots : bool, optional
        Skip generating plots (default: False).
    output_dir : pathlib.Path | str, optional
        Output directory (default: current directory).

    Examples
    --------
    >>> main("openff-2.2.1.offxml", "dataset.json", n_processes=8)
    """
    offxml = pathlib.Path(offxml)
    input_dataset = pathlib.Path(input_dataset)
    database = pathlib.Path(database)
    output_dir = pathlib.Path(output_dir)

    # Validate inputs
    if not offxml.exists():
        raise FileNotFoundError(f"Force field not found: {offxml}")

    # Determine force field name/tag
    if force_field_name is None:
        force_field_name = offxml.name

    logger.info(f"Benchmarking force field: {force_field_name}")

    # Load dataset and create/get store
    dataset = load_qcarchive_dataset(input_dataset)
    store = get_or_create_store(database, dataset)

    # Run MM optimizations
    run_benchmark(store, str(offxml), n_processes=n_processes)

    # Compute metrics
    metrics = get_metrics(store)
    if force_field_name is not None and force_field_name != str(offxml):
        try:
            if str(offxml) in metrics.metrics:
                metrics.metrics[force_field_name] = metrics.metrics.pop(str(offxml))
        except (KeyError, AttributeError, TypeError) as e:
            logger.warning(f"Could not relabel metrics to '{force_field_name}': {e}")

    # Parameter type analysis (bonds, angles, dihedrals, impropers by SMIRKS)
    parameter_type_summary = None
    if not skip_parameter_analysis:
        try:
            param_metrics = analyze_by_parameter_type(
                store, str(offxml), n_processes=n_processes
            )
            parameter_type_summary = compute_parameter_type_summary(param_metrics)
        except Exception as e:
            logger.warning(f"Parameter type analysis failed: {e}")
            logger.warning("Continuing without parameter type breakdown")

    # Save results
    save_results(output_dir, metrics, parameter_type_summary, store)

    # Generate plots using the saved JSON files
    if not skip_plots:
        try:
            # Load the saved metrics JSON for plotting
            metrics_path = output_dir / "metrics.json"
            with open(metrics_path, "r") as f:
                metrics_dict = json.load(f)

            generate_basic_plots([metrics_dict], [force_field_name], output_dir)

            if parameter_type_summary:
                generate_parameter_type_plots(
                    [parameter_type_summary], [force_field_name], output_dir
                )
        except Exception as e:
            logger.warning(f"Plot generation failed: {e}")
            logger.warning("Continuing without plots")

    # Print summary
    print_summary(metrics)

    logger.info("Benchmarking complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark force fields using YAMMBS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    Basic benchmarking:
        python benchmarking.py --offxml my-forcefield.offxml \\
            --input-dataset qcarchive-dataset.json

    With custom database and processes:
        python benchmarking.py --offxml openff-2.2.1.offxml \\
            --input-dataset dataset.json --database results.sqlite --n-processes 8

    Multiple force fields (run script multiple times with same database):
        python benchmarking.py --offxml ff1.offxml --input-dataset data.json --database bench.sqlite
        python benchmarking.py --offxml ff2.offxml --input-dataset data.json --database bench.sqlite

Output Files:
    - {database}.sqlite       : SQLite database with all results
    - metrics.json            : Summary metrics for all force fields
    - metrics_by_parameter_type.json : Metrics broken down by SMIRKS parameter type
    - minimized.json          : MM-minimized geometries and energies
    - plots/                  : Directory containing benchmark plots (PDF vector format)
        - dde_histogram.pdf     : DDE distribution histogram
        - rmsd_cdf.pdf          : RMSD cumulative distribution
        - tfd_cdf.pdf           : TFD cumulative distribution
        - icrmsd_comparison.pdf : Bar chart of ICRMSD by type (Bond, Angle, Dihedral, Improper)
        - icrmsd_by_type.pdf    : CDF plots for each internal coordinate type
        - deviation_by_*_type.pdf : Mean deviation by SMIRKS parameter type
        - deviation_dist_*_type.pdf : Deviation distributions by parameter type
        - deviation_summary_all_types.pdf : Summary of all IC type deviations
        """,
    )
    parser.add_argument(
        "--offxml",
        type=str,
        required=True,
        help="Path to OFFXML force field file to benchmark",
    )
    parser.add_argument(
        "--input-dataset",
        type=str,
        required=True,
        help="Path to YAMMBS QCArchiveDataset JSON file",
    )
    parser.add_argument(
        "--database",
        type=str,
        default="benchmark.sqlite",
        help="Path to SQLite database file (default: benchmark.sqlite)",
    )
    parser.add_argument(
        "--n-processes",
        type=int,
        default=1,
        help="Number of parallel processes for MM optimization (default: 1)",
    )
    parser.add_argument(
        "--force-field-name",
        type=str,
        default=None,
        help=(
            "Custom display name/tag for the force field (default: filename). "
            "This is only used for reporting/logging and will NOT be used to "
            "locate or load the OFFXML file — provide the OFFXML via --offxml."
        ),
    )
    parser.add_argument(
        "--skip-type-analysis",
        action="store_true",
        help="Skip bond/angle/dihedral/improper type breakdown analysis",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Skip generating benchmark plots",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Directory for output files (default: current directory)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase verbosity: -v for WARNING, -vv for INFO, -vvv for DEBUG",
    )

    args = parser.parse_args()

    # Configure logging
    if args.verbose == 0:
        pass  # Silent
    elif args.verbose == 1:
        logger.add(sys.stdout, level="WARNING")
    elif args.verbose == 2:
        logger.add(sys.stdout, level="INFO")
    elif args.verbose >= 3:
        logger.add(sys.stdout, level="DEBUG")

    # Required for multiprocessing on some platforms
    freeze_support()

    main(
        args.offxml,
        args.input_dataset,
        args.database,
        args.n_processes,
        args.force_field_name,
        args.skip_type_analysis,
        args.skip_plots,
        args.output_dir,
    )
