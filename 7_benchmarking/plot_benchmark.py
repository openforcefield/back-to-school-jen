"""Generate benchmark plots from existing benchmark results.

This script reads the output files from benchmarking.py and generates
visualization plots. It can work with metrics.json alone for basic plots,
or with benchmark.sqlite and an offxml file for parameter type analysis.

Command-line Arguments
----------------------
--metrics : str
    Path to metrics.json file from benchmarking run.
--output-dir : str
    Directory to save plot files (default: current directory).
--database : str, optional
    Path to benchmark.sqlite for parameter type analysis.
--offxml : str, optional
    Path to OFFXML force field file (required for parameter type analysis).
--param-json : str, optional
    Path to existing metrics_by_parameter_type.json (alternative to database+offxml).

Examples
--------
Basic plots from metrics.json only:
    $ python plot_benchmark.py --metrics benchmark_results/metrics.json

Full plots including parameter type analysis:
    $ python plot_benchmark.py --metrics benchmark_results/metrics.json \\
        --database benchmark.sqlite --offxml openff-2.2.1.offxml

Using pre-computed parameter type analysis:
    $ python plot_benchmark.py --metrics benchmark_results/metrics.json \\
        --param-json benchmark_results/metrics_by_parameter_type.json
"""

import argparse
import json
import pathlib
import sys
from collections import defaultdict
from multiprocessing import Pool, freeze_support

import numpy as np
from loguru import logger
import matplotlib.pyplot as plt

# Optional imports for parameter type analysis
from openff.toolkit import Molecule, ForceField
from yammbs import MoleculeStore
from yammbs.analysis import get_internal_coordinate_differences

logger.remove()


def load_metrics(metrics_path: pathlib.Path) -> dict:
    """Load metrics from JSON file.

    Parameters
    ----------
    metrics_path : pathlib.Path
        Path to metrics.json file.

    Returns
    -------
    dict
        Loaded metrics data.
    """
    logger.info(f"Loading metrics from: {metrics_path}")
    with open(metrics_path, "r") as f:
        return json.load(f)


def load_parameter_type_summary(param_path: pathlib.Path) -> dict:
    """Load parameter type summary from JSON file.

    Parameters
    ----------
    param_path : pathlib.Path
        Path to metrics_by_parameter_type.json file.

    Returns
    -------
    dict
        Parameter type summary statistics.
    """
    logger.info(f"Loading parameter type summary from: {param_path}")
    with open(param_path, "r") as f:
        return json.load(f)


def get_parameter_labels_for_molecule(
    molecule: "Molecule",
    force_field: "ForceField",
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

    if "Bonds" in labels:
        for indices, param in labels["Bonds"].items():
            norm_indices = tuple(sorted(indices))
            result["Bond"][norm_indices] = param.id

    if "Angles" in labels:
        for indices, param in labels["Angles"].items():
            result["Angle"][indices] = param.id

    if "ProperTorsions" in labels:
        for indices, param in labels["ProperTorsions"].items():
            result["Dihedral"][indices] = param.id

    if "ImproperTorsions" in labels:
        for indices, param in labels["ImproperTorsions"].items():
            result["Improper"][indices] = param.id

    return result


def _process_single_molecule(args):
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
    store: "MoleculeStore",
    force_field_path: str,
    n_processes: int = 1,
) -> dict[str, dict[str, list[float]]]:
    """Analyze internal coordinate deviations by SMIRKS parameter type.

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
            result = _process_single_molecule(work_item)
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
            for result in pool.imap(_process_single_molecule, work_items, chunksize=50):
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


def generate_basic_plots(
    metrics: dict,
    output_dir: pathlib.Path,
) -> None:
    """Generate basic benchmark plots from metrics.json.

    Parameters
    ----------
    metrics : dict
        Loaded metrics data.
    output_dir : pathlib.Path
        Directory to save plot files.
    """
    logger.info("Generating basic benchmark plots...")

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Extract force fields and metrics
    ff_metrics = metrics.get("metrics", metrics)
    force_fields = list(ff_metrics.keys())
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(force_fields), 1)))

    # Extract metrics by force field
    ff_data: dict[str, dict[str, list[float]]] = {}
    for ff, records in ff_metrics.items():
        ff_data[ff] = {
            "dde": [],
            "rmsd": [],
            "tfd": [],
            "icrmsd_bond": [],
            "icrmsd_angle": [],
            "icrmsd_dihedral": [],
            "icrmsd_improper": [],
        }
        for _, metric in records.items():
            if isinstance(metric, dict):
                if metric.get("dde") is not None:
                    ff_data[ff]["dde"].append(metric["dde"])
                if metric.get("rmsd") is not None:
                    ff_data[ff]["rmsd"].append(metric["rmsd"])
                if metric.get("tfd") is not None:
                    ff_data[ff]["tfd"].append(metric["tfd"])
                if metric.get("icrmsd"):
                    icrmsd = metric["icrmsd"]
                    if icrmsd.get("Bond") is not None:
                        ff_data[ff]["icrmsd_bond"].append(icrmsd["Bond"])
                    if icrmsd.get("Angle") is not None:
                        ff_data[ff]["icrmsd_angle"].append(icrmsd["Angle"])
                    if icrmsd.get("Dihedral") is not None:
                        ff_data[ff]["icrmsd_dihedral"].append(icrmsd["Dihedral"])
                    if icrmsd.get("Improper") is not None:
                        ff_data[ff]["icrmsd_improper"].append(icrmsd["Improper"])

    # Plot 1: DDE Histogram
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, ff in enumerate(force_fields):
        data = np.array(ff_data[ff]["dde"])
        data = data[np.isfinite(data)]
        if len(data) > 0:
            counts, bins = np.histogram(data, bins=np.linspace(-15, 15, 31))
            ax.stairs(counts, bins, label=ff, color=colors[i])
    ax.set_xlabel("DDE (kcal/mol)")
    ax.set_ylabel("Count")
    ax.set_title("Deformation-Driven Energy Difference Distribution")
    ax.legend(loc="best")
    ax.axvline(x=0, color="gray", linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(plots_dir / "dde_histogram.pdf", bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {plots_dir / 'dde_histogram.pdf'}")

    # Plot 2: RMSD CDF
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, ff in enumerate(force_fields):
        data = np.sort(ff_data[ff]["rmsd"])
        if len(data) > 0:
            cdf = np.arange(1, len(data) + 1) / len(data)
            ax.plot(data, cdf, "-", label=ff, color=colors[i])
    ax.set_xlabel("RMSD (Å)")
    ax.set_ylabel("CDF")
    ax.set_title("Root Mean Square Deviation - Cumulative Distribution")
    ax.set_xlim(-0.1, 3.0)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(plots_dir / "rmsd_cdf.pdf", bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {plots_dir / 'rmsd_cdf.pdf'}")

    # Plot 3: TFD CDF
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, ff in enumerate(force_fields):
        data = np.sort(ff_data[ff]["tfd"])
        if len(data) > 0:
            cdf = np.arange(1, len(data) + 1) / len(data)
            ax.plot(data, cdf, "-", label=ff, color=colors[i])
    ax.set_xlabel("TFD")
    ax.set_ylabel("CDF")
    ax.set_title("Torsion Fingerprint Deviation - Cumulative Distribution")
    ax.set_xlim(-0.02, 0.5)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(plots_dir / "tfd_cdf.pdf", bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {plots_dir / 'tfd_cdf.pdf'}")

    # Plot 4: ICRMSD Comparison Bar Chart
    ic_types = ["Bond", "Angle", "Dihedral", "Improper"]
    ic_keys = ["icrmsd_bond", "icrmsd_angle", "icrmsd_dihedral", "icrmsd_improper"]
    ic_units = ["Å", "°", "°", "°"]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    x = np.arange(len(force_fields))
    width = 0.6

    for idx, (ic_type, ic_key, unit) in enumerate(zip(ic_types, ic_keys, ic_units)):
        ax = axes[idx]
        means = []
        stds = []
        for ff in force_fields:
            data = np.array(ff_data[ff][ic_key])
            data = data[np.isfinite(data)]
            if len(data) > 0:
                means.append(np.mean(data))
                stds.append(np.std(data))
            else:
                means.append(0)
                stds.append(0)

        ax.bar(x, means, width, yerr=stds, capsize=3, color=colors[: len(force_fields)])
        ax.set_ylabel(f"ICRMSD ({unit})")
        ax.set_title(f"{ic_type} RMSD")
        ax.set_xticks(x)
        ax.set_xticklabels(
            [ff.split("/")[-1][:15] for ff in force_fields], rotation=45, ha="right"
        )

    fig.suptitle("Internal Coordinate RMSD by Type", fontsize=14)
    fig.tight_layout()
    fig.savefig(plots_dir / "icrmsd_comparison.pdf", bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {plots_dir / 'icrmsd_comparison.pdf'}")

    # Plot 5: ICRMSD CDFs by type
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    xlims = {
        "icrmsd_bond": (0, 0.1),
        "icrmsd_angle": (0, 10),
        "icrmsd_dihedral": (0, 50),
        "icrmsd_improper": (0, 20),
    }

    for idx, (ic_type, ic_key, unit) in enumerate(zip(ic_types, ic_keys, ic_units)):
        ax = axes[idx]
        for i, ff in enumerate(force_fields):
            data = np.array(ff_data[ff][ic_key])
            data = data[np.isfinite(data)]
            if len(data) > 0:
                sorted_data = np.sort(data)
                cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
                ax.plot(sorted_data, cdf, "-", label=ff, color=colors[i])

        ax.set_xlabel(f"{ic_type} RMSD ({unit})")
        ax.set_ylabel("CDF")
        ax.set_title(f"{ic_type} Internal Coordinate RMSD")
        ax.set_xlim(xlims[ic_key])
        ax.set_ylim(-0.05, 1.05)
        ax.legend(loc="best", fontsize=8)

    fig.suptitle("Internal Coordinate RMSD - Cumulative Distributions", fontsize=14)
    fig.tight_layout()
    fig.savefig(plots_dir / "icrmsd_by_type.pdf", bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {plots_dir / 'icrmsd_by_type.pdf'}")

    logger.info(f"Basic plots saved to: {plots_dir}")


def generate_parameter_type_plots(
    param_summary: dict[str, dict[str, dict[str, float]]],
    output_dir: pathlib.Path,
) -> None:
    """Generate plots showing deviations by parameter type (SMIRKS).

    Parameters
    ----------
    param_summary : dict
        Summary statistics from compute_parameter_type_summary().
    output_dir : pathlib.Path
        Directory to save plot files.
    """
    logger.info("Generating parameter type breakdown plots...")

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    ic_type_info = {
        "Bond": {"unit": "Å", "title": "Bond Length Deviations"},
        "Angle": {"unit": "°", "title": "Angle Deviations"},
        "Dihedral": {"unit": "°", "title": "Dihedral Angle Deviations"},
        "Improper": {"unit": "°", "title": "Improper Torsion Deviations"},
    }

    for ic_type, info in ic_type_info.items():
        if ic_type not in param_summary or not param_summary[ic_type]:
            logger.warning(f"No data for {ic_type}, skipping...")
            continue

        # Sort parameters by count (most common first)
        all_params = sorted(
            param_summary[ic_type].items(),
            key=lambda x: x[1].get("count", 0),
            reverse=True,
        )

        if not all_params:
            continue

        smirks_labels = [p[0] for p in all_params]
        means = [p[1]["mean"] for p in all_params]
        stds = [p[1]["std"] for p in all_params]
        counts = [p[1]["count"] for p in all_params]

        # Plot 1: Horizontal bar chart of mean deviations (axes swapped)
        # Dynamic figure height based on number of parameters
        fig_height = max(8, len(all_params) * 0.3)
        fig, ax = plt.subplots(figsize=(12, fig_height))

        y = np.arange(len(smirks_labels))
        bars = ax.barh(
            y, means, xerr=stds, capsize=2, color="steelblue", alpha=0.8, height=0.7
        )

        ax.set_xlabel(f"Mean Absolute Deviation ({info['unit']})")
        ax.set_ylabel("SMIRKS Parameter ID")
        ax.set_title(f"{info['title']} by Parameter Type Sorted by Count")
        ax.set_yticks(y)
        ax.set_yticklabels(smirks_labels, fontsize=7)
        ax.set_xlim(left=0)  # Set x-axis minimum to 0

        # Invert y-axis so most common is at top
        ax.invert_yaxis()

        # Add count labels on bars
        for i, (bar, count) in enumerate(zip(bars, counts)):
            ax.annotate(
                f"n={count}",
                xy=(bar.get_width(), bar.get_y() + bar.get_height() / 2),
                xytext=(3, 3),
                textcoords="offset points",
                ha="left",
                va="center",
                fontsize=6,
            )

        fig.tight_layout()
        fig.savefig(
            plots_dir / f"deviation_by_{ic_type.lower()}_type.pdf", bbox_inches="tight"
        )
        plt.close(fig)
        logger.info(f"Saved: {plots_dir / f'deviation_by_{ic_type.lower()}_type.pdf'}")

        # Plot 2: Horizontal box plot for distribution (ALL parameters, axes swapped)
        fig_height = max(8, len(all_params) * 0.3)
        fig, ax = plt.subplots(figsize=(12, fig_height))

        labels_all = [p[0][:40] + "..." if len(p[0]) > 40 else p[0] for p in all_params]

        # Create synthetic box plot data from summary stats
        box_data = []
        for smirks, stats in all_params:
            mean_val = stats["mean"]
            std = stats["std"]
            median = stats["median"]
            min_val = stats["min"]
            max_val = stats["max"]
            # Use actual stats for box plot approximation
            q1 = max(min_val, median - std * 0.675)  # Approximate Q1
            q3 = min(max_val, median + std * 0.675)  # Approximate Q3
            box_data.append(
                {
                    "min": min_val,
                    "q1": q1,
                    "median": median,
                    "q3": q3,
                    "max": max_val,
                    "mean": mean_val,
                }
            )

        positions = np.arange(len(all_params))

        # Draw horizontal box plots manually for better control
        for i, (pos, data) in enumerate(zip(positions, box_data)):
            # Box from Q1 to Q3 (horizontal)
            box_width = data["q3"] - data["q1"]
            ax.barh(
                pos,
                box_width,
                left=data["q1"],
                height=0.6,
                color="lightsteelblue",
                edgecolor="black",
                linewidth=1,
            )

            # Median line (vertical)
            ax.vlines(data["median"], pos - 0.3, pos + 0.3, colors="black", linewidth=2)

            # Mean marker
            ax.plot(
                data["mean"],
                pos,
                "D",
                color="red",
                markersize=4,
                label="Mean" if i == 0 else "",
            )

            # Whiskers (horizontal)
            ax.hlines(pos, data["min"], data["q1"], colors="black", linewidth=1)
            ax.hlines(pos, data["q3"], data["max"], colors="black", linewidth=1)

            # Whisker caps (vertical)
            ax.vlines(data["min"], pos - 0.15, pos + 0.15, colors="black", linewidth=1)
            ax.vlines(data["max"], pos - 0.15, pos + 0.15, colors="black", linewidth=1)

        ax.set_xlabel(f"Deviation ({info['unit']})")
        ax.set_ylabel("SMIRKS Parameter ID")
        ax.set_title(
            f"{info['title']} Distribution by Parameter Type (All {len(all_params)} parameters)"
        )
        ax.set_yticks(positions)
        ax.set_yticklabels(labels_all, fontsize=7)
        ax.set_ylim(-0.5, len(all_params) - 0.5)
        ax.invert_yaxis()  # Most common at top
        ax.legend(loc="lower right")

        fig.tight_layout()
        fig.savefig(
            plots_dir / f"deviation_dist_{ic_type.lower()}_type.pdf",
            bbox_inches="tight",
        )
        plt.close(fig)
        logger.info(
            f"Saved: {plots_dir / f'deviation_dist_{ic_type.lower()}_type.pdf'}"
        )

    # Summary plot: All IC types comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    for idx, (ic_type, info) in enumerate(ic_type_info.items()):
        ax = axes[idx]

        if ic_type not in param_summary or not param_summary[ic_type]:
            ax.text(
                0.5,
                0.5,
                f"No {ic_type} data",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title(info["title"])
            continue

        # Get all mean deviations for this IC type
        all_means = [stats["mean"] for stats in param_summary[ic_type].values()]
        all_counts = [stats["count"] for stats in param_summary[ic_type].values()]

        # Weighted histogram
        ax.hist(
            all_means,
            bins=30,
            weights=all_counts,
            color="steelblue",
            alpha=0.7,
            edgecolor="black",
        )
        ax.set_xlabel(f"Mean Deviation ({info['unit']})")
        ax.set_ylabel("Count (weighted by occurrences)")
        ax.set_title(info["title"])

        # Add summary stats
        weighted_mean = np.average(all_means, weights=all_counts) if all_means else 0
        ax.axvline(
            weighted_mean,
            color="red",
            linestyle="--",
            label=f"Weighted Mean: {weighted_mean:.3f}",
        )
        ax.legend()

    fig.suptitle("Internal Coordinate Deviation Distributions by Type", fontsize=14)
    fig.tight_layout()
    fig.savefig(plots_dir / "deviation_summary_all_types.pdf", bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {plots_dir / 'deviation_summary_all_types.pdf'}")

    logger.info(f"Parameter type plots saved to: {plots_dir}")


def print_summary(
    metrics: dict, param_summary: dict | None = None, offxml_path: str | None = None
) -> None:
    """Print summary of benchmark results to console.

    Parameters
    ----------
    metrics : dict
        Loaded metrics data.
    param_summary : dict, optional
        Parameter type summary statistics.
    offxml_path : str, optional
        Path to force field file for looking up SMIRKS patterns.
    """
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)

    ff_metrics = metrics.get("metrics", metrics)

    for force_field, records in ff_metrics.items():
        print(f"\nForce Field: {force_field}")
        print("-" * 40)

        ddes = []
        rmsds = []
        tfds = []

        for record_id, metric in records.items():
            if isinstance(metric, dict):
                if metric.get("dde") is not None:
                    ddes.append(metric["dde"])
                if metric.get("rmsd") is not None:
                    rmsds.append(metric["rmsd"])
                if metric.get("tfd") is not None:
                    tfds.append(metric["tfd"])

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

    if param_summary:
        print("\n" + "=" * 60)
        print("TOP PARAMETER TYPES BY MEAN DEVIATION")
        print("=" * 60)

        # Try to load force field for SMIRKS lookup
        smirks_lookup = {}
        if offxml_path:
            try:
                ff = ForceField(offxml_path)
                # Build lookup from parameter ID to SMIRKS
                for handler_name in [
                    "Bonds",
                    "Angles",
                    "ProperTorsions",
                    "ImproperTorsions",
                ]:
                    handler = ff.get_parameter_handler(handler_name)
                    for param in handler.parameters:
                        smirks_lookup[param.id] = param.smirks
            except Exception as e:
                logger.debug(f"Could not load force field for SMIRKS lookup: {e}")

        for ic_type in ["Bond", "Angle", "Dihedral", "Improper"]:
            if ic_type not in param_summary or not param_summary[ic_type]:
                continue

            unit = "Å" if ic_type == "Bond" else "°"
            print(f"\n{ic_type}s ({unit}):")
            print("-" * 80)

            # Sort by mean deviation (largest first)
            params_sorted = sorted(
                param_summary[ic_type].items(),
                key=lambda x: x[1].get("mean", 0),
                reverse=True,
            )

            for i, (param_id, data) in enumerate(params_sorted[:5], 1):
                count = data.get("count", 0)
                mean_err = data.get("mean", float("nan"))
                smirks = smirks_lookup.get(param_id, "")
                smirks_suffix = f", SMIRKS: {smirks}" if smirks else ""
                print(
                    f"  {i}. {param_id}: mean={mean_err:.4f}{unit}, n={count}{smirks_suffix}"
                )


def main():
    parser = argparse.ArgumentParser(
        description="Generate benchmark plots from existing results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    Basic plots from metrics.json only:
        python plot_benchmark.py --metrics benchmark_results/metrics.json

    Full plots including parameter type analysis (from database):
        python plot_benchmark.py --metrics benchmark_results/metrics.json \\
            --database benchmark.sqlite --offxml openff-2.2.1.offxml

    Using pre-computed parameter type analysis:
        python plot_benchmark.py --metrics benchmark_results/metrics.json \\
            --param-json benchmark_results/metrics_by_parameter_type.json

Output Files:
    - plots/dde_histogram.pdf           : DDE distribution histogram
    - plots/rmsd_cdf.pdf                : RMSD cumulative distribution
    - plots/tfd_cdf.pdf                 : TFD cumulative distribution
    - plots/icrmsd_comparison.pdf       : Bar chart of ICRMSD by type
    - plots/icrmsd_by_type.pdf          : CDF plots for each IC type
    - plots/deviation_by_*_type.pdf     : Mean deviation by SMIRKS (requires param analysis)
    - plots/deviation_dist_*_type.pdf   : Deviation distributions by parameter type
    - plots/deviation_summary_all_types.pdf : Summary of all IC type deviations
        """,
    )
    parser.add_argument(
        "--metrics",
        type=str,
        required=True,
        help="Path to metrics.json file from benchmarking run",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Directory to save plot files (default: current directory)",
    )
    parser.add_argument(
        "--database",
        type=str,
        default=None,
        help="Path to benchmark.sqlite for parameter type analysis",
    )
    parser.add_argument(
        "--offxml",
        type=str,
        default=None,
        help="Path to OFFXML force field file (required with --database)",
    )
    parser.add_argument(
        "--param-json",
        type=str,
        default=None,
        help="Path to existing metrics_by_parameter_type.json",
    )
    parser.add_argument(
        "--n-processes",
        type=int,
        default=1,
        help="Number of parallel processes for parameter type analysis (default: 1)",
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
        logger.add(sys.stdout, level="INFO")
    elif args.verbose == 1:
        logger.add(sys.stdout, level="WARNING")
    elif args.verbose == 2:
        logger.add(sys.stdout, level="INFO")
    elif args.verbose >= 3:
        logger.add(sys.stdout, level="DEBUG")

    metrics_path = pathlib.Path(args.metrics)
    output_dir = pathlib.Path(args.output_dir)

    if not metrics_path.exists():
        logger.error(f"Metrics file not found: {metrics_path}")
        sys.exit(1)

    # Load metrics
    metrics = load_metrics(metrics_path)

    # Generate basic plots
    generate_basic_plots(metrics, output_dir)

    # Handle parameter type analysis
    param_summary = None

    if args.param_json:
        # Load from explicitly provided JSON path
        param_path = pathlib.Path(args.param_json)
        if param_path.exists():
            param_summary = load_parameter_type_summary(param_path)
        else:
            logger.warning(f"Parameter type JSON not found: {param_path}")

    else:
        # Auto-detect metrics_by_parameter_type.json in output directory
        auto_param_path = output_dir / "metrics_by_parameter_type.json"
        if auto_param_path.exists():
            logger.info(f"Found existing parameter type analysis: {auto_param_path}")
            param_summary = load_parameter_type_summary(auto_param_path)

        elif args.database and args.offxml:
            database_path = pathlib.Path(args.database)
            offxml_path = pathlib.Path(args.offxml)

            if not database_path.exists():
                logger.error(f"Database not found: {database_path}")
            elif not offxml_path.exists():
                logger.error(f"Force field not found: {offxml_path}")
            else:
                try:
                    store = MoleculeStore(str(database_path))
                    param_deviations = analyze_by_parameter_type(
                        store, str(offxml_path), n_processes=args.n_processes
                    )
                    param_summary = compute_parameter_type_summary(param_deviations)

                    # Save for future use
                    param_output = output_dir / "metrics_by_parameter_type.json"
                    with open(param_output, "w") as f:
                        json.dump(param_summary, f, indent=2)
                    logger.info(f"Saved parameter type analysis to: {param_output}")

                except Exception as e:
                    logger.error(f"Parameter type analysis failed: {e}")

        elif args.database or args.offxml:
            logger.warning(
                "Both --database and --offxml are required for parameter type analysis"
            )

    # Generate parameter type plots if available
    if param_summary:
        generate_parameter_type_plots(param_summary, output_dir)

    # Print summary (pass offxml for SMIRKS lookup if available)
    print_summary(metrics, param_summary, args.offxml)

    logger.info("Plot generation complete!")


if __name__ == "__main__":
    freeze_support()
    main()
