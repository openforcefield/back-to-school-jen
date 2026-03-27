"""Generate benchmark plots from existing benchmark results.

Parameter summaries are only produced if a single dataset
is passed.

This script reads the output files from benchmarking.py and generates
visualization plots. It can work with metrics.json alone for basic plots,
or with benchmark.sqlite and an offxml file for parameter type analysis.

Supports comparing multiple force fields by providing multiple input files.

Command-line Arguments
----------------------
--metrics : str (one or more)
    Path(s) to metrics.json file(s) from benchmarking run(s).
--labels : str (optional)
    Labels for each force field (defaults to filenames).
--output-dir : str
    Directory to save plot files (default: current directory).
--database : str, optional
    Path(s) to benchmark.sqlite for parameter type analysis.
--offxml : str, optional
    Path(s) to OFFXML force field file(s) (required for parameter type analysis).
--param-json : str, optional
    Path(s) to existing metrics_by_parameter_type.json (alternative to database+offxml).
--plot-difference : flag, optional
    If set and multiple metrics files are provided, subtract values from the first-listed
    force field from all subsequent force fields before plotting. Useful to visualize
    differences relative to a reference.

Examples
--------
Basic plots from metrics.json only:
    $ python plot_benchmark.py --metrics benchmark_results/metrics.json

Compare multiple force fields:
    $ python plot_benchmark.py --metrics ff1/metrics.json ff2/metrics.json \\
        --labels "OpenFF 2.2.1" "Custom FF" --output-dir comparison/

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

LINEWIDTH = 2
ALPHA = 0.4

EXTENSION = "png"


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
    metrics_list: list[dict],
    labels: list[str],
    output_dir: pathlib.Path,
    plot_difference: bool = False,
) -> None:
    """Generate basic benchmark plots from metrics.json files.

    Parameters
    ----------
    metrics_list : list[dict]
        List of loaded metrics data dictionaries (for multiple FFs).
    labels : list[str], optional
        Labels for each force field. If None
        uses force field names from the metrics file.
    output_dir : pathlib.Path or str
        Directory to save plot files (default: current directory).
    """
    logger.info("Generating basic benchmark plots...")

    output_dir = pathlib.Path(output_dir)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_diff" if plot_difference else ""

    if labels is None:
        labels = [f"FF_{i+1}" for i in range(len(metrics_list))]
    elif len(labels) != len(metrics_list):
        raise ValueError(
            f"Number of metrics files ({len(metrics_list)}) must match number of labels ({len(labels)})"
        )

    # Generate colors and style cycles for accessibility (not relying on color alone)
    n_ff = len(labels)
    colors = plt.cm.tab10(np.linspace(0, 1, max(n_ff, 1)))
    line_styles = ["-", "--", ":", "-."]
    hatches = ["", "//", "\\\\", "xx", ".."]

    # Extract metrics by force field label. Use dicts keyed by record id so
    # we can compute differences between matching records when requested.
    ff_data: dict[str, dict[str, dict]] = {}
    for metrics, label in zip(metrics_list, labels):
        ff_metrics = metrics.get("metrics", metrics)
        ff_data[label] = {
            "dde": {},
            "rmsd": {},
            "tfd": {},
            "icrmsd_bond": {},
            "icrmsd_angle": {},
            "icrmsd_dihedral": {},
            "icrmsd_improper": {},
        }
        # Iterate through all force fields in this metrics file
        for ff_name, records in ff_metrics.items():
            for rec_id, metric in records.items():
                if not isinstance(metric, dict):
                    continue
                if metric.get("dde") is not None:
                    ff_data[label]["dde"][rec_id] = metric["dde"]
                if metric.get("rmsd") is not None:
                    ff_data[label]["rmsd"][rec_id] = metric["rmsd"]
                if metric.get("tfd") is not None:
                    ff_data[label]["tfd"][rec_id] = metric["tfd"]
                if metric.get("icrmsd"):
                    icrmsd = metric["icrmsd"]
                    if icrmsd.get("Bond") is not None:
                        ff_data[label]["icrmsd_bond"][rec_id] = icrmsd["Bond"]
                    if icrmsd.get("Angle") is not None:
                        ff_data[label]["icrmsd_angle"][rec_id] = icrmsd["Angle"]
                    if icrmsd.get("Dihedral") is not None:
                        ff_data[label]["icrmsd_dihedral"][rec_id] = icrmsd["Dihedral"]
                    if icrmsd.get("Improper") is not None:
                        ff_data[label]["icrmsd_improper"][rec_id] = icrmsd["Improper"]

    # Helper: extract all values for a metric for each FF (no record alignment)
    def get_data(metric_key: str) -> dict[str, np.ndarray]:
        """Extract metric data for all force fields independently.

        Parameters
        ----------
        metric_key : str
            Key in ff_data (e.g., "dde", "rmsd", "tfd", "icrmsd_bond").

        Returns
        -------
        dict[str, np.ndarray]
            Mapping of label to data array (all available records, unaligned).
        """
        return {
            label: np.array(list(ff_data[label][metric_key].values()))
            for label in labels
        }

    # Helper: evaluate empirical CDF at specified x-grid points via interpolation
    def evaluate_cdf(data: np.ndarray, x_grid: np.ndarray) -> np.ndarray:
        """Evaluate empirical CDF on a grid of x values."""
        sorted_data = np.sort(data)
        cdf_vals = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        return np.interp(x_grid, sorted_data, cdf_vals, left=0.0, right=1.0)

    # Plot 1: DDE Histogram
    dde_bins = np.linspace(-15, 15, 31)
    fig, ax = plt.subplots(figsize=(10, 6))
    dde_data = get_data("dde")
    # Determine which labels to actually plot (skip reference when showing diffs)
    if plot_difference and n_ff > 1:
        plot_indices = list(range(1, n_ff))
    else:
        plot_indices = list(range(0, n_ff))

    logger.info(
        f"plot_difference={plot_difference}, n_ff={n_ff}, plot_indices={plot_indices}"
    )

    if plot_difference and n_ff > 1:
        ref_dde = dde_data[labels[0]]
        ref_dde = ref_dde[np.isfinite(ref_dde)]
        ref_counts, _ = np.histogram(ref_dde, bins=dde_bins)
        for i in plot_indices:
            label = labels[i]
            data = dde_data[label]
            data = data[np.isfinite(data)]
            if len(data) > 0:
                counts, _ = np.histogram(data, bins=dde_bins)
                ax.stairs(
                    counts - ref_counts,
                    dde_bins,
                    label=label,
                    color=colors[i % len(colors)],
                    linestyle=line_styles[i % len(line_styles)],
                )
    else:
        for i in plot_indices:
            label = labels[i]
            data = dde_data[label]
            data = data[np.isfinite(data)]
            if len(data) > 0:
                counts, _ = np.histogram(data, bins=dde_bins)
                ax.stairs(
                    counts,
                    dde_bins,
                    label=label,
                    color=colors[i % len(colors)],
                    linestyle=line_styles[i % len(line_styles)],
                )
    ax.set_xlabel("DDE (kcal/mol)")
    ax.set_ylabel("Count")
    if plot_difference:
        ax.set_title(f"Deformation-Driven Energy Distribution Relative to {labels[0]}")
    else:
        ax.set_title("Deformation-Driven Energy Distribution")
    ax.legend(loc="best")
    # Draw a zero reference line when plotting differences so viewers can see
    # positive/negative deviations. Do not add a legend entry (label=None).
    if plot_difference:
        ax.axvline(
            x=0,
            color="black",
            linestyle="--",
            linewidth=LINEWIDTH,
            label=None,
            alpha=ALPHA,
        )
        # Use a symmetric log scale on y to show orders-of-magnitude differences
        # try:
        #    ax.set_yscale("symlog", linthresh=1)
        #    logger.info("DDE histogram: using symlog y-scale for difference plot")
        # except Exception:
        #    logger.debug("Could not set symlog y-scale for DDE histogram")
    else:
        ax.axvline(x=0, color="gray", linestyle="--", linewidth=LINEWIDTH, alpha=ALPHA)
    fig.tight_layout()
    fig.savefig(plots_dir / f"dde_histogram{suffix}.{EXTENSION}", bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {plots_dir / f'dde_histogram{suffix}.{EXTENSION}'}")

    # Plot 2: RMSD CDF
    fig, ax = plt.subplots(figsize=(10, 6))
    rmsd_data = get_data("rmsd")
    if plot_difference and n_ff > 1:
        rmsd_x_grid = np.linspace(0, 3.0, 500)
        ref_rmsd = rmsd_data[labels[0]]
        ref_rmsd = ref_rmsd[np.isfinite(ref_rmsd)]
        ref_cdf = evaluate_cdf(ref_rmsd, rmsd_x_grid)
        for i in plot_indices:
            label = labels[i]
            data = rmsd_data[label]
            data = data[np.isfinite(data)]
            if len(data) > 0:
                cdf_i = evaluate_cdf(data, rmsd_x_grid)
                ls = line_styles[i % len(line_styles)]
                ax.plot(
                    rmsd_x_grid,
                    cdf_i - ref_cdf,
                    linestyle=ls,
                    label=label,
                    linewidth=LINEWIDTH,
                    color=colors[i % len(colors)],
                )
        ax.axhline(y=0, color="black", linestyle="--", linewidth=LINEWIDTH, alpha=ALPHA)
    else:
        for i in plot_indices:
            label = labels[i]
            data = np.sort(rmsd_data[label])
            data = data[np.isfinite(data)]
            if len(data) > 0:
                cdf = np.arange(1, len(data) + 1) / len(data)
                ls = line_styles[i % len(line_styles)]
                ax.plot(
                    data,
                    cdf,
                    linestyle=ls,
                    label=label,
                    linewidth=LINEWIDTH,
                    color=colors[i % len(colors)],
                )
        ax.set_ylim(0, 1.05)
    ax.set_xlabel("RMSD (Å)")
    ax.set_ylabel("CDF" if not plot_difference else "ΔCDF")
    ax.set_title("Root Mean Square Deviation - Cumulative Distribution")
    ax.set_xlim(0, 3.0)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(plots_dir / f"rmsd_cdf{suffix}.{EXTENSION}", bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {plots_dir / f'rmsd_cdf{suffix}.{EXTENSION}'}")

    # Plot 3: TFD CDF
    fig, ax = plt.subplots(figsize=(10, 6))
    tfd_data = get_data("tfd")
    if plot_difference and n_ff > 1:
        tfd_x_grid = np.linspace(0, 0.5, 500)
        ref_tfd = tfd_data[labels[0]]
        ref_tfd = ref_tfd[np.isfinite(ref_tfd)]
        ref_cdf = evaluate_cdf(ref_tfd, tfd_x_grid)
        for i in plot_indices:
            label = labels[i]
            data = tfd_data[label]
            data = data[np.isfinite(data)]
            if len(data) > 0:
                cdf_i = evaluate_cdf(data, tfd_x_grid)
                ls = line_styles[i % len(line_styles)]
                ax.plot(
                    tfd_x_grid,
                    cdf_i - ref_cdf,
                    linestyle=ls,
                    label=label,
                    linewidth=LINEWIDTH,
                    color=colors[i % len(colors)],
                )
        ax.axhline(y=0, color="black", linestyle="--", linewidth=LINEWIDTH, alpha=ALPHA)
    else:
        for i in plot_indices:
            label = labels[i]
            data = np.sort(tfd_data[label])
            data = data[np.isfinite(data)]
            if len(data) > 0:
                cdf = np.arange(1, len(data) + 1) / len(data)
                ls = line_styles[i % len(line_styles)]
                ax.plot(
                    data,
                    cdf,
                    linestyle=ls,
                    label=label,
                    linewidth=LINEWIDTH,
                    color=colors[i % len(colors)],
                )
        ax.set_ylim(0, 1.05)
    ax.set_xlabel("TFD")
    ax.set_ylabel("CDF" if not plot_difference else "ΔCDF")
    ax.set_title("Torsion Fingerprint Deviation - Cumulative Distribution")
    ax.set_xlim(0, 0.5)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(plots_dir / f"tfd_cdf{suffix}.{EXTENSION}", bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {plots_dir / f'tfd_cdf{suffix}.{EXTENSION}'}")

    # Plot 4: ICRMSD Comparison Bar Chart
    ic_types = ["Bond", "Angle", "Dihedral", "Improper"]
    ic_keys = ["icrmsd_bond", "icrmsd_angle", "icrmsd_dihedral", "icrmsd_improper"]
    ic_units = ["Å", "°", "°", "°"]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    plot_n = len(plot_indices)
    x = np.arange(plot_n)
    width = 0.6

    for idx, (ic_type, ic_key, unit) in enumerate(zip(ic_types, ic_keys, ic_units)):
        ax = axes[idx]
        ic_data = get_data(ic_key)
        means = []
        stds = []
        ref_mean = 0.0
        if plot_difference and n_ff > 1:
            ref_ic = ic_data[labels[0]]
            ref_ic = ref_ic[np.isfinite(ref_ic)] if ref_ic.size > 0 else np.array([])
            ref_mean = np.mean(ref_ic) if ref_ic.size > 0 else 0.0
        for idx_i, i in enumerate(plot_indices):
            label = labels[i]
            data = ic_data[label]
            data = data[np.isfinite(data)] if data.size > 0 else np.array([])
            if data.size > 0:
                m = np.mean(data)
                if plot_difference and n_ff > 1:
                    m = m - ref_mean
                means.append(m)
                stds.append(np.std(data))
            else:
                means.append(0)
                stds.append(0)

        bars = ax.bar(
            x,
            means,
            width,
            yerr=stds,
            capsize=3,
            color=[colors[i % len(colors)] for i in plot_indices],
        )
        # add hatches to bars to avoid using color only
        for bi, bar in enumerate(bars):
            hatch = hatches[bi % len(hatches)]
            bar.set_hatch(hatch)
            bar.set_edgecolor("black")
        # Draw horizontal zero reference for difference plots (bar heights)
        if plot_difference:
            ax.axhline(
                y=0,
                color="black",
                linestyle="--",
                linewidth=LINEWIDTH,
                label=None,
                alpha=ALPHA,
            )
        ax.set_ylabel(f"ICRMSD ({unit})")
        ax.set_title(f"{ic_type} RMSD")
        if not plot_difference:
            ax.set_ylim(ymin=0)
        ax.set_xticks(x)
        ax.set_xticklabels(
            [labels[i] for i in plot_indices], rotation=45, ha="right", fontsize=8
        )

    fig.suptitle("Internal Coordinate RMSD by Type", fontsize=14)
    fig.tight_layout()
    fig.savefig(
        plots_dir / f"icrmsd_comparison{suffix}.{EXTENSION}", bbox_inches="tight"
    )
    plt.close(fig)
    logger.info(f"Saved: {plots_dir / f'icrmsd_comparison{suffix}.{EXTENSION}'}")

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
        ic_data = get_data(ic_key)
        xlim = xlims[ic_key]
        if plot_difference and n_ff > 1:
            x_grid = np.linspace(xlim[0], xlim[1], 500)
            ref_ic = ic_data[labels[0]]
            ref_ic = ref_ic[np.isfinite(ref_ic)]
            ref_cdf = (
                evaluate_cdf(ref_ic, x_grid)
                if ref_ic.size > 0
                else np.zeros_like(x_grid)
            )
            for i in plot_indices:
                label = labels[i]
                data = ic_data[label]
                data = data[np.isfinite(data)]
                if data.size > 0:
                    cdf_i = evaluate_cdf(data, x_grid)
                    ls = line_styles[i % len(line_styles)]
                    ax.plot(
                        x_grid,
                        cdf_i - ref_cdf,
                        linestyle=ls,
                        label=label,
                        linewidth=LINEWIDTH,
                        color=colors[i % len(colors)],
                    )
            ax.axhline(
                y=0, color="black", linestyle="--", linewidth=LINEWIDTH, alpha=ALPHA
            )
        else:
            for i in plot_indices:
                label = labels[i]
                data = ic_data[label]
                data = data[np.isfinite(data)]
                if data.size > 0:
                    sorted_data = np.sort(data)
                    cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
                    ls = line_styles[i % len(line_styles)]
                    ax.plot(
                        sorted_data,
                        cdf,
                        linestyle=ls,
                        label=label,
                        linewidth=LINEWIDTH,
                        color=colors[i % len(colors)],
                    )
            ax.set_ylim(0, 1.05)

        ax.set_xlabel(f"{ic_type} RMSD ({unit})")
        ax.set_ylabel("CDF" if not plot_difference else "ΔCDF")
        ax.set_title(f"{ic_type} Internal Coordinate RMSD")
        ax.set_xlim(xlim)
        ax.legend(loc="best", fontsize=8)

    fig.suptitle("Internal Coordinate RMSD - Cumulative Distributions", fontsize=14)
    fig.tight_layout()
    fig.savefig(plots_dir / f"icrmsd_by_type{suffix}.{EXTENSION}", bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {plots_dir / f'icrmsd_by_type{suffix}.{EXTENSION}'}")

    logger.info(f"Basic plots saved to: {plots_dir}")


def generate_parameter_type_plots(
    param_summaries: list[dict[str, dict[str, dict[str, float]]]],
    labels: list[str] | None = None,
    output_dir: pathlib.Path | str = ".",
    plot_difference: bool = False,
) -> None:
    """Generate plots showing deviations by parameter type (SMIRKS).

    Parameters
    ----------
    param_summaries : list[dict]
        List of summary statistics from compute_parameter_type_summary() (one per FF).
    labels : list[str], optional
        Labels for each force field. If None uses "Force Field" as the label.
    output_dir : pathlib.Path or str
        Directory to save plot files (default: current directory).
    """
    logger.info("Generating parameter type breakdown plots...")

    output_dir = pathlib.Path(output_dir)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    if labels is None:
        labels = [f"FF_{i+1}" for i in range(len(param_summaries))]
    elif len(param_summaries) != len(labels):
        raise ValueError(
            f"Number of parameter summaries ({len(param_summaries)}) must match number of labels ({len(labels)})"
        )

    n_ff = len(labels)
    colors = plt.cm.tab10(np.linspace(0, 1, max(n_ff, 1)))
    hatches = ["", "//", "\\\\", "xx", ".."]

    # If plotting differences, build a display copy of summaries where each
    # subsequent FF (index > 0) has its means subtracted by the first FF's mean
    display_summaries = []
    if plot_difference and len(param_summaries) > 1:
        base = param_summaries[0]
        for idx, summary in enumerate(param_summaries):
            if idx == 0:
                display_summaries.append(summary)
                continue
            # Build a new adjusted summary
            adjusted: dict[str, dict[str, dict[str, float]]] = {}
            for ic_type, smirks_dict in summary.items():
                adjusted[ic_type] = {}
                base_dict = base.get(ic_type, {})
                for param_id, stats in smirks_dict.items():
                    base_mean = base_dict.get(param_id, {}).get("mean")
                    if base_mean is None:
                        adjusted_mean = stats.get("mean", 0.0)
                    else:
                        adjusted_mean = stats.get("mean", 0.0) - base_mean
                    # Copy other stats as-is (counts, std may not be meaningful for diffs)
                    adjusted[ic_type][param_id] = {
                        "mean": adjusted_mean,
                        "std": stats.get("std", 0.0),
                        "count": stats.get("count", 0),
                        "min": stats.get("min", 0.0),
                        "max": stats.get("max", 0.0),
                    }
            display_summaries.append(adjusted)
    else:
        display_summaries = param_summaries

    ic_type_info = {
        "Bond": {"unit": "Å", "title": "Bond Length Deviations"},
        "Angle": {"unit": "°", "title": "Angle Deviations"},
        "Dihedral": {"unit": "°", "title": "Dihedral Angle Deviations"},
        "Improper": {"unit": "°", "title": "Improper Torsion Deviations"},
    }

    # Decide which FF indices to plot (skip base when showing differences)
    suffix = "_diff" if plot_difference else ""
    if plot_difference and n_ff > 1:
        plot_indices = list(range(1, n_ff))
    else:
        plot_indices = list(range(0, n_ff))

    for ic_type, info in ic_type_info.items():
        # Collect all unique parameter IDs across all force fields
        all_param_ids: set[str] = set()
        for param_summary in param_summaries:
            if ic_type in param_summary:
                all_param_ids.update(param_summary[ic_type].keys())

        if not all_param_ids:
            logger.warning(f"No data for {ic_type}, skipping...")
            continue

        # Use first FF's ordering (by count) as the reference
        first_summary = param_summaries[0].get(ic_type, {})
        all_params_sorted = sorted(
            all_param_ids,
            key=lambda x: first_summary.get(x, {}).get("count", 0),
            reverse=True,
        )

        # Plot 1: Grouped horizontal bar chart of mean deviations
        fig_height = max(8, len(all_params_sorted) * 0.4)
        fig, ax = plt.subplots(figsize=(14, fig_height))

        y = np.arange(len(all_params_sorted))
        bar_height = 0.8 / max(1, len(plot_indices))

        for idx_pos, ff_idx in enumerate(plot_indices):
            param_summary = display_summaries[ff_idx]
            label = labels[ff_idx]
            ic_data = param_summary.get(ic_type, {})
            means = []
            stds = []
            for param_id in all_params_sorted:
                if param_id in ic_data:
                    means.append(ic_data[param_id]["mean"])
                    stds.append(ic_data[param_id]["std"])
                else:
                    means.append(0)
                    stds.append(0)

            y_positions = y - 0.4 + bar_height * (idx_pos + 0.5)
            bar = ax.barh(
                y_positions,
                means,
                xerr=stds,
                height=bar_height,
                label=label,
                color=colors[ff_idx % len(colors)],
                alpha=ALPHA,
                capsize=1,
            )
            # add hatch for accessibility
            for b in bar:
                b.set_edgecolor("black")
                b.set_hatch(hatches[idx_pos % len(hatches)])

        ax.set_xlabel(f"Mean Absolute Deviation ({info['unit']})")
        ax.set_ylabel("SMIRKS Parameter ID")
        ax.set_title(f"{info['title']} by Parameter Type")
        ax.set_yticks(y)
        ax.set_yticklabels(all_params_sorted, fontsize=7)
        if plot_difference:
            # Expand x-limits to include negative values when plotting differences
            all_means_vals = []
            for ps in display_summaries:
                vals = [v["mean"] for v in ps.get(ic_type, {}).values()]
                all_means_vals.extend(vals)
            if all_means_vals:
                minv = min(all_means_vals)
                maxv = max(all_means_vals)
                margin = max((maxv - minv) * 0.05, 1e-6)
                ax.set_xlim(minv - margin, maxv + margin)
            # Draw zero reference line
            ax.axvline(
                x=0,
                color="black",
                linestyle="--",
                linewidth=LINEWIDTH,
                label=None,
                alpha=ALPHA,
            )
        else:
            ax.set_xlim(left=0)
        ax.invert_yaxis()
        ax.legend(loc="lower right")

        fig.tight_layout()
        fig.tight_layout()
        fig.savefig(
            plots_dir / f"deviation_by_{ic_type.lower()}_type{suffix}.{EXTENSION}",
            bbox_inches="tight",
        )
        plt.close(fig)
        logger.info(
            f"Saved: {plots_dir / f'deviation_by_{ic_type.lower()}_type{suffix}.{EXTENSION}'}"
        )

    # Summary plot: All IC types comparison across force fields
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    for idx, (ic_type, info) in enumerate(ic_type_info.items()):
        ax = axes[idx]
        has_data = False
        for idx_pos, ff_idx in enumerate(plot_indices):
            param_summary = display_summaries[ff_idx]
            label = labels[ff_idx]
            if ic_type not in param_summary or not param_summary[ic_type]:
                continue
            has_data = True

            all_means = [stats["mean"] for stats in param_summary[ic_type].values()]
            all_counts = [stats["count"] for stats in param_summary[ic_type].values()]

            if all_means:
                weighted_mean = np.average(all_means, weights=all_counts)
                patches = ax.hist(
                    all_means,
                    bins=30,
                    weights=all_counts,
                    color=colors[ff_idx % len(colors)],
                    alpha=ALPHA,
                    edgecolor="black",
                    label=f"{label} (μ={weighted_mean:.3f})",
                )
                # Apply hatch pattern to histogram bars for accessibility
                for p in patches[2]:
                    p.set_hatch(hatches[idx_pos % len(hatches)])
                    # Draw zero reference for difference plots
                    if plot_difference:
                        ax.axvline(
                            x=0,
                            color="black",
                            linestyle="--",
                            linewidth=LINEWIDTH,
                            label=None,
                            alpha=ALPHA,
                        )

        if not has_data:
            ax.text(
                0.5,
                0.5,
                f"No {ic_type} data",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )

        ax.set_xlabel(f"Mean Deviation ({info['unit']})")
        ax.set_ylabel("Count (weighted by occurrences)")
        ax.set_title(info["title"])
        ax.legend(fontsize=8)

    fig.suptitle("Internal Coordinate Deviation Distributions by Type", fontsize=14)
    fig.tight_layout()
    fig.savefig(
        plots_dir / f"deviation_summary_all_types{suffix}.{EXTENSION}",
        bbox_inches="tight",
    )
    plt.close(fig)
    logger.info(
        f"Saved: {plots_dir / f'deviation_summary_all_types{suffix}.{EXTENSION}'}"
    )

    logger.info(f"Parameter type plots saved to: {plots_dir}")


def print_summary(
    metrics_list: list[dict],
    labels: list[str] | None = None,
    param_summaries: list[dict] | None = None,
    offxml_paths: list[str] | None = None,
) -> None:
    """Print summary of benchmark results to console.

    Parameters
    ----------
    metrics_list : list[dict]
        List of loaded metrics data dictionaries.
    labels : list[str]
        Labels for each force field.
    param_summaries : list[dict], optional
        List of parameter type summary statistics.
    offxml_paths : list[str], optional
        Paths to force field files for looking up SMIRKS patterns.
    """

    if labels is None:
        # Extract force field names from the metrics
        if len(metrics_list) > 0:
            ff_metrics = metrics_list[0].get("metrics", metrics_list[0])
            labels = list(ff_metrics.keys())
        else:
            labels = ["Force Field"]

    if len(metrics_list) != len(labels):
        raise ValueError(
            f"Number of metrics ({len(metrics_list)}) must match number of labels ({len(labels)})"
        )
    if param_summaries is not None and len(metrics_list) != len(param_summaries):
        raise ValueError(
            f"Number of metrics ({len(metrics_list)}) must match number of parameter summaries ({len(param_summaries)})."
        )
    if offxml_paths is not None and len(metrics_list) != len(offxml_paths):
        raise ValueError(
            f"Number of metrics ({len(metrics_list)}) must match number of offxml paths ({len(offxml_paths)})."
        )

    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)

    # Print metrics for each force field
    for metrics, label in zip(metrics_list, labels):
        ff_metrics = metrics.get("metrics", metrics)

        print(f"\n{label}")
        print("-" * 50)

        ddes = []
        rmsds = []
        tfds = []

        for _, records in ff_metrics.items():
            for _, metric in records.items():
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

    # Print parameter type analysis if available
    if param_summaries:
        print("\n" + "=" * 70)
        print("TOP PARAMETER TYPES BY MEAN DEVIATION")
        print("=" * 70)

        # Build SMIRKS lookup from force fields if available
        smirks_lookups = []
        if offxml_paths:
            for offxml_path in offxml_paths:
                lookup = {}
                if offxml_path:
                    try:
                        ff = ForceField(offxml_path)
                        for handler_name in [
                            "Bonds",
                            "Angles",
                            "ProperTorsions",
                            "ImproperTorsions",
                        ]:
                            handler = ff.get_parameter_handler(handler_name)
                            for param in handler.parameters:
                                lookup[param.id] = param.smirks
                    except Exception as e:
                        logger.debug(
                            f"Could not load force field for SMIRKS lookup: {e}"
                        )
                smirks_lookups.append(lookup)
        else:
            smirks_lookups = [{}] * len(param_summaries)

        for ic_type in ["Bond", "Angle", "Dihedral", "Improper"]:
            unit = "Å" if ic_type == "Bond" else "°"
            print(f"\n{ic_type}s ({unit}):")
            print("-" * 70)

            for ff_idx, (param_summary, label) in enumerate(
                zip(param_summaries, labels)
            ):
                if ic_type not in param_summary or not param_summary[ic_type]:
                    continue

                print(f"\n  {label}:")
                smirks_lookup = (
                    smirks_lookups[ff_idx] if ff_idx < len(smirks_lookups) else {}
                )

                params_sorted = sorted(
                    param_summary[ic_type].items(),
                    key=lambda x: x[1].get("mean", 0),
                    reverse=True,
                )

                for i, (param_id, data) in enumerate(params_sorted[:5], 1):
                    count = data.get("count", 0)
                    mean_err = data.get("mean", float("nan"))
                    smirks = smirks_lookup.get(param_id, "")
                    print(f"    {i}. {param_id}: mean={mean_err:.4f} {unit}, n={count}")
                    if smirks:
                        print(f"       SMIRKS: {smirks}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate benchmark plots from existing results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Plot benchmarking results where parameter summaries are only produced if a single dataset
is passed.

Examples:
    Basic plots from single metrics.json:
        python plot_benchmark.py --metrics benchmark_results/metrics.json

    Compare multiple force fields:
        python plot_benchmark.py \\
            --metrics ff1/metrics.json ff2/metrics.json \\
            --labels "OpenFF 2.2.1" "Custom FF" \\
            --output-dir comparison/

    With parameter type analysis:
        python plot_benchmark.py \\
            --metrics ff1/metrics.json ff2/metrics.json \\
            --labels "OpenFF 2.2.1" "Custom FF" \\
            --param-json ff1/metrics_by_parameter_type.json ff2/metrics_by_parameter_type.json \\
            --offxml ff1.offxml ff2.offxml

Output Files:
    - plots/dde_histogram.pdf           : DDE distribution histogram
    - plots/rmsd_cdf.pdf                : RMSD cumulative distribution
    - plots/tfd_cdf.pdf                 : TFD cumulative distribution
    - plots/icrmsd_comparison.pdf       : Bar chart of ICRMSD by type
    - plots/icrmsd_by_type.pdf          : CDF plots for each IC type
    - plots/deviation_by_*_type.pdf     : Mean deviation by SMIRKS (requires param analysis)
    - plots/deviation_summary_all_types.pdf : Summary of all IC type deviations
        """,
    )
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        required=True,
        help="Path(s) to metrics.json file(s) from benchmarking run(s)",
    )
    parser.add_argument(
        "--labels",
        type=str,
        nargs="+",
        default=None,
        help="Labels for each force field (defaults to parent directory names)",
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
        nargs="+",
        default=None,
        help="Path(s) to benchmark.sqlite for parameter type analysis",
    )
    parser.add_argument(
        "--offxml",
        type=str,
        nargs="+",
        default=None,
        help="Path(s) to OFFXML force field file(s) for SMIRKS lookup",
    )
    parser.add_argument(
        "--param-json",
        type=str,
        nargs="+",
        default=None,
        help="Path(s) to existing metrics_by_parameter_type.json file(s)",
    )
    parser.add_argument(
        "--n-processes",
        type=int,
        default=1,
        help="Number of parallel processes for parameter type analysis (default: 1)",
    )
    parser.add_argument(
        "--plot-difference",
        action="store_true",
        default=False,
        help=(
            "If set and multiple metrics files are provided, subtract the values "
            "from the first-listed force field from all subsequent force fields "
            "before plotting. Useful to visualize differences relative to a "
            "reference. Default: False"
        ),
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

    output_dir = pathlib.Path(args.output_dir)
    n_ff = len(args.metrics)

    # Generate labels if not provided
    if args.labels:
        labels = args.labels
        if len(labels) != n_ff:
            logger.error(
                f"Number of labels ({len(labels)}) must match number of metrics files ({n_ff})"
            )
            sys.exit(1)
    else:
        # Use parent directory names as labels
        labels = []
        for metrics_path in args.metrics:
            path = pathlib.Path(metrics_path)
            labels.append(path.parent.name if path.parent.name != "." else path.stem)

    # Load all metrics files
    metrics_list = []
    for metrics_path in args.metrics:
        path = pathlib.Path(metrics_path)
        if not path.exists():
            logger.error(f"Metrics file not found: {path}")
            sys.exit(1)
        metrics_list.append(load_metrics(path))

    logger.info(f"Loaded {n_ff} metrics file(s): {labels}")

    # Generate basic plots (may plot differences if requested)
    generate_basic_plots(
        metrics_list, labels, output_dir, plot_difference=args.plot_difference
    )

    # Handle parameter type analysis
    param_summaries: list[dict] = []

    if args.param_json:
        # Load from explicitly provided JSON paths
        for param_path in args.param_json:
            path = pathlib.Path(param_path)
            if path.exists():
                param_summaries.append(load_parameter_type_summary(path))
            else:
                logger.warning(f"Parameter type JSON not found: {path}")
                param_summaries.append({})
    else:
        # Try auto-detecting metrics_by_parameter_type.json next to each metrics file
        for metrics_path in args.metrics:
            metrics_dir = pathlib.Path(metrics_path).parent
            auto_param_path = metrics_dir / "metrics_by_parameter_type.json"
            if auto_param_path.exists():
                logger.info(
                    f"Found existing parameter type analysis: {auto_param_path}"
                )
                param_summaries.append(load_parameter_type_summary(auto_param_path))
            else:
                param_summaries.append({})

    # Filter out empty param summaries for plotting
    has_param_data = any(bool(ps) for ps in param_summaries)

    # Generate parameter type plots if available
    if has_param_data:
        if len(labels) == 1:
            generate_parameter_type_plots(param_summaries, labels, output_dir)

    # Get offxml paths for SMIRKS lookup
    offxml_paths = args.offxml if args.offxml else []

    # Print summary
    print_summary(
        metrics_list,
        labels,
        param_summaries if has_param_data else None,
        offxml_paths if offxml_paths else None,
    )

    logger.info("Plot generation complete!")


if __name__ == "__main__":
    freeze_support()
    main()
