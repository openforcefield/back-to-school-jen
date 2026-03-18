"""Cluster force field parameters from an OFFXML file.

This module performs unsupervised clustering analysis on valence parameters
(Bonds and Angles) extracted from OpenFF force field files. It identifies
groups of similar parameters by clustering in (log(k), equilibrium) space
and generates visualizations and summary statistics.

Command-line Arguments
----------------------
--offxml : str
    Path to an OFFXML file containing the force field to analyze.
--output-dir : str, optional
    Directory where plots and CSV summaries will be written (default: "./clusters").
--method : str, optional
    Clustering method to use: "hdbscan" or "dbscan" (default: "hdbscan").
--kwargs-bonds : str, optional
    Clustering algorithm keyword arguments for bonds as a Python dict string
    (e.g. '{"min_cluster_size": 3}').
--kwargs-angles : str, optional
    Clustering algorithm keyword arguments for angles as a Python dict string
    (e.g. '{"min_cluster_size": 10, "min_samples": 5}').
--show : flag
    If provided, matplotlib windows will be shown interactively.

Examples
--------
Cluster with default HDBSCAN parameters:
$ python clustering.py --offxml openff-2.2.1.offxml --output-dir ./clusters

Use custom clustering parameters:
$ python clustering.py --offxml my-ff.offxml \\
    --kwargs-bonds '{"min_cluster_size": 5}' \\
    --kwargs-angles '{"min_cluster_size": 20, "min_samples": 10}'

Output Structure
----------------
Creates the following outputs in specified output directory:
- bonds_clusters.csv          # Bond parameter cluster assignments
- angles_clusters.csv          # Angle parameter cluster assignments
- bonds_clusters.pdf           # Scatter plot with cluster labels
- angles_clusters.pdf          # Scatter plot with cluster labels

CSV Schema
----------
bonds_clusters.csv columns:
    smirks : str
        SMIRKS pattern defining the bond parameter scope
    k_kJ_mol_per_A2 : float
        Force constant in kJ/(mol·Å²)
    length_A : float
        Equilibrium bond length in Å
    cluster : int
        Cluster label (-1 indicates noise)

angles_clusters.csv columns:
    smirks : str
        SMIRKS pattern defining the angle parameter scope
    k_kJ_mol_per_rad2 : float
        Force constant in kJ/(mol·rad²)
    angle_deg : float
        Equilibrium angle in degrees
    cluster : int
        Cluster label (-1 indicates noise)

Notes
-----
Parameter Units and Conversions:
- Bond force constants: converted to kJ/(mol·Å²)
- Bond lengths: converted to Å
- Angle force constants: converted to kJ/(mol·rad²)
- Angles: converted to degrees for readability

Clustering Approach:
- Features: [log(k + ε), equilibrium_value] where ε=1e-8
- Scaling: Each feature divided by its mean (not StandardScaler)
- Default HDBSCAN parameters:
  * Bonds: min_cluster_size=3
  * Angles: min_cluster_size=10, min_samples=5 (more conservative)

Physical Interpretation:
- Clusters identify parameter families with similar stiffness and equilibrium values
- Noise points (-1) represent unique or outlier parameters
- Concentric patterns in angle plots reflect different periodicities
"""

from __future__ import annotations

import argparse
import ast
import csv
import pathlib
from typing import cast

import hdbscan
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from matplotlib.figure import Figure
from openff.toolkit import ForceField
from openff.units import unit as _UNIT
from sklearn.cluster import DBSCAN
import re
import periodictable as pt


def load_offxml(offxml_path: pathlib.Path | str) -> ForceField:
    """Load an OFFXML ForceField from disk.

    Parameters
    ----------
    offxml_path : pathlib.Path | str
        Path to the OFFXML file.

    Returns
    -------
    ForceField
        An OpenFF ForceField instance parsed from the file.

    Raises
    ------
    FileNotFoundError
        If the OFFXML file does not exist at the specified path.

    Examples
    --------
    >>> ff = load_offxml("openff-2.2.1.offxml")
    >>> ff.get_parameter_handler("Bonds")
    """
    offxml_path = pathlib.Path(offxml_path)
    if not offxml_path.exists():
        raise FileNotFoundError(f"OFFXML file not found: {offxml_path}")

    logger.info(f"Loading OFFXML force field from: {offxml_path}")
    ff = ForceField(offxml_path)
    return ff


def _get_handler_parameters(handler) -> list:
    """Return parameter list for a handler using public or private API.

    Parameters
    ----------
    handler
        OpenFF parameter handler object.

    Returns
    -------
    list
        List of parameter objects from the handler.

    """
    if hasattr(handler, "parameters"):
        return handler.parameters
    elif hasattr(handler, "_parameters"):
        return handler._parameters
    return []


def extract_valence_params(
    ff: ForceField, handler_name: str, col_map: dict[str, str]
) -> list[dict]:
    """Extract parameter values for a given valence handler.

    Converts OpenFF Quantity objects to convenient plotting units:
    - Bond k: kJ/(mol·Å²)
    - Bond length: Å
    - Angle k: kJ/(mol·rad²)
    - Angle: degrees

    Parameters
    ----------
    ff : ForceField
        The OpenFF ForceField instance.
    handler_name : str
        Name of the parameter handler (e.g., "Bonds", "Angles").
    col_map : dict[str, str]
        Mapping from output column names (e.g., "k", "length") to
        OpenFF parameter attribute names.

    Returns
    -------
    list[dict]
        Each dict contains 'smirks' and mapped column names with
        float values in convenient units.

    Notes
    -----
    - Returns empty list if handler not present in force field
    - NaN values used for missing or unconvertible parameters
    - Multiple parameters with same SMIRKS are included separately

    Examples
    --------
    >>> bonds = extract_valence_params(
    ...     ff, "Bonds", {"k": "k", "length": "length"}
    ... )
    >>> bonds[0]
    {'smirks': '[#6X4:1]-[#6X4:2]', 'k': 2928.4, 'length': 1.526}
    """
    try:
        handler = ff.get_parameter_handler(handler_name)
    except Exception:
        logger.warning(f"Handler {handler_name} not present in force field")
        return []

    params = _get_handler_parameters(handler)
    out: list[dict] = []

    for param in params:
        entry = {"smirks": getattr(param, "smirks", None)}
        for dst_col, src_attr in col_map.items():
            if hasattr(param, src_attr):
                value = getattr(param, src_attr)
                # Convert OpenFF Quantity to convenient unit
                if hasattr(value, "m_as"):
                    if dst_col == "k" and handler_name == "Bonds":
                        target = "kilojoule / mol / angstrom ** 2"
                        val = value.m_as(target)
                    elif dst_col == "k" and handler_name == "Angles":
                        target = _UNIT.kilojoule / _UNIT.mole / (_UNIT.radian**2)
                        val = value.m_as(target)
                    elif dst_col == "length":
                        val = value.m_as(_UNIT.angstrom)
                    elif dst_col == "angle":
                        # Store angles in degrees for readability
                        val = value.m_as(_UNIT.degree)
                    else:
                        # Fall back to magnitude
                        try:
                            val = float(value.magnitude)
                        except Exception:
                            val = float(value)
                elif hasattr(value, "__len__") and len(value) == 1:
                    val = float(value[0])
                else:
                    try:
                        val = float(value)
                    except Exception:
                        logger.debug(
                            f"Unable to convert parameter value {value} for {src_attr}; using NaN"
                        )
                        val = float("nan")

                entry[dst_col] = float(val)
            else:
                entry[dst_col] = float("nan")
        out.append(entry)

    return out


def cluster_parameters(
    X: np.ndarray,
    method: str = "hdbscan",
    kwargs_cluster: dict = {},
) -> np.ndarray:
    """Cluster feature matrix X and return cluster labels.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features).
    method : str
        Clustering method; 'hdbscan' or 'dbscan'.
    kwargs_cluster : dict
        Kwargs for clustering method. Defaults are:
        - DBSCAN: {"eps": 0.5, "min_samples": 2}
        - HDBSCAN: {"min_cluster_size": 3}

    Returns
    -------
    np.ndarray
        Integer cluster labels for each sample. -1 indicates noise.

    Notes
    -----
    Scales each feature by dividing by its mean value, which is more appropriate
    than StandardScaler for features with different physical meanings (e.g.,
    log(k) vs angles). This preserves relative differences while normalizing scales.

    Examples
    --------
    >>> labels = cluster_parameters(X, method="hdbscan", kwargs_cluster={"min_cluster_size": 5})
    """
    # Scale by mean instead of StandardScaler for better handling of mixed feature types
    X_means = np.mean(np.abs(X), axis=0)
    # Avoid division by zero
    X_means[X_means < 1e-10] = 1.0
    X_scaled = X / X_means

    logger.info(f"Feature scaling: X_means = {X_means}")

    if method.lower() == "hdbscan":
        # Build explicit kwargs to avoid passing an arbitrary dict to HDBSCAN
        min_cluster_size = kwargs_cluster.get("min_cluster_size", 3)
        min_samples = kwargs_cluster.get("min_samples", None)
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size, min_samples=min_samples
        )
    elif method.lower() == "dbscan":
        eps = kwargs_cluster.get("eps", 0.5)
        min_samples = kwargs_cluster.get("min_samples", 2)
        clusterer = DBSCAN(eps=eps, min_samples=min_samples)
    else:
        supported_types = ["hdbscan", "dbscan"]
        raise ValueError(
            f"Clustering method '{method}' is not supported. Choose from: {supported_types}"
        )

    labels = clusterer.fit_predict(X_scaled)

    return labels


def plot_and_cluster(
    entries: list[dict],
    handler_name: str,
    output_dir: pathlib.Path,
    method: str = "hdbscan",
    kwargs_cluster: dict = {},
    show: bool = False,
) -> list[dict]:
    """Plot parameter scatter and cluster them.

    Parameters
    ----------
    entries : list[dict]
        Parameter entries returned by extract_valence_params.
    handler_name : str
        Either 'Bonds' or 'Angles'.
    output_dir : pathlib.Path
        Directory to save plots and CSV summaries.
    method : str
        Clustering method.
    kwargs_cluster : dict
        Keyword arguments for clustering method, see :func:`cluster_parameters` for more details.
    show : bool
        If True, show the matplotlib figure interactively.

    Returns
    -------
    list[dict]
        Annotated parameter entries with cluster labels.

    Notes
    -----
    Side effects:
    - Creates CSV file with parameter values and cluster assignments
    - Creates PDF plot visualizing clusters in parameter space
    - Logs summary information about clusters found
    """
    if not entries:
        logger.warning(f"No entries found for {handler_name}; skipping")
        return []

    # Build numpy arrays for k and equilibrium
    ks = np.array([e.get("k", float("nan")) for e in entries], dtype=float)
    eq = None
    if handler_name == "Bonds":
        eq = np.array([e.get("length", float("nan")) for e in entries], dtype=float)
        xlabel = "Length (Å)"
    else:
        eq = np.array([e.get("angle", float("nan")) for e in entries], dtype=float)
        xlabel = "Angle (°)"

    # Prepare feature matrix for clustering; we use log(k+eps) and eq
    eps = 1e-8
    logk = np.log(np.abs(ks) + eps)
    X = np.vstack([logk, eq]).T

    labels = cluster_parameters(X, method=method, kwargs_cluster=kwargs_cluster)

    # Log clustering results
    unique_labels = set(labels)
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
    n_noise = list(labels).count(-1)
    logger.info(
        f"{handler_name}: Found {n_clusters} clusters and {n_noise} noise points"
    )

    # Save handler-specific CSV summary
    out_csv = output_dir / f"{handler_name.lower()}_clusters.csv"
    with open(out_csv, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        if handler_name == "Bonds":
            header = ["smirks", "k_kJ_mol_per_A2", "length_A", "cluster"]
        else:
            header = ["smirks", "k_kJ_mol_per_rad2", "angle_deg", "cluster"]
        writer.writerow(header)
        annotated: list[dict] = []
        for e, lab in zip(entries, labels):
            smi = e.get("smirks", "")
            k_val = e.get("k", float("nan"))
            eq_val = e.get("length", e.get("angle", float("nan")))
            writer.writerow([smi, k_val, eq_val, int(lab)])
            annotated.append(
                {
                    "handler": handler_name,
                    "smirks": smi,
                    "k": k_val,
                    "equilibrium": eq_val,
                    "cluster": int(lab),
                }
            )

    logger.info(f"Saved cluster assignments to: {out_csv}")

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    sorted_labels = list(sorted(set(labels)))
    colors = matplotlib.colormaps.get_cmap("nipy_spectral")(
        np.linspace(0, 1, max(1, len(sorted_labels)))
    )

    for lab in sorted_labels:
        if lab == -1:
            color = "0.6"
            label = "noise"
            zorder = 1
        else:
            color = colors[lab % len(colors)]
            label = f"cluster {lab}"
            zorder = 2
        ax.scatter(eq[lab], ks[lab], label=label, color=color, alpha=0.8, zorder=zorder)

    ax.set_xlabel(xlabel)
    ax.set_ylabel("k (converted)")
    ax.set_yscale("log")
    ax.set_title(f"{handler_name} parameters clustered ({method})")
    ax.grid(True, alpha=0.3)

    # Robust legend placement: place legend outside and choose number of
    # columns so that the legend fills the vertical space as much as possible.
    n_labels = len(sorted_labels)
    fig = cast(Figure, ax.figure)
    fig_w, fig_h = fig.get_size_inches()
    # Estimate text row height in inches (points -> inches: 1pt = 1/72 in)
    base_fontsize = 10.0
    line_spacing = 1.2
    row_height_in = (base_fontsize * line_spacing) / 72.0
    available_height_in = fig_h * 0.9
    nrow = int(np.trunc(available_height_in / row_height_in))
    ncol = int(np.ceil(n_labels / nrow))
    # Adjust fontsize down for many columns to keep legend compact
    fontsize = max(6, int(base_fontsize - (ncol - 1)))
    ax.legend(
        loc="center left", bbox_to_anchor=(1.02, 0.5), ncol=ncol, fontsize=fontsize
    )
    # Shrink plot area to make room for the legend
    right_margin = 0.75 if ncol <= 2 else max(0.55, 1.0 - 0.2 * ncol)

    # If legend takes up more than half the width, increase figure width by 50%
    legend_width_fraction = 1.0 - right_margin
    if legend_width_fraction > 0.5:
        new_fig_w = fig_w * 1.5
        fig.set_size_inches(new_fig_w, fig_h)
        logger.info(
            f'Legend takes >50% width, increasing figure width from {fig_w:.1f}" to {new_fig_w:.1f}"'
        )
    plt.subplots_adjust(right=right_margin)

    out_plot = output_dir / f"{handler_name.lower()}_clusters.pdf"
    fig.tight_layout()
    fig.savefig(out_plot, dpi=150, bbox_inches="tight")
    logger.info(f"Saved cluster plot to: {out_plot}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return annotated


def _element_from_smirks_token(token: str) -> str | None:
    """Extract an element symbol from the contents of a SMIRKS atom bracket.

    Examples of token inputs: '#6X4:1', 'C', 'NH2:1', '*:1'
    """
    # Check for atomic number pattern like #6
    m = re.search(r"#(\d+)", token)
    if m:
        try:
            num = int(m.group(1))
            # Use periodictable package to get element symbol
            el = pt.elements[num]
            return getattr(el, "symbol", f"Z{num}")
        except Exception:
            return None

    # Check for explicit element symbol at start
    m = re.match(r"^([A-Z][a-z]?)", token)
    if m:
        return m.group(1)

    # Fallback: wildcard or unspecified
    if token.startswith("*"):
        return "*"

    return None


def _parse_smirks_elements(smirks: str) -> list[str]:
    """Return list of element symbols found in SMIRKS bracket atoms (order preserved).

    If parsing fails, returns an empty list.
    """
    tokens = re.findall(r"\[([^\]]+)\]", smirks or "")
    elems: list[str] = []
    for t in tokens:
        el = _element_from_smirks_token(t)
        if el:
            elems.append(el)
    return elems


def plot_k_vs_eq_by_element_pairs(
    entries: list[dict], handler_name: str, output_dir: pathlib.Path, show: bool = False
) -> None:
    """Plot k vs equilibrium distance for each element-element pair (Bonds only).

    Creates one PDF per element-element combination found in `entries` and
    saves them to `output_dir` with filenames of the form
    `bonds_k_vs_length_<EL1>_<EL2>.pdf` where EL1 and EL2 are element symbols.
    """
    if handler_name != "Bonds":
        logger.debug(
            "Element-element k vs equilibrium plotting currently implemented for Bonds only."
        )
        return

    if not entries:
        logger.debug("No bond entries to plot by element pairs.")
        return

    groups: dict[tuple[str, str], list[tuple[float, float, str]]] = {}
    for e in entries:
        smi = e.get("smirks", "") or ""
        ks = e.get("k", float("nan"))
        length = e.get("length", float("nan"))
        elems = _parse_smirks_elements(smi)
        if len(elems) >= 2:
            el1, el2 = elems[0], elems[1]
        else:
            # Try splitting on - or : as fallback
            parts = re.split(r"[-:.]", smi)
            el1 = elems[0] if elems else (parts[0] if parts else "X")
            el2 = elems[1] if len(elems) > 1 else (parts[1] if len(parts) > 1 else "X")

        # Canonicalize pair as alphabetical to group e.g. C-H and H-C together
        e1s, e2s = str(el1), str(el2)
        if e1s <= e2s:
            pair: tuple[str, str] = (e1s, e2s)
        else:
            pair = (e2s, e1s)
        groups.setdefault(pair, []).append((length, ks, smi))

    # Bond order color mapping
    bond_order_colors = {
        1: "#1f77b4",
        2: "#ff7f0e",
        3: "#2ca02c",
    }  # single, double, triple
    bond_order_labels = {1: "Single Bond", 2: "Double Bond", 3: "Triple Bond"}
    default_color = "#7f7f7f"

    def get_bond_order_from_smirks(smirks):
        # Extract bond order from the symbol between the two atom brackets
        # Example: [#6X4:1]-[#6X4:2] (single), [#6X3:1]=[#8X1:2] (double), [#7:1]#[#7:2] (triple)
        m = re.match(r"\[[^\]]+\](.|~)\[[^\]]+\]", smirks)
        if m:
            bond_symbol = m.group(1)
            if bond_symbol == "-":
                return 1
            elif bond_symbol == "=":
                return 2
            elif bond_symbol == "#":
                return 3
            else:
                return 1  # fallback to single if unknown
        return 1  # fallback to single if not matched

    for pair, values in sorted(groups.items()):
        pair_label = f"{pair[0]}-{pair[1]}"
        safe_pair = re.sub(r"[^A-Za-z0-9_]+", "_", pair_label)
        fig, ax = plt.subplots(figsize=(6, 4))
        legend_handles = []
        used_orders = set()
        for length, k_val, smi in values:
            bond_order = get_bond_order_from_smirks(smi)
            color = bond_order_colors.get(bond_order, default_color)
            label = bond_order_labels.get(bond_order, "Other")
            if bond_order not in used_orders:
                legend_handles.append(
                    plt.Line2D([0], [0], color=color, lw=2, label=label)
                )
                used_orders.add(bond_order)
            ax.scatter(length, k_val, color=color, alpha=0.8, zorder=2)
        ax.set_xlabel("Length (Å)")
        ax.set_ylabel("k (kJ / mol / Å²)")
        ax.set_yscale("log")
        ax.set_title(f"Bonds: k vs length for {pair_label} ({len(values)} params)")
        ax.grid(True, alpha=0.3)
        if legend_handles:
            ax.legend(
                handles=legend_handles,
                loc="center left",
                bbox_to_anchor=(1.02, 0.5),
                title="Bond Order",
                fontsize=10,
            )
        out_plot = output_dir / f"bonds_k_vs_length_{safe_pair}.pdf"
        fig.tight_layout()
        fig.savefig(out_plot, dpi=150, bbox_inches="tight")
        logger.info(f"Saved element-pair plot to: {out_plot}")
        if show:
            plt.show()
        else:
            plt.close(fig)


def main(
    offxml: pathlib.Path | str,
    output_dir: pathlib.Path | str = pathlib.Path("./clusters"),
    method: str = "hdbscan",
    kwargs_bonds: dict = {},
    kwargs_angles: dict = {},
    show: bool = False,
) -> None:
    """Main workflow for clustering force field parameters.

    Loads an OFFXML force field, extracts bond and angle parameters,
    performs clustering analysis, and generates visualizations and
    summary statistics.

    Parameters
    ----------
    offxml : pathlib.Path | str
        Path to an OFFXML file containing the force field to analyze.
    output_dir : pathlib.Path | str, optional
        Directory where plots and CSV summaries will be written (default: "./clusters").
    method : str, optional
        Clustering method to use: "hdbscan" or "dbscan" (default: "hdbscan").
    kwargs_bonds : dict, optional
        Clustering algorithm keyword arguments for bonds (default: {}).
    kwargs_angles : dict, optional
        Clustering algorithm keyword arguments for angles (default: {}).
    show : bool, optional
        If True, matplotlib windows will be shown interactively (default: False).

    Returns
    -------
    None

    Notes
    -----
    Workflow:
    1. Loads OFFXML force field from disk
    2. Extracts Bond and Angle parameters with unit conversions
    3. Performs clustering on each parameter type separately
    4. Generates scatter plots with cluster labels
    5. Saves CSV files with cluster assignments

    Output files created in output_dir:
    - bonds_clusters.csv: Bond parameter cluster assignments
    - angles_clusters.csv: Angle parameter cluster assignments
    - bonds_clusters.pdf: Bond parameter scatter plot
    - angles_clusters.pdf: Angle parameter scatter plot

    Examples
    --------
    >>> main("openff-2.2.1.offxml", output_dir="./my_clusters")
    >>> main(
    ...     "openff-2.2.1.offxml",
    ...     method="hdbscan",
    ...     kwargs_bonds={"min_cluster_size": 5},
    ...     kwargs_angles={"min_cluster_size": 15, "min_samples": 10}
    ... )
    """
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir.resolve()}")

    ff = load_offxml(offxml)

    # Extract bond and angle parameters
    logger.info("Extracting bond parameters...")
    bonds = extract_valence_params(
        ff,
        "Bonds",
        {"k": "k", "length": "length"},
    )
    logger.info(f"Found {len(bonds)} bond parameters")

    # Additionally: plot k vs equilibrium distance grouped by element-element pairs
    try:
        plot_k_vs_eq_by_element_pairs(bonds, "Bonds", output_dir, show=show)
    except Exception:
        logger.exception("Failed to generate element-pair plots for bonds")

    logger.info("Extracting angle parameters...")
    angles = extract_valence_params(
        ff,
        "Angles",
        {"k": "k", "angle": "angle"},
    )
    logger.info(f"Found {len(angles)} angle parameters")

    # Set defaults for HDBSCAN if not specified
    # Angles tend to create concentric patterns, so use larger min_cluster_size by default
    if method.lower() == "hdbscan":
        if "min_cluster_size" not in kwargs_bonds:
            kwargs_bonds["min_cluster_size"] = 3
        if "min_cluster_size" not in kwargs_angles:
            kwargs_angles["min_cluster_size"] = 3
        if "min_samples" not in kwargs_angles:
            kwargs_angles["min_samples"] = 5  # More conservative clustering

    logger.info(f"Clustering bonds with kwargs: {kwargs_bonds}")
    logger.info(f"Clustering angles with kwargs: {kwargs_angles}")

    _ = plot_and_cluster(
        bonds,
        "Bonds",
        output_dir,
        method=method,
        kwargs_cluster=kwargs_bonds,
        show=show,
    )

    _ = plot_and_cluster(
        angles,
        "Angles",
        output_dir,
        method=method,
        kwargs_cluster=kwargs_angles,
        show=show,
    )

    logger.info("Clustering analysis complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Cluster OFFXML valence parameters (Bonds, Angles)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python clustering.py --offxml openff-2.2.1.offxml
    python clustering.py --offxml my-ff.offxml --output-dir ./my_clusters
    python clustering.py --offxml my-ff.offxml --kwargs-bonds '{"min_cluster_size": 5}'
        """,
    )
    parser.add_argument(
        "--offxml",
        type=str,
        required=True,
        help="Path to OFFXML file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./clusters",
        help="Directory for plots and CSVs (default: ./clusters)",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="hdbscan",
        help="Clustering method: hdbscan or dbscan (default: hdbscan)",
    )
    parser.add_argument(
        "--kwargs-bonds",
        type=str,
        default="{}",
        help="Clustering kwargs for bonds as Python dict string (e.g. '{\"min_cluster_size\": 3}')",
    )
    parser.add_argument(
        "--kwargs-angles",
        type=str,
        default="{}",
        help="Clustering kwargs for angles as Python dict string (e.g. '{\"min_cluster_size\": 10}')",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show matplotlib windows",
    )

    args = parser.parse_args()

    # Safely parse kwargs strings to dicts
    try:
        kwargs_bonds = ast.literal_eval(args.kwargs_bonds)
        if not isinstance(kwargs_bonds, dict):
            raise ValueError
    except Exception:
        logger.warning("Failed to parse --kwargs-bonds argument; using empty dict.")
        kwargs_bonds = {}

    try:
        kwargs_angles = ast.literal_eval(args.kwargs_angles)
        if not isinstance(kwargs_angles, dict):
            raise ValueError
    except Exception:
        logger.warning("Failed to parse --kwargs-angles argument; using empty dict.")
        kwargs_angles = {}

    main(
        offxml=args.offxml,
        output_dir=args.output_dir,
        method=args.method,
        kwargs_bonds=kwargs_bonds,
        kwargs_angles=kwargs_angles,
        show=args.show,
    )
