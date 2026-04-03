"""Compare element-element valence parameters across multiple OFFXML files.

Plots k vs equilibrium value (bond length or angle) for each element-element
pair found across all supplied force fields. Each force field is rendered in
a distinct colour. Bond order is encoded by marker shape (○ single, □ double,
△ triple). When all force fields share an identical set of SMIRKS for a
handler, an optional connecting line can be drawn between corresponding points
to visualise the parameter drift across force fields.

Command-line Arguments
----------------------
--offxml : str (one or more)
    Paths to OFFXML files to compare.
--labels : str (optional, same count as --offxml)
    Display names for each force field.  Defaults to the filename stem.
--handler : str, optional
    Parameter handler to plot: "Bonds", "Angles", or "both" (default: "Bonds").
--output-dir : str, optional
    Directory for output plots (default: "./compare_ff").
--alpha : float or list of float, optional
    Scatter marker transparency: one value applied to all FFs, or one per FF
    in the same order as --offxml (default: 0.7).
--line-alpha : float, optional
    Connecting line transparency when --connect-lines is used (default: 0.3).
--marker-size : float or list of float, optional
    Scatter marker size in points²: one value applied to all FFs, or one per
    FF in the same order as --offxml (default: 20).
--connect-lines : flag
    Draw lines connecting the same SMIRKS across force fields. Active for
    every SMIRKS present in all supplied force fields.
--show : flag
    Show matplotlib windows interactively.

Output Structure
----------------
For each element-element pair and handler, writes one PNG to output-dir:
    bonds_compare_<EL1>_<EL2>.png
    angles_compare_<EL1>_<EL2>.png

Examples
--------
$ python compare_ff_params.py \\
    --offxml ff_v1.offxml ff_v2.offxml \\
    --labels "v1" "v2" \\
    --connect-lines --alpha 0.8

$ python compare_ff_params.py \\
    --offxml a.offxml b.offxml c.offxml \\
    --handler both --output-dir ./out --alpha 0.6 --line-alpha 0.2
"""

from __future__ import annotations

import argparse
import pathlib
import re

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import periodictable as pt
from loguru import logger
from openff.toolkit import ForceField
from openff.units import unit as _UNIT


# ── force field loading ───────────────────────────────────────────────────────


def _load_offxml(path: str | pathlib.Path) -> ForceField:
    p = pathlib.Path(path)
    if not p.exists():
        raise FileNotFoundError(f"OFFXML not found: {p}")
    logger.info(f"Loading {p}")
    return ForceField(p)


def _get_handler_parameters(handler) -> list:
    if hasattr(handler, "parameters"):
        return handler.parameters
    if hasattr(handler, "_parameters"):
        return handler._parameters
    return []


# ── parameter extraction ──────────────────────────────────────────────────────


def _extract_bonds(ff: ForceField) -> dict[str, tuple[float, float]]:
    """Return {smirks: (k_kJ_mol_per_A2, length_A)} for all bond parameters."""
    try:
        handler = ff.get_parameter_handler("Bonds")
    except Exception:
        return {}
    out: dict[str, tuple[float, float]] = {}
    for param in _get_handler_parameters(handler):
        smi = getattr(param, "smirks", None)
        if smi is None:
            continue
        k_val = float("nan")
        l_val = float("nan")
        if hasattr(param, "k") and hasattr(param.k, "m_as"):
            k_val = param.k.m_as("kilojoule / mol / angstrom ** 2")
        if hasattr(param, "length") and hasattr(param.length, "m_as"):
            l_val = param.length.m_as(_UNIT.angstrom)
        out[smi] = (float(k_val), float(l_val))
    return out


def _extract_angles(ff: ForceField) -> dict[str, tuple[float, float]]:
    """Return {smirks: (k_kJ_mol_per_rad2, angle_deg)} for all angle parameters."""
    try:
        handler = ff.get_parameter_handler("Angles")
    except Exception:
        return {}
    out: dict[str, tuple[float, float]] = {}
    for param in _get_handler_parameters(handler):
        smi = getattr(param, "smirks", None)
        if smi is None:
            continue
        k_val = float("nan")
        a_val = float("nan")
        if hasattr(param, "k") and hasattr(param.k, "m_as"):
            k_val = param.k.m_as(_UNIT.kilojoule / _UNIT.mole / (_UNIT.radian**2))
        if hasattr(param, "angle") and hasattr(param.angle, "m_as"):
            a_val = param.angle.m_as(_UNIT.degree)
        out[smi] = (float(k_val), float(a_val))
    return out


# ── SMIRKS utilities ──────────────────────────────────────────────────────────


def _element_from_token(token: str) -> str | None:
    """Extract element symbol from the *contents* of a SMIRKS bracket atom."""
    m = re.search(r"#(\d+)", token)
    if m:
        try:
            el = pt.elements[int(m.group(1))]
            return getattr(el, "symbol", f"Z{m.group(1)}")
        except Exception:
            return None
    m = re.match(r"^([A-Z][a-z]?)", token)
    if m:
        return m.group(1)
    if token.startswith("*"):
        return "*"
    return None


def _parse_smirks_elements(smirks: str) -> list[str]:
    """Return ordered list of element symbols found in top-level bracket atoms."""
    tokens = re.findall(r"\[([^\]]+)\]", smirks or "")
    return [e for t in tokens if (e := _element_from_token(t))]


def _element_pair(smirks: str) -> tuple[str, str]:
    """Return canonicalized (el1, el2) pair from a SMIRKS string."""
    elems = _parse_smirks_elements(smirks)
    el1 = elems[0] if len(elems) > 0 else "X"
    el2 = elems[1] if len(elems) > 1 else "X"
    return (el1, el2) if el1 <= el2 else (el2, el1)


def _bond_order_from_smirks(smirks: str) -> int:
    """Return bond order (1/2/3) from the character after the first top-level bracket.

    Handles recursive SMIRKS where brackets are nested (e.g. $([...])).
    """
    if not smirks or smirks[0] != "[":
        return 1
    depth, end = 0, -1
    for i, c in enumerate(smirks):
        if c == "[":
            depth += 1
        elif c == "]":
            depth -= 1
            if depth == 0:
                end = i
                break
    if end == -1 or end + 1 >= len(smirks):
        return 1
    return {"-": 1, "=": 2, "#": 3}.get(smirks[end + 1], 1)


# ── plotting ──────────────────────────────────────────────────────────────────

# Marker shape encodes bond order; colour encodes FF identity
_BOND_MARKERS = {1: "o", 2: "s", 3: "^"}
_BOND_ORDER_LABELS = {1: "single", 2: "double", 3: "triple"}

# Up to 10 distinct FF colours (tab10-derived)
_FF_COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]


def _to_per_ff(val: float | list[float], n: int) -> list[float]:
    """Broadcast a scalar or list to a per-FF list of length *n*."""
    if isinstance(val, (list, tuple)):
        if len(val) == 1:
            return [float(val[0])] * n
        if len(val) != n:
            raise ValueError(f"Expected 1 or {n} values, got {len(val)}")
        return [float(v) for v in val]
    return [float(val)] * n


def _plot_handler_compare(
    handler_name: str,
    ff_data: list[dict[str, tuple[float, float]]],
    labels: list[str],
    output_dir: pathlib.Path,
    alpha: float | list[float] = 0.7,
    line_alpha: float = 0.3,
    marker_size: float | list[float] = 20,
    connect_lines: bool = False,
    show: bool = False,
) -> None:
    """Produce one plot per element pair for a single handler type.

    Parameters
    ----------
    handler_name : str
        "Bonds" or "Angles".
    ff_data : list[dict]
        One dict per FF mapping smirks -> (k, eq_value).
    labels : list[str]
        Display name for each FF.
    output_dir : pathlib.Path
        Directory to write plots.
    alpha : float or list[float]
        Scatter marker opacity: one value for all FFs or one per FF.
    line_alpha : float
        Connecting line opacity.
    marker_size : float or list[float]
        Scatter marker size in points²: one value for all FFs or one per FF.
    connect_lines : bool
        Draw lines linking same-SMIRKS points across consecutive FFs.
    show : bool
        Show interactive windows.
    """
    is_bonds = handler_name == "Bonds"
    xlabel = "Length (Å)" if is_bonds else "Angle (°)"
    ylabel = "k (kJ / mol / Å²)" if is_bonds else "k (kJ / mol / rad²)"
    file_prefix = "bonds_compare" if is_bonds else "angles_compare"

    # Collect all element pairs across every FF
    all_smirks: set[str] = set()
    for fd in ff_data:
        all_smirks.update(fd.keys())

    pairs: dict[tuple[str, str], set[str]] = {}
    for smi in all_smirks:
        pairs.setdefault(_element_pair(smi), set()).add(smi)

    # Determine SMIRKS shared by all FFs (for connecting lines)
    shared_smirks: set[str] = set()
    if len(ff_data) > 1:
        shared_smirks = set.intersection(*(set(fd.keys()) for fd in ff_data))
        smirks_sets = [set(fd.keys()) for fd in ff_data]
        all_identical = all(s == smirks_sets[0] for s in smirks_sets[1:])
        if connect_lines and not all_identical:
            unique_to = sum(
                1 for s in smirks_sets[0] if not all(s in sd for sd in smirks_sets[1:])
            )
            logger.warning(
                f"SMIRKS sets differ across FFs ({unique_to} SMIRKS not shared). "
                "Lines drawn only for SMIRKS present in all FFs."
            )

    colors = [_FF_COLORS[i % len(_FF_COLORS)] for i in range(len(ff_data))]
    n_ff = len(ff_data)
    alphas = _to_per_ff(alpha, n_ff)
    marker_sizes = _to_per_ff(marker_size, n_ff)

    for pair, pair_smirks in sorted(pairs.items()):
        pair_label = f"{pair[0]}-{pair[1]}"
        safe_pair = re.sub(r"[^A-Za-z0-9_]+", "_", pair_label)

        fig, ax = plt.subplots(figsize=(6, 4))

        # Build legend entries — scale legend marker proportionally to per-FF scatter size
        legend_mss = [max(4, (marker_sizes[i] ** 0.5) * 0.9) for i in range(n_ff)]
        ff_legend_handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="None",
                markerfacecolor=colors[i],
                markeredgecolor="none",
                markersize=legend_mss[i],
                label=labels[i],
            )
            for i in range(n_ff)
        ]
        used_bond_orders: set[int] = set()
        bond_order_handles: list = []

        for ff_idx, (fd, color) in enumerate(zip(ff_data, colors)):
            for smi in sorted(pair_smirks):
                if smi not in fd:
                    continue
                k_val, eq_val = fd[smi]
                if is_bonds:
                    bond_order = _bond_order_from_smirks(smi)
                    marker = _BOND_MARKERS.get(bond_order, "o")
                    if bond_order not in used_bond_orders:
                        bond_order_handles.append(
                            Line2D(
                                [0],
                                [0],
                                marker=marker,
                                linestyle="None",
                                markerfacecolor="k",
                                markeredgecolor="none",
                                markersize=legend_mss[ff_idx],
                                label=_BOND_ORDER_LABELS[bond_order],
                            )
                        )
                        used_bond_orders.add(bond_order)
                else:
                    marker = "o"
                ax.scatter(
                    eq_val,
                    k_val,
                    color=color,
                    marker=marker,
                    s=marker_sizes[ff_idx],
                    alpha=alphas[ff_idx],
                    edgecolors="none",
                    zorder=2,
                )

        # Connecting lines for SMIRKS present in all FFs
        if connect_lines and shared_smirks:
            for smi in sorted(pair_smirks & shared_smirks):
                xs = [fd[smi][1] for fd in ff_data]
                ys = [fd[smi][0] for fd in ff_data]
                ax.plot(
                    xs,
                    ys,
                    color="0.4",
                    lw=0.8,
                    alpha=line_alpha,
                    zorder=1,
                    solid_capstyle="round",
                )

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_yscale("log")
        if is_bonds:
            ax.set_ylim(ymin=90)
        ax.set_title(
            f"{handler_name}: k vs {'length' if is_bonds else 'angle'} — {pair_label}"
        )
        ax.grid(True, alpha=0.3)

        all_handles = ff_legend_handles + bond_order_handles
        ax.legend(
            all_handles,
            [str(h.get_label()) for h in all_handles],
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            fontsize=9,
        )

        out_plot = output_dir / f"{file_prefix}_{safe_pair}.png"
        fig.tight_layout()
        fig.savefig(out_plot, dpi=150, bbox_inches="tight")
        logger.info(f"Saved {out_plot}")
        if show:
            plt.show()
        else:
            plt.close(fig)


# ── entry point ───────────────────────────────────────────────────────────────


def main(
    offxmls: list[str | pathlib.Path],
    labels: list[str] | None = None,
    handler: str = "Bonds",
    output_dir: str | pathlib.Path = "./compare_ff",
    alpha: float | list[float] = 0.7,
    line_alpha: float = 0.3,
    marker_size: float | list[float] = 20,
    connect_lines: bool = False,
    show: bool = False,
) -> None:
    """Compare valence parameters across multiple OFFXML force fields.

    Parameters
    ----------
    offxmls : list
        Paths to OFFXML files.
    labels : list[str], optional
        Display names; defaults to filename stems.
    handler : str
        "Bonds", "Angles", or "both".
    output_dir : str or Path
        Directory for output plots.
    alpha : float or list[float]
        Scatter marker opacity: one value for all FFs or one per FF.
    line_alpha : float
        Connecting line opacity.
    marker_size : float or list[float]
        Scatter marker size in points²: one value for all FFs or one per FF.
    connect_lines : bool
        Draw lines between same-SMIRKS points across FFs.
    show : bool
        Show interactive plots.
    """
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ffs = [_load_offxml(p) for p in offxmls]

    # Resolve labels
    if labels is None or len(labels) != len(ffs):
        if labels and len(labels) != len(ffs):
            logger.warning(
                f"Expected {len(ffs)} labels, got {len(labels)}; using filename stems."
            )
        labels = [pathlib.Path(p).stem for p in offxmls]

    do_bonds = handler.lower() in ("bonds", "both")
    do_angles = handler.lower() in ("angles", "both")

    if do_bonds:
        logger.info("Extracting bond parameters...")
        bonds_data = [_extract_bonds(ff) for ff in ffs]
        for lbl, bd in zip(labels, bonds_data):
            logger.info(f"  {lbl}: {len(bd)} bond parameters")
        _plot_handler_compare(
            "Bonds",
            bonds_data,
            labels,
            output_dir,
            alpha=alpha,
            line_alpha=line_alpha,
            marker_size=marker_size,
            connect_lines=connect_lines,
            show=show,
        )

    if do_angles:
        logger.info("Extracting angle parameters...")
        angles_data = [_extract_angles(ff) for ff in ffs]
        for lbl, ad in zip(labels, angles_data):
            logger.info(f"  {lbl}: {len(ad)} angle parameters")
        _plot_handler_compare(
            "Angles",
            angles_data,
            labels,
            output_dir,
            alpha=alpha,
            line_alpha=line_alpha,
            marker_size=marker_size,
            connect_lines=connect_lines,
            show=show,
        )

    logger.info("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare element-element FF parameters across multiple OFFXML files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python compare_ff_params.py --offxml ff_v1.offxml ff_v2.offxml --labels v1 v2
  python compare_ff_params.py --offxml a.offxml b.offxml --connect-lines --alpha 0.8
  python compare_ff_params.py --offxml a.offxml b.offxml c.offxml \\
      --handler both --output-dir ./out --line-alpha 0.15
        """,
    )
    parser.add_argument(
        "--offxml",
        nargs="+",
        required=True,
        help="One or more OFFXML files to compare",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        default=None,
        help="Display labels for each FF (same order as --offxml)",
    )
    parser.add_argument(
        "--handler",
        default="Bonds",
        help="Parameter type: Bonds, Angles, or both (default: Bonds)",
    )
    parser.add_argument(
        "--output-dir",
        default="./compare_ff",
        help="Output directory for plots (default: ./compare_ff)",
    )
    parser.add_argument(
        "--alpha",
        nargs="+",
        type=float,
        default=[0.7],
        help="Scatter marker alpha: one value for all FFs or one per FF (default: 0.7)",
    )
    parser.add_argument(
        "--line-alpha",
        type=float,
        default=0.3,
        help="Connecting line alpha (default: 0.3)",
    )
    parser.add_argument(
        "--marker-size",
        nargs="+",
        type=float,
        default=[20],
        help="Scatter marker size in points²: one value for all FFs or one per FF (default: 20)",
    )
    parser.add_argument(
        "--connect-lines",
        action="store_true",
        help="Draw lines connecting the same SMIRKS across FFs",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show matplotlib windows interactively",
    )

    args = parser.parse_args()

    # Validate per-FF list lengths against the number of --offxml files
    n_ff = len(args.offxml)
    for arg_name, values in [
        ("--alpha", args.alpha),
        ("--marker-size", args.marker_size),
    ]:
        if len(values) not in (1, n_ff):
            parser.error(
                f"{arg_name}: expected 1 value (broadcast to all FFs) or {n_ff} values "
                f"(one per FF), but got {len(values)}."
            )

    # Unwrap single-element lists to scalars so _to_per_ff broadcasts correctly
    alpha_arg = args.alpha[0] if len(args.alpha) == 1 else args.alpha
    marker_size_arg = (
        args.marker_size[0] if len(args.marker_size) == 1 else args.marker_size
    )
    main(
        offxmls=args.offxml,
        labels=args.labels,
        handler=args.handler,
        output_dir=args.output_dir,
        alpha=alpha_arg,
        line_alpha=args.line_alpha,
        marker_size=marker_size_arg,
        connect_lines=args.connect_lines,
        show=args.show,
    )
