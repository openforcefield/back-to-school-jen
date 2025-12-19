"""
Compare SMEE force field checkpoints to analyze training progress.

This script loads checkpoint files from a training run and reports:
- The change in loss between checkpoints
- The parameters with the largest absolute changes
- Plots of loss vs epoch and parameter evolution over training

Usage
-----
    python compare_checkpoints.py --checkpoint-dir my-smee-fit/

    python compare_checkpoints.py \
        --checkpoint-first my-smee-fit/force-field-epoch-10.pt \
        --checkpoint-last my-smee-fit/force-field-epoch-990.pt

    python compare_checkpoints.py --checkpoint-dir my-smee-fit/ --plot
"""

import argparse
import pathlib
import re
from collections import defaultdict

import smee
import torch
import numpy as np
import matplotlib.pyplot as plt
from loguru import logger


def get_epoch_from_path(path: pathlib.Path) -> int:
    """Extract epoch number from checkpoint filename."""
    match = re.search(r"epoch-(\d+)", path.name)
    return int(match.group(1)) if match else 0


def find_all_checkpoints(checkpoint_dir: pathlib.Path) -> list[pathlib.Path]:
    """Find all checkpoint files in a directory, sorted by epoch.

    Parameters
    ----------
    checkpoint_dir : pathlib.Path
        Directory containing .pt checkpoint files.

    Returns
    -------
    list[pathlib.Path]
        List of checkpoint file paths sorted by epoch number.

    Raises
    ------
    FileNotFoundError
        If no checkpoint files are found in the directory.
    """
    checkpoint_files = list(checkpoint_dir.glob("force-field-epoch-*.pt"))

    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoint files found in {checkpoint_dir}")

    checkpoint_files.sort(key=get_epoch_from_path)
    checkpoint_files += list(checkpoint_dir.glob("final-force-field.pt"))

    logger.info(f"Found {len(checkpoint_files)} checkpoint files")
    logger.info(f"First checkpoint: {checkpoint_files[0].name}")
    logger.info(f"Last checkpoint: {checkpoint_files[-1].name}")

    return checkpoint_files


def find_checkpoints(checkpoint_dir: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path]:
    """Find the first and most recent checkpoint files in a directory.

    Parameters
    ----------
    checkpoint_dir : pathlib.Path
        Directory containing .pt checkpoint files.

    Returns
    -------
    tuple[pathlib.Path, pathlib.Path]
        Paths to the first and most recent checkpoint files.
    """
    all_checkpoints = find_all_checkpoints(checkpoint_dir)
    return all_checkpoints[0], all_checkpoints[-1]


def load_checkpoint(checkpoint_path: pathlib.Path) -> smee.TensorForceField:
    """Load a SMEE force field checkpoint.

    Parameters
    ----------
    checkpoint_path : pathlib.Path
        Path to the .pt checkpoint file.

    Returns
    -------
    smee.TensorForceField
        SMEE TensorForceField object.
    """
    logger.info(f"Loading checkpoint: {checkpoint_path}")
    return torch.load(checkpoint_path, map_location="cpu", weights_only=False)


def compare_force_fields(
    ff_first: smee.TensorForceField,
    ff_last: smee.TensorForceField,
    trained_handlers: set[str] = {"Bonds", "Angles"},
) -> dict[str, dict]:
    """Compare two SMEE force field checkpoints and find parameter changes.

    Parameters
    ----------
    ff_first : smee.TensorForceField
        SMEE TensorForceField from the first checkpoint.
    ff_last : smee.TensorForceField
        SMEE TensorForceField from the most recent checkpoint.

    Returns
    -------
    dict
        Dictionary containing comparison results by handler type.
    """

    # trained_handlers is now a function argument
    results: dict[str, dict] = defaultdict(lambda: defaultdict(list))

    for pot_first, pot_last in zip(ff_first.potentials, ff_last.potentials):
        handler_name = pot_first.parameter_keys[0].associated_handler
        if handler_name not in trained_handlers:
            continue
        parameter_cols = pot_first.parameter_cols
        parameter_units = pot_first.parameter_units

        logger.info(f"Comparing {handler_name} parameters...")

        for i, (params_first, params_last) in enumerate(
            zip(pot_first.parameters, pot_last.parameters)
        ):
            smirks = pot_first.parameter_keys[i].id

            params_first_np = params_first.detach().cpu().numpy()
            params_last_np = params_last.detach().cpu().numpy()

            for j, (col, unit) in enumerate(zip(parameter_cols, parameter_units)):
                val_first = float(params_first_np[j])
                val_last = float(params_last_np[j])
                abs_change = abs(val_last - val_first)
                rel_change = (
                    abs_change / abs(val_first) if val_first != 0 else float("inf")
                )

                results[handler_name][col].append(
                    {
                        "smirks": smirks,
                        "parameter": col,
                        "unit": str(unit),
                        "value_first": val_first,
                        "value_last": val_last,
                        "abs_change": abs_change,
                        "rel_change": rel_change,
                    }
                )

    # Sort by absolute change within each handler
    for handler_name in results:
        for param_type in results[handler_name].keys():
            results[handler_name][param_type].sort(
                key=lambda x: x["abs_change"], reverse=True
            )

    return dict(results)


def extract_parameter_value(
    ff: smee.TensorForceField, handler_name: str, smirks: str, param_col: str
) -> float:
    """Extract a specific parameter value from a force field checkpoint.

    Parameters
    ----------
    ff : smee.TensorForceField
        SMEE TensorForceField object.
    handler_name : str
        Name of the handler (e.g., "Bonds", "Angles").
    smirks : str
        SMIRKS pattern identifying the parameter.
    param_col : str
        Name of the parameter column (e.g., "k", "length").

    Returns
    -------
    float
        The parameter value.
    """
    for potential in ff.potentials:
        if potential.parameter_keys[0].associated_handler != handler_name:
            continue

        param_cols = potential.parameter_cols
        if param_col not in param_cols:
            continue

        col_idx = param_cols.index(param_col)

        for i, params in enumerate(potential.parameters):
            if potential.parameter_keys[i].id == smirks:
                return float(params.detach().cpu().numpy()[col_idx])

    raise ValueError(f"Parameter not found: {handler_name}/{smirks}/{param_col}")


def track_parameters_over_epochs(
    checkpoint_files: list[pathlib.Path],
    top_params: list[dict],
) -> dict:
    """Track specific parameters across all checkpoints.

    Parameters
    ----------
    checkpoint_files : list[pathlib.Path]
        List of checkpoint files sorted by epoch.
    top_params : list[dict]
        List of parameter info dicts with 'handler', 'smirks', 'parameter' keys.

    Returns
    -------
    dict
        Dictionary with 'epochs' and parameter tracking data.
    """
    epochs: list[int] = []
    param_values: dict[str, list[float]] = {
        f"{p['handler']}/{p['smirks']}/{p['parameter']}": [] for p in top_params
    }

    for i, checkpoint_path in enumerate(checkpoint_files):
        epoch = get_epoch_from_path(checkpoint_path)
        if i > 0 and epoch == 0:
            epoch = epochs[-1] + (epochs[-1] - epochs[-2])
        epochs.append(epoch)

        ff = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        for param_info in top_params:
            key = f"{param_info['handler']}/{param_info['smirks']}/{param_info['parameter']}"
            try:
                value = extract_parameter_value(
                    ff,
                    param_info["handler"],
                    param_info["smirks"],
                    param_info["parameter"],
                )
                param_values[key].append(value)
            except ValueError:
                param_values[key].append(float("nan"))

    return {"epochs": epochs, "parameters": param_values}


def parse_tensorboard_loss(checkpoint_dir: pathlib.Path) -> dict | None:
    """Parse loss values from TensorBoard event files.

    Parameters
    ----------
    checkpoint_dir : pathlib.Path
        Directory containing TensorBoard event files.

    Returns
    -------
    dict | None
        Dictionary with 'epochs' and 'loss' lists, or None if not found.
    """
    try:
        from tensorboard.backend.event_processing.event_accumulator import (
            EventAccumulator,
        )
    except ImportError:
        logger.warning("tensorboard package not installed, cannot parse loss history")
        return None

    event_files = list(checkpoint_dir.glob("events.out.tfevents.*"))
    if not event_files:
        logger.warning(f"No TensorBoard event files found in {checkpoint_dir}")
        return None

    # Use the most recent event file
    event_file = max(event_files, key=lambda p: p.stat().st_mtime)
    logger.info(f"Reading TensorBoard events from: {event_file.name}")

    ea = EventAccumulator(str(event_file))
    ea.Reload()

    # Look for loss scalar
    scalar_tags = ea.Tags().get("scalars", [])
    loss_tag = None
    for tag in scalar_tags:
        if "loss" in tag.lower():
            loss_tag = tag
            break

    if not loss_tag:
        logger.warning(
            f"No loss scalar found in TensorBoard. Available tags: {scalar_tags}"
        )
        return None

    events = ea.Scalars(loss_tag)
    epochs = [e.step for e in events]
    losses = [e.value for e in events]

    return {"epochs": epochs, "loss": losses, "tag": loss_tag}


def generate_training_plots(
    checkpoint_dir: pathlib.Path,
    results: dict,
    output_dir: pathlib.Path,
    top_n: int = 5,
) -> None:
    """Generate plots showing training progress.

    Parameters
    ----------
    checkpoint_dir : pathlib.Path
        Directory containing checkpoint files and TensorBoard events.
    results : dict
        Comparison results from compare_force_fields().
    output_dir : pathlib.Path
        Directory to save plot files.
    top_n : int, optional
        Number of top changed parameters to plot per handler (default: 5).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect top changed parameters across all handlers
    top_params = []
    for handler_name, param_dict in results.items():
        for param_type, params in param_dict.items():
            for param_info in params[:top_n]:
                top_params.append(
                    {
                        "handler": handler_name,
                        "smirks": param_info["smirks"],
                        "parameter": param_type,
                        "unit": param_info["unit"],
                    }
                )

    # Get all checkpoint files
    checkpoint_files = find_all_checkpoints(checkpoint_dir)

    # Track parameters over epochs
    logger.info(
        f"Tracking {len(top_params)} parameters across {len(checkpoint_files)} checkpoints..."
    )
    param_history = track_parameters_over_epochs(checkpoint_files, top_params)

    # Parse loss history from TensorBoard
    loss_history = parse_tensorboard_loss(checkpoint_dir)

    # Create figure with subplots
    n_rows = 1 + sum(
        len(x) for x in results.values()
    )  # Loss + one row per handler/parameter combination
    fig, axes = plt.subplots(n_rows, 1, figsize=(12, 4 * n_rows), sharex=True)

    if n_rows == 1:
        axes = [axes]

    # Plot 1: Loss vs Epoch
    ax = axes[0]
    if loss_history:
        ax.plot(loss_history["epochs"], loss_history["loss"], "b-", linewidth=1.5)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title(f"Training Loss ({loss_history['tag']})")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
    else:
        ax.text(
            0.5,
            0.5,
            "Loss history not available\n(TensorBoard events not found)",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=12,
        )
        ax.set_title("Training Loss")

    # Plot parameters by handler
    colors = plt.cm.get_cmap("viridis")(np.linspace(0, 1, top_n))

    for idx, (handler_name, param_dict) in enumerate(results.items()):
        for jdx, (param_type, params) in enumerate(param_dict.items()):
            ax = axes[1 + idx * len(results) + jdx]

            for i, param_info in enumerate(params[:top_n]):
                key = f"{handler_name}/{param_info['smirks']}/{param_info['parameter']}"
                values = param_history["parameters"].get(key, [])

                if not values:
                    continue
                if len(values) != len(param_history["epochs"]):
                    logger.warning(
                        f"Skipping {key}: values length {len(values)} != epochs length {len(param_history['epochs'])}"
                    )
                    continue

                # Truncate SMIRKS for legend
                smirks_short = param_info["smirks"]
                if len(smirks_short) > 30:
                    smirks_short = smirks_short[:27] + "..."

                label = f"{smirks_short}"
                ax.plot(
                    param_history["epochs"],
                    values,
                    color=colors[i],
                    linewidth=1.5,
                    label=label,
                )
            ax.set_ylabel(f"{handler_name} {param_type}")
            ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.set_xlabel("Epoch")

    plt.tight_layout()

    # Save figure
    output_path = output_dir / "training_progress.pdf"
    fig.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    logger.info(f"Saved training progress plot to: {output_path}")

    # Also create individual plots for each handler
    for handler_name, param_dict in results.items():
        for param_type, params in param_dict.items():
            fig, ax = plt.subplots(figsize=(10, 6))

            for i, param_info in enumerate(params[:top_n]):
                key = f"{handler_name}/{param_info['smirks']}/{param_info['parameter']}"
                values = param_history["parameters"].get(key, [])

                if values:
                    label = f"{param_info["smirks"]}"
                    ax.plot(
                        param_history["epochs"],
                        values,
                        color=colors[i % len(colors)],
                        linewidth=1.5,
                        label=label,
                    )

            ax.set_xlabel("Epoch", fontsize=12)
            ax.set_ylabel("{handler_name} {param_type}", fontsize=12)
            ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)
            ax.grid(True, alpha=0.3)

            output_path = (
                output_dir
                / f"parameter_evolution_{handler_name.lower()}_{param_type}.pdf"
            )
            fig.savefig(output_path, bbox_inches="tight", dpi=150)
            plt.close(fig)
            logger.info(
                f"Saved {handler_name} {param_type} parameter plot to: {output_path}"
            )


def print_comparison_report(
    results: dict,
    first_path: pathlib.Path,
    last_path: pathlib.Path,
    top_n: int = 10,
) -> None:
    """Print a formatted comparison report.

    Parameters
    ----------
    results : dict
        Comparison results from compare_force_fields().
    first_path : pathlib.Path
        Path to the first checkpoint file.
    last_path : pathlib.Path
        Path to the last checkpoint file.
    top_n : int, optional
        Number of top changed parameters to show per handler (default: 10).
    """
    print("\n" + "=" * 80)
    print("CHECKPOINT COMPARISON REPORT")
    print("=" * 80)
    print(f"\nFirst checkpoint:  {first_path.name}")
    print(f"Last checkpoint:   {last_path.name}")

    for handler_name, param_dict in results.items():
        for param_type, params in param_dict.items():
            print(f"\n{'-' * 80}")
            print(
                f"{handler_name} - {param_type} - Top {min(top_n, len(params))} Parameters by Absolute Change"
            )
            print(f"{'-' * 80}")

            if not params:
                print("  No parameters found")
                continue

            # Header
            print(
                f"  {'First':>12} {'Last':>12} "
                f"{'Δ (abs)':>12} {'Δ (%)':>10} {'SMIRKS':}"
            )
            print("  " + "-" * 103)

            for param_info in params[:top_n]:
                smirks = param_info["smirks"]

                rel_pct = param_info["rel_change"] * 100
                rel_str = f"{rel_pct:.2f}%" if rel_pct < 1e6 else "inf"

                print(
                    f"  {param_info['value_first']:>12.6f} {param_info['value_last']:>12.6f} "
                    f"{param_info['abs_change']:>12.6f} {rel_str:>10} {smirks}"
                )

    # Summary statistics
    print(f"\n{'=' * 80}")
    print("SUMMARY STATISTICS")
    print("=" * 80)

    for handler_name, param_dict in results.items():
        for param_type, params in param_dict.items():
            if not params:
                continue

            total_params = len(params)
            abs_changes = [p["abs_change"] for p in params]
            mean_change = sum(abs_changes) / len(abs_changes) if abs_changes else 0
            max_change = max(abs_changes) if abs_changes else 0
            changed_params = sum(1 for c in abs_changes if c > 1e-10)

            print(f"\n{handler_name} - {param_type}:")
            print(f"  Total parameters:     {total_params}")
            print(
                f"  Parameters changed:   {changed_params} ({100*changed_params/total_params:.1f}%)"
            )
            print(f"  Mean absolute change: {mean_change:.6f}")
            print(f"  Max absolute change:  {max_change:.6f}")

    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Compare SMEE checkpoints to analyze training progress",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    Auto-detect first and last checkpoints in a directory:
        python compare_checkpoints.py --checkpoint-dir my-smee-fit/

    Generate plots of training progress:
        python compare_checkpoints.py --checkpoint-dir my-smee-fit/ --plot

    Explicitly specify checkpoint files:
        python compare_checkpoints.py \\
            --checkpoint-first my-smee-fit/force-field-epoch-10.pt \\
            --checkpoint-last my-smee-fit/force-field-epoch-990.pt

    Show more parameters per handler:
        python compare_checkpoints.py --checkpoint-dir my-smee-fit/ --top-n 20
        """,
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Directory containing .pt checkpoint files (auto-detects first/last)",
    )
    parser.add_argument(
        "--checkpoint-first",
        type=str,
        default=None,
        help="Path to the first/initial .pt checkpoint file",
    )
    parser.add_argument(
        "--checkpoint-last",
        type=str,
        default=None,
        help="Path to the last/most recent .pt checkpoint file",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Number of top changed parameters to show per handler (default: 10)",
    )
    parser.add_argument(
        "--handlers",
        type=str,
        nargs="+",
        default=["Bonds", "Angles"],
        help="Parameter types to compare (default: Bonds Angles)",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate plots of loss and parameter evolution over training",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save plots (default: same as checkpoint-dir or current directory)",
    )

    args = parser.parse_args()

    # Determine checkpoint paths
    if args.checkpoint_dir:
        checkpoint_dir = pathlib.Path(args.checkpoint_dir)
        first_path, last_path = find_checkpoints(checkpoint_dir)
    elif args.checkpoint_first and args.checkpoint_last:
        first_path = pathlib.Path(args.checkpoint_first)
        last_path = pathlib.Path(args.checkpoint_last)
        checkpoint_dir = first_path.parent
    else:
        parser.error(
            "Either --checkpoint-dir or both --checkpoint-first and --checkpoint-last are required"
        )

    # Load checkpoints or final OFFXML
    ff_first = load_checkpoint(first_path)
    ff_last = load_checkpoint(last_path)

    # Compare and report
    results = compare_force_fields(
        ff_first, ff_last, trained_handlers=set(args.handlers)
    )
    print_comparison_report(results, first_path, last_path, top_n=args.top_n)

    # Generate plots if requested
    if args.plot:
        output_dir = (
            pathlib.Path(args.output_dir) if args.output_dir else checkpoint_dir
        )
        generate_training_plots(
            checkpoint_dir,
            results,
            output_dir,
            top_n=args.top_n,
        )

    logger.info("Done!")


if __name__ == "__main__":
    main()
