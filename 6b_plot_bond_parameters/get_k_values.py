"""Plot bond force constants from a SMIRNOFF offxml file."""

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
from openff.toolkit.typing.engines.smirnoff import ForceField
from openff.units import unit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a k vs length scatter plot from a SMIRNOFF force field."
    )
    parser.add_argument(
        "--force-field",
        type=Path,
        required=True,
        help="Path to the force field offxml file that defines the bond parameters.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path to save the plot (PNG). Shows interactively when omitted.",
    )
    return parser.parse_args()


def load_bond_parameters(force_field_path: Path) -> list[dict[str, float]]:
    force_field = ForceField(str(force_field_path))
    handler = force_field.get_parameter_handler("Bonds")
    parameters = []
    for parameter in handler.parameters:
        if parameter.k is None or parameter.length is None:
            continue
        parameters.append(
            {
                "smirks": parameter.smirks,
                "length": parameter.length.to(unit.angstrom).magnitude,
                "k": parameter.k.to(
                    unit.kilocalorie_per_mole / unit.angstrom**2
                ).magnitude,
            }
        )
    return parameters


def plot_bond_constants(
    parameters: list[dict[str, float]], output_path: Path | None
) -> None:
    lengths = [entry["length"] for entry in parameters]
    ks = [entry["k"] for entry in parameters]
    figure, axis = plt.subplots()
    axis.scatter(lengths, ks, alpha=0.7)
    axis.set_yscale("log")
    axis.set_xlabel("Bond Length ($\AA$)")
    axis.set_ylabel("Force Constant (kcal / (mol $\AA^2$)")
    axis.set_xlim(0.5, 4.0)
    axis.set_ylim(10, 3000)
    axis.set_yscale("log")
    axis.plot([0.9, 0.9], [10, 3000], color="k")
    axis.plot([2.5, 2.5], [10, 3000], color="k")
    axis.grid(True, which="both", linestyle=":", linewidth=0.8)
    if output_path:
        figure.savefig(output_path, bbox_inches="tight", dpi=200)
    else:
        plt.show()


def main() -> None:
    args = parse_args()
    if not args.force_field.exists():
        raise FileNotFoundError(f"Force field not found at {args.force_field}")
    bond_parameters = load_bond_parameters(args.force_field)

    os.makedirs(os.path.split(args.output)[0], exist_ok=True)
    if not bond_parameters:
        raise SystemExit("No bonds were found in the force field.")
    plot_bond_constants(bond_parameters, args.output)


if __name__ == "__main__":
    main()
