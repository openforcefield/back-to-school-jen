"""
Convert a saved SMEE force field checkpoint (.pt) to an OFFXML file.

This script loads a force field checkpoint saved during training and writes
the optimized parameters to an OFFXML file.

Usage
-----
    python convert_pt_to_offxml.py \
        --checkpoint my-smee-fit/force-field-epoch-990.pt \
        --offxml-in /path/to/openff_unconstrained-2.2.1.offxml \
        --offxml-out final-force-field.offxml
"""

import argparse
import pathlib
from collections import defaultdict

import torch
from loguru import logger
from openff.toolkit import ForceField


def write_new_offxml(
    smee_force_field, offxml_in: pathlib.Path | str, offxml_out: pathlib.Path | str
) -> None:
    """Convert optimized SMEE force field parameters to OFFXML format.

    Parameters
    ----------
    smee_force_field : smee.TensorForceField
        Optimized SMEE force field tensor object containing fitted parameters.
    offxml_in : pathlib.Path | str
        Path to the reference OFFXML file used as the template.
    offxml_out : pathlib.Path | str
        Path for the output OFFXML file with updated parameters.
    """
    offxml_in = pathlib.Path(offxml_in)
    offxml_out = pathlib.Path(offxml_out)

    logger.info(f"Loading template force field from: {offxml_in}")
    starting_ff = ForceField(str(offxml_in))

    for potential in smee_force_field.potentials:
        handler_name = potential.parameter_keys[0].associated_handler

        parameter_attrs = potential.parameter_cols
        parameter_units = potential.parameter_units

        if handler_name in ["Bonds", "Angles"]:
            handler = starting_ff.get_parameter_handler(handler_name)
            for i, opt_parameters in enumerate(potential.parameters):
                smirks = potential.parameter_keys[i].id
                ff_parameter = handler[smirks]
                opt_parameters = opt_parameters.detach().cpu().numpy()
                for j, (p, unit) in enumerate(zip(parameter_attrs, parameter_units)):
                    setattr(ff_parameter, p, opt_parameters[j] * unit)

        elif handler_name in ["ProperTorsions"]:
            handler = starting_ff.get_parameter_handler(handler_name)
            k_index = parameter_attrs.index("k")
            p_index = parameter_attrs.index("periodicity")
            # collect the k values into a list across the entries
            collection_data: dict[str, dict[int, float]] = defaultdict(dict)
            for i, opt_parameters in enumerate(potential.parameters):
                smirks = potential.parameter_keys[i].id
                ff_parameter = handler[smirks]
                opt_parameters = opt_parameters.detach().cpu().numpy()
                k = opt_parameters[k_index] * parameter_units[k_index]
                p = int(opt_parameters[p_index])
                collection_data[smirks][p] = k
            # update the force field
            for smirks, k_s in collection_data.items():
                ff_parameter = handler[smirks]
                k_mapped_to_p = [k_s[p] for p in ff_parameter.periodicity]
                ff_parameter.k = k_mapped_to_p

        elif handler_name in ["ImproperTorsions"]:
            k_index = parameter_attrs.index("k")
            handler = starting_ff.get_parameter_handler(handler_name)
            for i, opt_parameters in enumerate(potential.parameters):
                smirks = potential.parameter_keys[i].id
                ff_parameter = handler[smirks]
                opt_parameters = opt_parameters.detach().cpu().numpy()
                ff_parameter.k = [opt_parameters[k_index] * parameter_units[k_index]]

    logger.info(f"Saving new forcefield to: {offxml_out}")
    starting_ff.to_file(str(offxml_out))


def main():
    parser = argparse.ArgumentParser(
        description="Convert SMEE checkpoint (.pt) to OFFXML format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python convert_pt_to_offxml.py \\
        --checkpoint my-smee-fit/force-field-epoch-990.pt \\
        --offxml-in ../../forcefields/openff_unconstrained-2.2.1.offxml \\
        --offxml-out final-force-field.offxml
        """,
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the .pt checkpoint file from training",
    )
    parser.add_argument(
        "--offxml-in",
        type=str,
        required=True,
        help="Path to the template OFFXML file (must match the force field used for training)",
    )
    parser.add_argument(
        "--offxml-out",
        type=str,
        default="final-force-field.offxml",
        help="Output path for the updated OFFXML file (default: final-force-field.offxml)",
    )

    args = parser.parse_args()

    logger.info(f"Loading checkpoint from: {args.checkpoint}")
    smee_force_field = torch.load(
        args.checkpoint, map_location="cpu", weights_only=False
    )

    write_new_offxml(smee_force_field, args.offxml_in, args.offxml_out)
    logger.info("Done!")


if __name__ == "__main__":
    main()
