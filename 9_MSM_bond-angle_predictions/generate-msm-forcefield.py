"""
This script generates a force field from MSM data.
It reads a set of MSM data files, aggregates the data by parameter type,
and sets initial values for bonds and angles accordingly for an input force field.
"""

import collections
import concurrent.futures
import json
import os
import pathlib
import logging
import sys
import typing

import click
import tqdm

import numpy as np
import pyarrow.dataset as ds

from openff.toolkit import Molecule, ForceField
from openff.units import unit

if typing.TYPE_CHECKING:
    from openff.toolkit.typing.engines.smirnoff.parameters import ParameterType

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

_WORKER_FF: ForceField | None = None


def _init_worker(input_forcefield: str):
    """Load force field once per worker process."""
    global _WORKER_FF
    _WORKER_FF = ForceField(input_forcefield)


def _extract_parameter_values(
    cmiles: str,
    subdf,
    ff: ForceField,
) -> dict[str, dict[str, dict[str, typing.Any]]]:
    """Collect MSM values for one mapped SMILES entry."""
    mol = Molecule.from_mapped_smiles(
        cmiles,
        allow_undefined_stereo=True,
    )
    labels = ff.label_molecules(mol.to_topology())[0]

    out: dict[str, dict[str, dict[str, typing.Any]]] = {}

    for parameter_type, parameter_df in subdf.groupby("parameter_type"):
        parameter_labels = labels[parameter_type]
        parameter_type_dict = out.setdefault(parameter_type, {})
        smirks_by_parameter_id: dict[str, str] = {}

        for _, row in parameter_df.iterrows():
            indices = tuple(row["indices"])
            try:
                parameter = parameter_labels[indices]
            except KeyError:
                continue

            parameter_id = str(parameter.id)
            parameter_dict = parameter_type_dict.setdefault(
                parameter_id,
                {
                    "eq": [],
                    "k": [],
                    "id": [],
                    "cmiles": [],
                    "smirks": None,
                },
            )
            parameter_dict["eq"].append(row["eq"])
            parameter_dict["k"].append(row["force_constant"])
            parameter_dict["id"].append(row["id"])
            parameter_dict["cmiles"].append(cmiles)
            smirks_by_parameter_id[parameter_id] = str(parameter.smirks)

        for parameter_id, smirks in smirks_by_parameter_id.items():
            parameter_type_dict[parameter_id]["smirks"] = smirks

    return out


def _worker_process_cmiles(args):
    """Worker wrapper to process one molecule group."""
    cmiles, subdf = args
    if _WORKER_FF is None:
        raise RuntimeError("Worker force field is not initialized")
    return _extract_parameter_values(cmiles, subdf, _WORKER_FF)


def _merge_parameter_values(
    target: collections.defaultdict,
    source: dict[str, dict[str, dict[str, typing.Any]]],
):
    """Merge per-molecule MSM aggregates into one nested dictionary."""
    for parameter_type, source_type_dict in source.items():
        target_type_dict = target[parameter_type]
        for parameter_id, source_parameter_dict in source_type_dict.items():
            target_parameter_dict = target_type_dict[parameter_id]
            target_parameter_dict["eq"].extend(source_parameter_dict["eq"])
            target_parameter_dict["k"].extend(source_parameter_dict["k"])
            target_parameter_dict["id"].extend(source_parameter_dict["id"])
            target_parameter_dict["cmiles"].extend(source_parameter_dict["cmiles"])
            target_parameter_dict["smirks"] = source_parameter_dict["smirks"]


@click.command()
@click.option(
    "--input-forcefield",
    "-i",
    type=str,
    help="Input forcefield file to modify.",
)
@click.option(
    "--output-forcefield",
    "-o",
    type=str,
    help="Output forcefield file to write.",
)
@click.option(
    "--output-msm",
    "-om",
    type=str,
    help="Output MSM file to write.",
)
@click.option(
    "--msm-data-directory",
    "-im",
    default="msm-data",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    help=(
        "Directory containing the input MSM data. "
        "This is a directory containing parquet files."
    ),
)
@click.option(
    "--aggregator",
    "-a",
    default="mean",
    type=click.Choice(["mean", "median"]),
    help=("Aggregator to use for the MSM data. " "This is either 'mean' or 'median'."),
)
@click.option(
    "--n-workers",
    "-n",
    type=click.IntRange(min=1),
    default=max(1, (os.cpu_count() or 1) - 1),
    show_default=True,
    help=(
        "Number of worker processes to use for per-molecule MSM labeling. "
        "Set to 1 to disable parallel processing."
    ),
)
def main(
    input_forcefield: str,
    output_msm: str,
    output_forcefield: str,
    msm_data_directory: str = "msm-data",
    aggregator: typing.Literal["mean", "median"] = "mean",
    n_workers: int = max(1, (os.cpu_count() or 1) - 1),
):
    msm_data = ds.dataset(msm_data_directory)
    logger.info(f"Loaded {msm_data.count_rows()} MSM records")

    df = msm_data.to_table().to_pandas()
    n_cmiles = len(df.cmiles.unique())
    logger.info(
        f"Found {n_cmiles} unique molecules, {len(df)} total records in MSM data"
    )

    ff = ForceField(input_forcefield)
    logger.info(f"Loaded forcefield from {input_forcefield}")

    output_msm_path = pathlib.Path(output_msm)
    output_msm_path.parent.mkdir(parents=True, exist_ok=True)

    # collect all MSM values by parameter
    all_msm_values: collections.defaultdict[
        str,
        collections.defaultdict[
            str,
            collections.defaultdict[str, typing.Any],
        ],
    ] = collections.defaultdict(
        lambda: collections.defaultdict(lambda: collections.defaultdict(list))
    )

    logger.info(f"Processing molecules with {n_workers} worker process(es)")

    if n_workers == 1:
        for cmiles, subdf in tqdm.tqdm(
            df.groupby("cmiles"),
            total=n_cmiles,
            desc="Processing molecules to FF parameters",
        ):
            result = _extract_parameter_values(cmiles, subdf, ff)
            _merge_parameter_values(all_msm_values, result)
    else:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=n_workers,
            initializer=_init_worker,
            initargs=(input_forcefield,),
        ) as executor:
            for result in tqdm.tqdm(
                executor.map(
                    _worker_process_cmiles, df.groupby("cmiles"), chunksize=20
                ),
                total=n_cmiles,
                desc="Processing molecules to FF parameters",
            ):
                _merge_parameter_values(all_msm_values, result)

    # save output json for debugging
    with output_msm_path.open("w") as f:
        json.dump(
            all_msm_values,
            f,
            indent=4,
        )
    logger.info(f"Wrote MSM partitions to {output_msm_path}")

    # aggregate and update FF
    # known unit conversions
    kj_per_mol_per_nm2 = unit.kilojoule_per_mole / (unit.nanometer**2)
    kcal_per_mol_per_a2 = unit.kilocalorie_per_mole / (unit.angstrom**2)
    kj_per_mol_per_rad2 = unit.kilojoule_per_mole / (unit.radian**2)
    kcal_per_mol_per_rad2 = unit.kilocalorie_per_mole / (unit.radian**2)

    if aggregator == "mean":
        agg_func = np.mean
    elif aggregator == "median":
        agg_func = np.median
    else:
        raise ValueError(f"Aggregator must be 'mean' or 'median', not {aggregator}")
    logger.info(f"Aggregating with {aggregator}")

    for parameter_type, parameter_type_dict in all_msm_values.items():
        handler = ff.get_parameter_handler(parameter_type)
        for parameter_id, parameter_dict in parameter_type_dict.items():
            # aggregate the values
            parameter = typing.cast("ParameterType", handler[parameter_dict["smirks"]])
            k_value = float(agg_func(parameter_dict["k"]))
            eq_value = float(agg_func(parameter_dict["eq"]))

            if parameter_type == "Bonds":
                k_quantity = k_value * kj_per_mol_per_nm2
                eq_quantity = eq_value * unit.nanometer

                parameter.length = eq_quantity.to(unit.angstrom)
                parameter.k = k_quantity.to(kcal_per_mol_per_a2)

            elif parameter_type == "Angles":
                k_quantity = k_value * kj_per_mol_per_rad2
                eq_quantity = eq_value * unit.radian

                parameter.k = k_quantity.to(kcal_per_mol_per_rad2)

                if np.isclose(parameter.angle.m, 180.0, atol=0, rtol=0):
                    # this is linear, ignore
                    continue
                parameter.angle = eq_quantity.to(unit.degree)

    ff.to_file(output_forcefield)
    logger.info(f"Wrote forcefield to {output_forcefield}")


if __name__ == "__main__":
    main()
