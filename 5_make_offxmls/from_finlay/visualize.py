"""
Molecular Visualization and Statistical Analysis Tools.

This module provides utilities for visualizing molecular mechanics components
and analyzing their statistical distributions. It combines matplotlib/seaborn
for statistical plots with RDKit molecular visualization capabilities to
provide comprehensive analysis tools for force field development workflows.

Key Functions
-------------
Statistical Analysis:
    - plot_histogram_of_n_mol_per_type: Histogram of molecule counts per component type
    - plot_cdf_of_n_mol_per_type: Cumulative distribution function visualization

Molecular Visualization:
    - show_molecule_with_atom_indices: 2D molecular structure with highlighted atoms

Examples
--------
Analyzing component type distributions:

>>> import matplotlib.pyplot as plt
>>> from molecular_classes import Bond, Angle
>>> # Sample data structure with component types
>>> mm_components = {
...     0: {"specific": [Bond(...), Bond(...)]},
...     1: {"general": [Angle(...), Angle(...), Angle(...)]}
... }
>>> plot_histogram_of_n_mol_per_type(mm_components)
>>> plt.show()

Visualizing specific molecular components:

>>> # Highlight a bond between atoms 0 and 1
>>> img = show_molecule_with_atom_indices("[C:1][C:2][O:3]", (0, 1))
>>> img.show()  # Display the molecular structure with highlighted atoms

Notes
-----
This module requires RDKit for molecular visualization and matplotlib/seaborn
for statistical plotting. The visualization functions are designed to work
with the molecular mechanics component hierarchy from process_mmcomponents.
"""

from PIL import Image
import io

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem import rdDepictor

from openff.toolkit import Molecule

from .molecular_classes import MMComponent
from .process_mmcomponents import flatten_mm_component_types


def plot_histogram_of_n_mol_per_type(
    mm_component_types: dict[int, dict[str, list[MMComponent]]],
) -> None:
    """
    Plot histogram of molecule counts per molecular mechanics component type.

    Creates a histogram visualization showing the distribution of how many
    molecules contain each component type. Uses log scale and includes a
    kernel density estimate overlay for better visualization of the distribution.

    Parameters
    ----------
    mm_component_types : dict[int, dict[str, list[MMComponent]]]
        Hierarchical dictionary containing molecular mechanics components organized
        by specificity level and component type. Structure:
        {specificity_level: {component_type: [component_instances]}}

    Returns
    -------
    None
        Displays the plot using matplotlib.pyplot.show().

    Notes
    -----
    This function flattens the hierarchical component structure to count total
    occurrences of each component type across all specificity levels. The
    resulting histogram uses a log scale for better visibility of the
    distribution tail.

    Examples
    --------
    >>> from molecular_classes import Bond, Angle
    >>> import matplotlib.pyplot as plt
    >>> # Sample component hierarchy
    >>> components = {
    ...     0: {"[#6:1]-[#6:2]": [Bond(...), Bond(...)]},
    ...     1: {"[*:1]-[*:2]": [Bond(...) for _ in range(10)]}
    ... }
    >>> plot_histogram_of_n_mol_per_type(components)
    >>> plt.show()  # Displays histogram of component type frequencies
    """

    counts = flatten_mm_component_types(mm_component_types)
    fig, ax = plt.subplots(figsize=(10, 6))
    # Plot normalised histogram with KDE
    sns.histplot(counts, bins=30, kde=True, ax=ax, log_scale=True)  # , stat="density")
    ax.set_xlabel("Number of Molecules per Component Type")
    ax.set_ylabel("Frequency")
    ax.set_title("Histogram of Number of Molecules per Component Type")
    # Use log scale for better visibility
    plt.tight_layout()
    plt.show()


def plot_cdf_of_n_mol_per_type(
    mm_component_types: dict[int, dict[str, list[MMComponent]]],
) -> None:
    """
    Plot cumulative distribution function (CDF) of molecule counts per component type.

    Creates a CDF visualization showing the cumulative probability distribution
    of molecule counts across different component types. Uses log10 scaling
    for the x-axis to better visualize the wide range of count values.

    Parameters
    ----------
    mm_component_types : dict[int, dict[str, list[MMComponent]]]
        Hierarchical dictionary containing molecular mechanics components organized
        by specificity level and component type. Structure:
        {specificity_level: {component_type: [component_instances]}}

    Returns
    -------
    None
        Displays the plot using matplotlib.pyplot.show().

    Notes
    -----
    The CDF helps identify the distribution characteristics such as:
    - What percentage of component types have fewer than N molecules
    - The median number of molecules per component type
    - The presence of outlier component types with very high molecule counts

    The log10 transformation on counts helps visualize distributions that
    span several orders of magnitude.

    Examples
    --------
    >>> from molecular_classes import Bond, Angle, ProperTorsion
    >>> import matplotlib.pyplot as plt
    >>> # Component hierarchy with varied frequencies
    >>> components = {
    ...     0: {
    ...         "rare_bond": [Bond(...)],
    ...         "common_bond": [Bond(...) for _ in range(100)]
    ...     },
    ...     1: {
    ...         "generic_angle": [Angle(...) for _ in range(1000)]
    ...     }
    ... }
    >>> plot_cdf_of_n_mol_per_type(components)
    >>> plt.show()  # Shows CDF with log-scale x-axis
    """

    counts = list(flatten_mm_component_types(mm_component_types).values())
    sorted_counts = np.log10(np.sort(counts))  # Use log scale for better visibility
    # sorted_counts = np.sort(counts)
    cdf = np.arange(1, len(sorted_counts) + 1) / len(sorted_counts)

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.lineplot(x=sorted_counts, y=cdf, ax=ax)
    ax.set_xlabel("Log10(Number of Molecules per Component Type)")
    ax.set_ylabel("Cumulative Probability")
    ax.set_title("CDF of Number of Molecules per Component Type")
    plt.tight_layout()
    plt.show()


def show_molecule_with_atom_indices(
    mapped_smiles: str, highlight_indices: tuple[int, ...]
) -> Image.Image:
    """
    Generate 2D molecular structure visualization with highlighted atoms.

    Creates a 2D molecular structure image with specified atoms highlighted
    in green and labeled with their indices. Useful for visualizing specific
    molecular mechanics components like bonds, angles, or torsions.

    Parameters
    ----------
    mapped_smiles : str
        Mapped SMILES string representing the molecule to visualize.
        Atom mapping format should be [element:index] (e.g., "[C:1][C:2][O:3]").
    highlight_indices : tuple[int, ...]
        Tuple of atom indices (0-based) to highlight and label in the structure.
        These atoms will be colored green and labeled with their index numbers.
        Should correspond to the mapped indices in the SMILES string.

    Returns
    -------
    PIL.Image.Image
        PIL Image object containing the 2D molecular structure with highlighted
        atoms. Can be displayed using .show() or saved using .save().

    Notes
    -----
    This function uses RDKit's drawing capabilities with Cairo backend for
    high-quality molecular structure rendering. The molecule is automatically
    assigned 2D coordinates if not already present.

    Highlighted atoms are colored green and labeled with their index numbers
    to clearly identify the component being visualized.

    Examples
    --------
    >>> # Visualize a bond between atoms 0 and 1 in ethanol
    >>> img = show_molecule_with_atom_indices("[C:1][C:2][O:3]", (0, 1))
    >>> img.show()  # Display the image
    >>> img.save("ethanol_bond.png")  # Save to file

    >>> # Highlight an angle involving atoms 0, 1, 2
    >>> angle_img = show_molecule_with_atom_indices("[C:1][C:2][C:3][O:4]", (0, 1, 2))
    >>> angle_img.show()

    >>> # Visualize a torsion with all four atoms highlighted
    >>> torsion_img = show_molecule_with_atom_indices(
    ...     "[C:1][C:2][C:3][C:4]", (0, 1, 2, 3)
    ... )
    >>> torsion_img.show()
    """
    # Parse the molecule from the mapped SMILES
    mol = Molecule.from_mapped_smiles(mapped_smiles, allow_undefined_stereo=True)
    rdmol = mol.to_rdkit()  # Convert to RDKit molecule for visualization

    rdDepictor.Compute2DCoords(rdmol)

    # Set up drawer
    drawer = rdMolDraw2D.MolDraw2DCairo(300, 300)
    options = drawer.drawOptions()

    # Add red labels only for specified atoms
    for idx in highlight_indices:
        options.atomLabels[idx] = f"{idx}"

    # Draw with highlights
    drawer.DrawMolecule(
        rdmol,
        highlightAtoms=highlight_indices,
        # Colour the highlighted atoms green
        highlightAtomColors={idx: (0.0, 1.0, 0.0) for idx in highlight_indices},
    )
    drawer.FinishDrawing()

    # Convert to PIL image
    img_data = drawer.GetDrawingText()
    return Image.open(io.BytesIO(img_data))
