"""
Tests for get_atom_recursive_smirks in process_SMIRKS.py.

Checks that:
  - recursion_level=0  returns the base atom pattern only
  - recursion_level=1  produces valid RDKit-parseable SMARTS
  - the assembled SMARTS matches *exactly* the intended atom class
  - return_recursions=True returns a list of per-neighbor strings
  - single-neighbor atoms (H, terminal halogens) do not produce empty parentheses
  - ring atoms and multiple bond types are handled correctly
"""

import pytest
from rdkit import Chem
from from_finlay.process_SMIRKS import get_atom_recursive_smirks


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def make_mol(smiles: str) -> Chem.Mol:
    """Return an explicit-H RDKit molecule."""
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, f"Bad SMILES: {smiles}"
    return Chem.AddHs(mol)


def smarts_is_valid(smarts: str) -> bool:
    """Return True if RDKit can parse the SMARTS."""
    return Chem.MolFromSmarts(smarts) is not None


def matching_atom_indices(mol: Chem.Mol, smarts: str) -> list[int]:
    """Return atom indices that match a single-atom SMARTS pattern."""
    query = Chem.MolFromSmarts(smarts)
    assert query is not None, f"Invalid SMARTS: {smarts!r}"
    return [match[0] for match in mol.GetSubstructMatches(query)]


# ---------------------------------------------------------------------------
# recursion_level=0  →  base pattern only, no recursion
# ---------------------------------------------------------------------------


class TestLevel0:
    def test_carbon_base(self):
        mol = make_mol("CC")  # ethane
        pat = get_atom_recursive_smirks(0, mol, recursion_level=0)
        assert pat == "[#6X4]", f"Unexpected base: {pat!r}"

    def test_nitrogen_base(self):
        mol = make_mol("CN")  # methylamine
        pat = get_atom_recursive_smirks(1, mol, recursion_level=0)
        # N in methylamine + explicit H: degree 3 (1C + 2H)
        assert pat == "[#7X3]", f"Unexpected base: {pat!r}"

    def test_oxygen_base(self):
        mol = make_mol("CO")  # methanol
        pat = get_atom_recursive_smirks(1, mol, recursion_level=0)
        # O in methanol + explicit H: degree 2 (1C + 1H)
        assert pat == "[#8X2]", f"Unexpected base: {pat!r}"

    def test_hydrogen_base(self):
        mol = make_mol("CC")
        # First H – index depends on AddHs ordering; grab any #1 atom
        h_idx = next(a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 1)
        pat = get_atom_recursive_smirks(h_idx, mol, recursion_level=0)
        assert pat == "[#1X1]", f"Unexpected base: {pat!r}"

    def test_return_recursions_empty_at_level0(self):
        mol = make_mol("CC")
        result = get_atom_recursive_smirks(
            0, mol, recursion_level=0, return_recursions=True
        )
        assert result == [], f"Expected empty list, got {result!r}"


# ---------------------------------------------------------------------------
# recursion_level=1  →  valid SMARTS + correct matching
# ---------------------------------------------------------------------------


class TestLevel1ValidSMARTS:
    """Every assembled pattern must be parseable by RDKit."""

    @pytest.mark.parametrize(
        "smiles,idx",
        [
            ("CC", 0),  # sp3 carbon, multiple H neighbours
            ("CN", 0),  # carbon bonded to N
            ("CN", 1),  # nitrogen bonded to C
            ("CO", 1),  # oxygen bonded to C  (degree 2)
            ("C=O", 0),  # sp2 carbon, double bond
            ("C=O", 1),  # carbonyl oxygen
            ("c1ccccc1", 0),  # aromatic carbon
            ("CCCl", 2),  # chlorine (single heavy-atom neighbour)
        ],
    )
    def test_valid_smarts(self, smiles, idx):
        mol = make_mol(smiles)
        pat = get_atom_recursive_smirks(idx, mol, recursion_level=1)
        assert smarts_is_valid(pat), f"Invalid SMARTS for {smiles!r} idx={idx}: {pat!r}"

    def test_no_empty_parentheses(self):
        """Terminal atoms (degree 1 heavy: Cl, F, H) must not produce '()'."""
        for smiles, idx in [("CCl", 1), ("CF", 1), ("CI", 1)]:
            mol = make_mol(smiles)
            pat = get_atom_recursive_smirks(idx, mol, recursion_level=1)
            assert (
                "()" not in pat
            ), f"Empty parens in pattern for {smiles!r} idx={idx}: {pat!r}"


class TestLevel1Matching:
    """Assembled SMARTS should match only atoms of the same environment."""

    def test_carbon_in_methylamine_matches_only_carbon(self):
        mol = make_mol("CN")
        pat = get_atom_recursive_smirks(0, mol, recursion_level=1)
        matched = matching_atom_indices(mol, pat)
        # All matched atoms should be carbon (#6)
        for idx in matched:
            assert (
                mol.GetAtomWithIdx(idx).GetAtomicNum() == 6
            ), f"Pattern {pat!r} matched non-carbon atom {idx}"

    def test_carbon_matches_target_atom(self):
        mol = make_mol("CN")
        pat = get_atom_recursive_smirks(0, mol, recursion_level=1)
        matched = matching_atom_indices(mol, pat)
        assert 0 in matched, f"Pattern {pat!r} did not match atom 0 in CN"

    def test_nitrogen_pattern_does_not_match_carbon(self):
        mol = make_mol("CN")
        pat = get_atom_recursive_smirks(1, mol, recursion_level=1)
        matched = matching_atom_indices(mol, pat)
        for idx in matched:
            assert (
                mol.GetAtomWithIdx(idx).GetAtomicNum() == 7
            ), f"N pattern {pat!r} matched non-nitrogen atom {idx}"

    def test_carbonyl_carbon_vs_sp3_carbon(self):
        """A carbonyl C pattern should NOT match a pure sp3 C."""
        mol_ketone = make_mol("CC(=O)C")
        mol_propane = make_mol("CCC")

        # carbonyl carbon is idx 1 in CC(=O)C heavy atoms (with explicit H)
        # find it: the C bonded to the O
        o_idx = next(a.GetIdx() for a in mol_ketone.GetAtoms() if a.GetAtomicNum() == 8)
        c_carbonyl_idx = mol_ketone.GetAtomWithIdx(o_idx).GetNeighbors()[0].GetIdx()

        pat = get_atom_recursive_smirks(c_carbonyl_idx, mol_ketone, recursion_level=1)

        # should not match propane carbons
        matched_in_propane = matching_atom_indices(mol_propane, pat)
        assert (
            matched_in_propane == []
        ), f"Carbonyl-C pattern {pat!r} incorrectly matched sp3 C in propane"

    def test_aromatic_carbon_not_in_alkane(self):
        mol_benz = make_mol("c1ccccc1")
        mol_hex = make_mol("CCCCCC")
        pat = get_atom_recursive_smirks(0, mol_benz, recursion_level=1)
        matched = matching_atom_indices(mol_hex, pat)
        assert (
            matched == []
        ), f"Aromatic C pattern {pat!r} matched aliphatic C in hexane"


# ---------------------------------------------------------------------------
# return_recursions=True
# ---------------------------------------------------------------------------


class TestReturnRecursions:
    def test_returns_list(self):
        mol = make_mol("CN")
        result = get_atom_recursive_smirks(
            0, mol, recursion_level=1, return_recursions=True
        )
        assert isinstance(result, list)

    def test_length_equals_degree(self):
        """One recursion entry per neighbour (= atom degree)."""
        mol = make_mol("CN")
        for idx in range(mol.GetNumAtoms()):
            degree = mol.GetAtomWithIdx(idx).GetDegree()
            result = get_atom_recursive_smirks(
                idx, mol, recursion_level=1, return_recursions=True
            )
            assert (
                len(result) == degree
            ), f"Atom {idx} degree={degree} but got {len(result)} recursions"

    def test_each_entry_is_string(self):
        mol = make_mol("CN")
        result = get_atom_recursive_smirks(
            0, mol, recursion_level=1, return_recursions=True
        )
        for entry in result:
            assert isinstance(entry, str), f"Non-string entry: {entry!r}"

    def test_bond_type_in_entry(self):
        """With include_bond_order=True, each entry should start with a bond character."""
        mol = make_mol("C=O")  # carbonyl
        result = get_atom_recursive_smirks(
            0, mol, recursion_level=1, return_recursions=True, include_bond_order=True
        )
        bond_chars = {"-", "=", "#", ":", "~"}
        for entry in result:
            assert (
                entry[0] in bond_chars
            ), f"Entry {entry!r} does not start with a bond character"

    def test_wildcard_bond_when_no_bond_order(self):
        mol = make_mol("C=O")
        result = get_atom_recursive_smirks(
            0, mol, recursion_level=1, return_recursions=True, include_bond_order=False
        )
        for entry in result:
            assert entry.startswith("~"), f"Expected '~' bond, got {entry!r}"


# ---------------------------------------------------------------------------
# deeper recursion (level=2)
# ---------------------------------------------------------------------------


class TestLevel2:
    def test_valid_smarts_level2(self):
        mol = make_mol("CCC")
        pat = get_atom_recursive_smirks(1, mol, recursion_level=2)
        assert smarts_is_valid(pat), f"Level-2 SMARTS invalid: {pat!r}"

    def test_matches_middle_carbon(self):
        mol = make_mol("CCC")
        # middle carbon (idx 1) has degree 2 heavy + 2 H
        pat = get_atom_recursive_smirks(1, mol, recursion_level=2)
        matched = matching_atom_indices(mol, pat)
        assert 1 in matched, f"Level-2 pattern did not match middle carbon: {pat!r}"

    def test_benzoic_acid_ipso_carbon_single_match(self):
        """Depth-2 SMIRKS for the ipso ring carbon of benzoic acid should match
        exactly one atom in the molecule (the ipso carbon itself)."""
        mol = make_mol("OC(=O)c1ccccc1")

        # Identify the ipso carbon: aromatic C bonded to the carboxyl C (sp2, non-aromatic)
        ipso_idx = None
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 6 and atom.GetIsAromatic():
                for nbr in atom.GetNeighbors():
                    if nbr.GetAtomicNum() == 6 and not nbr.GetIsAromatic():
                        ipso_idx = atom.GetIdx()
                        break
            if ipso_idx is not None:
                break
        assert ipso_idx is not None, "Could not find ipso carbon in benzoic acid"

        pat = get_atom_recursive_smirks(ipso_idx, mol, recursion_level=2)
        assert smarts_is_valid(pat), f"Depth-2 ipso SMARTS invalid: {pat!r}"

        matched = matching_atom_indices(mol, pat)
        assert len(matched) == 1, (
            f"Expected exactly 1 match for ipso carbon pattern, got {len(matched)}: "
            f"matched indices={matched}, pattern={pat!r}"
        )
        assert (
            matched[0] == ipso_idx
        ), f"Matched atom {matched[0]} is not the ipso carbon {ipso_idx}"

    def test_para_acetylbenzoic_acid_level1_two_ring_types(self):
        """para-Acetylbenzoic acid has 2 chemically distinct ring-carbon environments
        at recursion level 1:
          - 4 unsubstituted H-bearing aromatic carbons (all identical)
          - 2 ipso carbons (COOH-bearing and acetyl-bearing are equivalent at depth 1)
        The two ipso patterns must collapse to the same string after sorting
        the neighbour recursions, giving group sizes [4, 2].
        """
        mol = make_mol("OC(=O)c1ccc(C(=O)C)cc1")
        ring_idxs = [
            a.GetIdx()
            for a in mol.GetAtoms()
            if a.GetAtomicNum() == 6 and a.GetIsAromatic()
        ]
        assert len(ring_idxs) == 6

        pats = [
            get_atom_recursive_smirks(idx, mol, recursion_level=1) for idx in ring_idxs
        ]

        from collections import Counter

        groups = Counter(pats)
        sizes = sorted(groups.values(), reverse=True)

        assert sizes == [4, 2], (
            f"Expected ring-carbon groups [4, 2] at level 1, got {sizes}. "
            f"Patterns: {dict(groups)}"
        )

    def test_para_acetylbenzoic_acid_level2_three_ring_types(self):
        """para-Acetylbenzoic acid has 3 chemically distinct ring-carbon environments
        at recursion level 2:
          - 4 unsubstituted H-bearing ortho/meta carbons (see both ipso environments
            at depth 2, but the two pairs are mirror-symmetric so collapse to one type)
          - 1 ipso carbon adjacent to the COOH group
          - 1 ipso carbon adjacent to the acetyl group
        Group sizes must be [4, 1, 1].
        """
        mol = make_mol("OC(=O)c1ccc(C(=O)C)cc1")
        ring_idxs = [
            a.GetIdx()
            for a in mol.GetAtoms()
            if a.GetAtomicNum() == 6 and a.GetIsAromatic()
        ]
        assert len(ring_idxs) == 6

        pats = [
            get_atom_recursive_smirks(idx, mol, recursion_level=2) for idx in ring_idxs
        ]

        from collections import Counter

        groups = Counter(pats)
        sizes = sorted(groups.values(), reverse=True)

        assert sizes == [4, 1, 1], (
            f"Expected ring-carbon groups [4, 1, 1] at level 2, got {sizes}. "
            f"Patterns: {dict(groups)}"
        )
