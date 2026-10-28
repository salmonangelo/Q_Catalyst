"""Unit tests for mutation parsing, validation, sequence preparation, and 5XJH mapping."""

import pytest

from qcatalyst.data.mutation import (
    apply_mutations,
    canonicalize_mutations,
    map_mutation_positions_to_5xjh,
    parse_mutation_token,
    parse_mutations,
    validate_mutations,
)


def test_parse_valid_single_mutation_token():
    wt, pos, mut = parse_mutation_token("N37P")
    assert wt == "N"
    assert pos == 37
    assert mut == "P"

    # Case insensitivity
    wt, pos, mut = parse_mutation_token("w104l")
    assert wt == "W"
    assert pos == 104
    assert mut == "L"


@pytest.mark.parametrize(
    "invalid_token",
    [
        "123",
        "N37",
        "37P",
        "N-5P",
        "N0P",
        "N37B_invalid",
        "NN37P",
        "N37PP",
        "",
        "   ",
    ],
)
def test_parse_invalid_mutation_token(invalid_token):
    with pytest.raises(ValueError):
        parse_mutation_token(invalid_token)


def test_parse_multiple_mutations():
    # Colon separator
    parsed = parse_mutations("S214H:I168R:W159H")
    assert len(parsed) == 3
    assert parsed[0] == ("S", 214, "H")
    assert parsed[1] == ("I", 168, "R")
    assert parsed[2] == ("W", 159, "H")

    # Comma separator
    parsed = parse_mutations("A1M,C3D")
    assert len(parsed) == 2
    assert parsed[0] == ("A", 1, "M")
    assert parsed[1] == ("C", 3, "D")


@pytest.mark.parametrize("wt_indicator", ["WT", "wt", "Wildtype", "wild-type", "none", "nan", "", None])
def test_parse_wildtype_representations(wt_indicator):
    assert parse_mutations(wt_indicator) == []
    assert canonicalize_mutations(wt_indicator) == "WT"


def test_canonicalize_mutations_sorting():
    # Unordered multi-mutant should sort by position ascending
    canonical = canonicalize_mutations("S214H:I168R:W159H:A10G")
    assert canonical == "A10G:W159H:I168R:S214H"


def test_validate_mutations_success():
    seq = "MNFPRASRLMQA"
    is_valid, errors = validate_mutations(seq, "M1A:F3W:A6G")
    assert is_valid
    assert len(errors) == 0


def test_validate_mutations_out_of_bounds():
    seq = "MNFP"
    is_valid, errors = validate_mutations(seq, "M1A:P5G")
    assert not is_valid
    assert any("out of sequence bounds" in err for err in errors)


def test_validate_mutations_residue_mismatch():
    seq = "MNFP"  # Position 2 is N
    is_valid, errors = validate_mutations(seq, "A2G")
    assert not is_valid
    assert any("Residue mismatch at position 2" in err for err in errors)


def test_validate_mutations_conflicting_positions():
    seq = "MNFP"
    is_valid, errors = validate_mutations(seq, "N2A:N2K")
    assert not is_valid
    assert any("Multiple conflicting mutations" in err for err in errors)


def test_validate_silent_mutation():
    seq = "MNFP"
    is_valid, errors = validate_mutations(seq, "N2N")
    assert not is_valid
    assert any("Silent mutation" in err for err in errors)


def test_apply_mutations():
    seq = "ABCDE"
    mutated = apply_mutations(seq, "A1M:C3D")
    assert mutated == "MBDDE"
    assert len(mutated) == len(seq)


def test_apply_mutations_raises_on_invalid():
    seq = "ABCDE"
    with pytest.raises(ValueError):
        apply_mutations(seq, "Z1M", strict=True)


def test_map_mutation_positions_to_5xjh():
    ispetase_seq = "M" * 36 + "N" + "A" * 253  # Len 290, pos 37 is N
    bhr_seq = "M" * 36 + "N" + "A" * 256        # Len 293

    # IsPETase position 37 maps 1:1 to 5XJH position 37
    pos_str, status = map_mutation_positions_to_5xjh(
        "IsPETase", [("N", 37, "P")], ispetase_seq, ispetase_seq
    )
    assert pos_str == "37"
    assert status == "exact_1to1"

    # IsPETase signal peptide position 10 (<37) maps to None
    pos_str, status = map_mutation_positions_to_5xjh(
        "IsPETase", [("M", 10, "A")], ispetase_seq, ispetase_seq
    )
    assert pos_str == "None"
    assert status == "unmapped"

    # Homolog alignment mapping
    pos_str, status = map_mutation_positions_to_5xjh(
        "BhrPETase", [("N", 37, "P")], bhr_seq, ispetase_seq
    )
    assert status in ("aligned", "partial_aligned", "unmapped")
