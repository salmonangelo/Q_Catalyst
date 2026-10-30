"""Mutation parsing, validation, sequence preparation, and structure mapping.

Provides strict parsing of single and multiple substitutions, verification
against parent FASTA sequence identities and length bounds, canonical sorting,
mutant sequence generation, and alignment-based mapping to the 5XJH reference structure.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Any

from Bio import Align

MUTATION_TOKEN_RE = re.compile(r"^([A-Za-z*])(\d+)([A-Za-z*])$")

# Valid single-letter amino acid codes (including stop codon *)
STANDARD_AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY*")


def parse_mutation_token(token: str) -> tuple[str, int, str]:
    """Parse a single mutation token like 'N37P' or 'W104L'.

    Args:
        token: Mutation string token.

    Returns:
        (wildtype_aa, one_based_position, mutant_aa) all uppercase.

    Raises:
        ValueError: If token format is invalid or characters are unrecognized.
    """
    clean_token = token.strip()
    match = MUTATION_TOKEN_RE.match(clean_token)
    if not match:
        raise ValueError(f"Invalid mutation token syntax: '{clean_token}'")

    wt_aa, pos_str, mut_aa = match.groups()
    wt_aa = wt_aa.upper()
    mut_aa = mut_aa.upper()
    pos = int(pos_str)

    if wt_aa not in STANDARD_AMINO_ACIDS:
        raise ValueError(f"Unrecognized wild-type amino acid '{wt_aa}' in '{token}'")
    if mut_aa not in STANDARD_AMINO_ACIDS:
        raise ValueError(f"Unrecognized mutant amino acid '{mut_aa}' in '{token}'")
    if pos <= 0:
        raise ValueError(f"Position must be a positive 1-based integer, got {pos} in '{token}'")

    return wt_aa, pos, mut_aa


def parse_mutations(mutations_str: str | Any) -> list[tuple[str, int, str]]:
    """Parse a mutation string containing one or more mutations.

    Supports separators ':' (PET-Gym default), ',', and whitespace.
    Recognizes wild-type indicators ('WT', 'wildtype', 'none', '', nan).

    Args:
        mutations_str: Raw mutation string (e.g. 'N37P', 'A1M:C3D', 'WT').

    Returns:
        List of parsed (wt_aa, position, mut_aa) tuples.
    """
    if mutations_str is None:
        return []

    text = str(mutations_str).strip()
    if not text or text.lower() in {"nan", "none", "wt", "wildtype", "wild-type"}:
        return []

    # Replace commas and spaces with colons
    normalized = text.replace(",", ":").replace(";", ":").replace(" ", ":")
    tokens = [t.strip() for t in normalized.split(":") if t.strip()]

    parsed: list[tuple[str, int, str]] = []
    for token in tokens:
        parsed.append(parse_mutation_token(token))

    return parsed


def canonicalize_mutations(mutations: list[tuple[str, int, str]] | str) -> str:
    """Return canonical, sorted, colon-separated representation of mutations.

    Sorts mutations strictly by 1-based position ascending.
    Returns 'WT' for empty mutation sets.
    """
    if isinstance(mutations, str):
        parsed = parse_mutations(mutations)
    else:
        parsed = mutations

    if not parsed:
        return "WT"

    # Sort primarily by position, then wt_aa, then mut_aa
    sorted_muts = sorted(parsed, key=lambda m: (m[1], m[0], m[2]))
    return ":".join(f"{wt}{pos}{mut}" for wt, pos, mut in sorted_muts)


def validate_mutations(
    sequence: str,
    mutations: list[tuple[str, int, str]] | str,
) -> tuple[bool, list[str]]:
    """Validate parsed mutations against parent sequence.

    Checks:
    - 1-based position within [1, len(sequence)]
    - sequence[pos - 1] matches the claimed wildtype amino acid
    - Multiple mutations at the exact same position (flagged as conflict)
    - Silent mutations (wt_aa == mut_aa)

    Args:
        sequence: Parent wild-type amino acid sequence.
        mutations: Parsed mutation tuples or raw string.

    Returns:
        (is_valid, list_of_error_reasons)
    """
    parsed = parse_mutations(mutations) if isinstance(mutations, str) else mutations
    errors: list[str] = []
    seq_len = len(sequence)

    positions_seen: set[int] = set()

    for wt_aa, pos, mut_aa in parsed:
        if pos < 1 or pos > seq_len:
            errors.append(
                f"Position {pos} out of sequence bounds [1, {seq_len}] for mutation {wt_aa}{pos}{mut_aa}"
            )
            continue

        actual_wt = sequence[pos - 1].upper()
        if actual_wt != wt_aa:
            errors.append(
                f"Residue mismatch at position {pos}: mutation claims '{wt_aa}', sequence has '{actual_wt}'"
            )

        if pos in positions_seen:
            errors.append(f"Multiple conflicting mutations specified for position {pos}")
        positions_seen.add(pos)

        if wt_aa == mut_aa:
            errors.append(f"Silent mutation at position {pos}: '{wt_aa}' to '{mut_aa}'")

    return (len(errors) == 0, errors)


def apply_mutations(
    sequence: str,
    mutations: list[tuple[str, int, str]] | str,
    strict: bool = True,
) -> str:
    """Apply substitutions to a parent wild-type sequence.

    Args:
        sequence: Wild-type sequence string.
        mutations: Parsed mutation tuples or raw string.
        strict: If True, validates before applying; raises ValueError on mismatch.

    Returns:
        Mutant sequence string.
    """
    parsed = parse_mutations(mutations) if isinstance(mutations, str) else mutations
    if not parsed:
        return sequence

    if strict:
        is_valid, errors = validate_mutations(sequence, parsed)
        if not is_valid:
            raise ValueError(f"Cannot apply invalid mutations: {'; '.join(errors)}")

    seq_chars = list(sequence)
    for _, pos, mut_aa in parsed:
        seq_chars[pos - 1] = mut_aa

    return "".join(seq_chars)


@lru_cache(maxsize=16)
def get_pairwise_aligner() -> Align.PairwiseAligner:
    """Return configured global pairwise sequence aligner."""
    aligner = Align.PairwiseAligner()
    aligner.mode = "global"
    aligner.match_score = 2
    aligner.mismatch_score = -1
    aligner.open_gap_score = -10
    aligner.extend_gap_score = -0.5
    return aligner


def build_parent_to_ispetase_map(
    parent_seq: str,
    ispetase_seq: str,
) -> dict[int, int | None]:
    """Map 1-based parent residue positions to 1-based IsPETase residue positions.

    Uses global Needleman-Wunsch alignment via Bio.Align.
    """
    aligner = get_pairwise_aligner()
    alignment = aligner.align(parent_seq, ispetase_seq)[0]
    t_aln = alignment[0]
    r_aln = alignment[1]

    mapping: dict[int, int | None] = {}
    t_pos = 0
    r_pos = 0

    for t_char, r_char in zip(t_aln, r_aln):
        if t_char != "-":
            t_pos += 1
            if r_char != "-":
                r_pos += 1
                mapping[t_pos] = r_pos
            else:
                mapping[t_pos] = None
        else:
            if r_char != "-":
                r_pos += 1

    return mapping


def map_mutation_positions_to_5xjh(
    parent_enzyme: str,
    parent_mutations: list[tuple[str, int, str]],
    parent_seq: str,
    ispetase_seq: str,
) -> tuple[str, str]:
    """Map mutation positions on parent sequence to 5XJH reference structure numbering.

    5XJH reference numbering rules:
    - 5XJH Chain A contains modeled residues 37..290 identical to IsPETase UniProt sequence.
    - Residues 30..36 in 5XJH are the N-terminal expression tag ('GSHMRGP').
    - IsPETase residues 1..27 are the signal peptide (unmodeled in 5XJH).
    - For IsPETase:
        - If position in [37, 290]: 1:1 exact mapping to 5XJH residue `pos`.
        - If position < 37: mapped as None ('signal_peptide_unmapped').
    - For non-IsPETase enzymes:
        - Position is mapped via global alignment to IsPETase; if mapped IsPETase residue
          is in [37, 290], structure pos is that residue. Otherwise None.

    Returns:
        (structure_5xjh_positions_str, structure_mapping_status)
        where structure_5xjh_positions_str is colon-separated positions or '' if unmapped.
    """
    if not parent_mutations:
        return ("", "no_mutations")

    mapped_positions: list[str] = []
    statuses: list[str] = []

    if parent_enzyme.lower() == "ispetase":
        for _, pos, _ in sorted(parent_mutations, key=lambda m: m[1]):
            if 37 <= pos <= 290:
                mapped_positions.append(str(pos))
                statuses.append("exact_1to1")
            elif pos < 37:
                mapped_positions.append("None")
                statuses.append("signal_peptide_unmapped")
            else:
                mapped_positions.append("None")
                statuses.append("c_terminal_unmapped")
    else:
        # Non-IsPETase homolog: align to IsPETase
        pos_map = build_parent_to_ispetase_map(parent_seq, ispetase_seq)
        for _, pos, _ in sorted(parent_mutations, key=lambda m: m[1]):
            ispet_pos = pos_map.get(pos)
            if ispet_pos is not None and 37 <= ispet_pos <= 290:
                mapped_positions.append(str(ispet_pos))
                statuses.append("aligned")
            elif ispet_pos is not None:
                mapped_positions.append("None")
                statuses.append("aligned_to_unmodeled")
            else:
                mapped_positions.append("None")
                statuses.append("unaligned_gap")

    # Aggregate overall status
    if all(s == "exact_1to1" for s in statuses):
        overall_status = "exact_1to1"
    elif all(s == "aligned" for s in statuses):
        overall_status = "aligned"
    elif any(s in ("aligned", "exact_1to1") for s in statuses):
        overall_status = "partial_aligned"
    else:
        overall_status = "unmapped"

    positions_str = ":".join(mapped_positions)
    return positions_str, overall_status
