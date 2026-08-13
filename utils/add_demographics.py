"""Attach participant race and gender to the analysis frame.

Reads data/processed/applications_pruned.csv, collapses demographics to
one record per participant, and left-joins onto Frame A as built by
lg.prep().

Run directly to inspect and write the merged frame:

    python add_demographics.py

Or import attach_demographics() into lg.py to fold this into the
pipeline permanently.
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Paths and lg import
# ---------------------------------------------------------------------

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

_LG = ROOT / "models" / "lg.py"
_spec = importlib.util.spec_from_file_location("lg", _LG)
lg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lg)

SOURCE = ROOT / "data" / "processed" / "applications.csv"

OUTDIR = ROOT / "data" / "models"
OUTDIR.mkdir(parents=True, exist_ok=True)

OUTFILE = OUTDIR / "analysis_frame_with_demographics.csv"


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

# Set these explicitly if auto-detection picks the wrong column.
RACE_COL: str | None = None
GENDER_COL: str | None = None
YEAR_COL: str | None = None

# Categories below this share of non-missing records collapse to "Other".
# Set to 0.0 to keep every category as-is.
MIN_CATEGORY_SHARE = 0.01

# Values that mean "no usable answer". Compared case-insensitively after
# stripping whitespace and punctuation.
MISSING_TOKENS = {
    "", "na", "n/a", "none", "null", "nan", "unknown", "unspecified",
    "declined", "declined to answer", "decline to answer",
    "prefer not to answer", "prefer not to say", "not reported",
    "no response", "other/unknown", "-", "--", "?",
}

RACE_PATTERNS = [r"^race$", r"race", r"ethnic"]
GENDER_PATTERNS = [r"^gender$", r"gender", r"^sex$"]
YEAR_PATTERNS = [r"^year$", r"program.?year", r"app.?year"]


# ---------------------------------------------------------------------
# Column detection
# ---------------------------------------------------------------------

def detect_column(columns, patterns, override=None, label=""):
    """First column matching the earliest pattern that hits anything."""
    if override is not None:
        if override not in columns:
            raise KeyError(
                f"{label} column {override!r} not in source file. "
                f"Available: {list(columns)}"
            )
        return override

    for pattern in patterns:
        hits = [c for c in columns if re.search(pattern, c, re.IGNORECASE)]
        if hits:
            return hits[0]

    return None


# ---------------------------------------------------------------------
# Value cleaning
# ---------------------------------------------------------------------

def normalise(value):
    """Trim, collapse whitespace, and map non-answers to NaN."""
    if pd.isna(value):
        return np.nan

    text = re.sub(r"\s+", " ", str(value)).strip()

    if text.lower().strip(".") in MISSING_TOKENS:
        return np.nan

    return text


def collapse_rare(series: pd.Series, min_share: float) -> pd.Series:
    """Fold thin categories into 'Other' so model cells stay estimable."""
    if min_share <= 0:
        return series

    shares = series.value_counts(normalize=True, dropna=True)
    rare = set(shares[shares < min_share].index)

    if not rare:
        return series

    print(f"    collapsing to 'Other': {sorted(rare)}")

    return series.where(~series.isin(rare), "Other")


# ---------------------------------------------------------------------
# Person-level collapse
# ---------------------------------------------------------------------

def modal_value(series: pd.Series):
    """Most common non-missing value; ties go to the earliest record.

    Assumes the group is already sorted so that row order is
    chronological.
    """
    clean = series.dropna()

    if clean.empty:
        return np.nan

    counts = clean.value_counts()
    top = counts[counts.eq(counts.max())].index

    if len(top) == 1:
        return top[0]

    return clean.iloc[0]


def person_level_demographics(verbose: bool = True) -> pd.DataFrame:
    """One row per participant with a resolved race and gender."""
    if not SOURCE.exists():
        raise FileNotFoundError(f"Source not found: {SOURCE}")

    src = pd.read_csv(SOURCE, low_memory=False)

    if verbose:
        print(f"Source: {SOURCE}")
        print(f"  rows: {len(src):,}   columns: {len(src.columns)}")

    id_col = lg.ID

    if id_col not in src.columns:
        raise KeyError(
            f"Participant id {id_col!r} not in source file.\n"
            f"Columns: {list(src.columns)}"
        )

    race_col = detect_column(src.columns, RACE_PATTERNS, RACE_COL, "Race")
    gender_col = detect_column(
        src.columns, GENDER_PATTERNS, GENDER_COL, "Gender"
    )
    year_col = detect_column(src.columns, YEAR_PATTERNS, YEAR_COL, "Year")

    if verbose:
        print(f"  id column:     {id_col}")
        print(f"  race column:   {race_col}")
        print(f"  gender column: {gender_col}")
        print(f"  year column:   {year_col or '(none - using file order)'}")

    missing = [
        name for name, col in
        [("race", race_col), ("gender", gender_col)]
        if col is None
    ]

    if missing:
        raise KeyError(
            f"Could not auto-detect: {', '.join(missing)}. "
            f"Set RACE_COL / GENDER_COL at the top of this script.\n"
            f"Columns: {list(src.columns)}"
        )

    keep = [id_col, race_col, gender_col] + ([year_col] if year_col else [])
    src = src[keep].copy()

    src["race"] = src[race_col].map(normalise)
    src["gender"] = src[gender_col].map(normalise)

    # Chronological order so modal ties resolve to the earliest record.
    if year_col:
        src = src.sort_values([id_col, year_col], kind="mergesort")

    # How often does a person carry conflicting values?
    if verbose:
        print("\n--- Within-person consistency ---")

        for field in ("race", "gender"):
            distinct = (
                src.groupby(id_col)[field]
                .nunique(dropna=True)
            )
            conflicted = int((distinct > 1).sum())
            observed = int((distinct > 0).sum())

            pct = 100 * conflicted / observed if observed else 0.0

            print(
                f"  {field:7s}: {conflicted:,} of {observed:,} participants "
                f"with >1 distinct value ({pct:.2f}%)"
            )

    people = (
        src
        .groupby(id_col, sort=False)
        .agg(race=("race", modal_value), gender=("gender", modal_value))
        .reset_index()
    )

    if verbose:
        print("\n--- Category cleanup ---")
        print("  race:")

    people["race"] = collapse_rare(people["race"], MIN_CATEGORY_SHARE)

    if verbose:
        print("  gender:")

    people["gender"] = collapse_rare(people["gender"], MIN_CATEGORY_SHARE)

    # Non-response is kept as a level. Dropping it would quietly delete a
    # group that may differ systematically on the outcome.
    people["race"] = people["race"].fillna("Unknown")
    people["gender"] = people["gender"].fillna("Unknown")

    if verbose:
        print(f"\nPerson-level demographics: {len(people):,} participants")

    return people


# ---------------------------------------------------------------------
# Join
# ---------------------------------------------------------------------

def attach_demographics(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Left-join person-level race and gender onto an analysis frame."""
    people = person_level_demographics(verbose=verbose)

    before = len(df)

    out = df.merge(people, on=lg.ID, how="left", validate="many_to_one")

    if len(out) != before:
        raise RuntimeError(
            f"Join changed row count: {before:,} -> {len(out):,}. "
            "The demographics table is not unique on participant id."
        )

    # Frame rows with no match at all in the source file.
    unmatched = out["race"].isna()

    if verbose:
        print(
            f"\nUnmatched frame rows: {int(unmatched.sum()):,} "
            f"({100 * unmatched.mean():.2f}%)"
        )

    out["race"] = out["race"].fillna("Unknown")
    out["gender"] = out["gender"].fillna("Unknown")

    # Largest group as reference level keeps model output interpretable.
    for field in ("race", "gender"):
        order = (
            out.loc[out[field].ne("Unknown"), field]
            .value_counts()
            .index
            .tolist()
        )

        if "Other" in order:
            order = [c for c in order if c != "Other"] + ["Other"]

        order = order + ["Unknown"]

        out[field] = pd.Categorical(out[field], categories=order, ordered=False)

    return out


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    dfA = lg.prep()

    print(f"Frame A: {len(dfA):,} person-years, "
          f"{dfA[lg.ID].nunique():,} participants\n")

    out = attach_demographics(dfA)

    print("\n--- Composition of the analysis frame ---")

    for field in ("race", "gender"):
        tab = (
            out[field]
            .value_counts(dropna=False)
            .rename("n")
            .to_frame()
        )
        tab["percent"] = 100 * tab["n"] / len(out)

        print(f"\n{field}:")
        print(tab.to_string(float_format=lambda v: f"{v:.1f}"))

    out.to_csv(OUTFILE, index=False)

    print(f"\nWritten: {OUTFILE}")
    print(f"  rows: {len(out):,}   columns: {len(out.columns)}")


if __name__ == "__main__":
    main()
