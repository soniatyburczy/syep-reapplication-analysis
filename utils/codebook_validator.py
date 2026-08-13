"""
codebook_validator.py
======================
A script for validating free-text survey/application responses against
a codebook of canonical categories (e.g. gender, sexual orientation, borough,
sector taxonomy, etc.) using fuzzy string matching.

WORKFLOW
--------
1. Build a `CodebookConfig` describing which column holds the raw text and
   which codebook (dict of label -> list of canonical values) to match against.
2. Call `run()` on your dataframe. This produces a review CSV listing every
   DISTINCT normalized value found, with a suggested label, a match score,
   and a decision ('exact' / 'auto' / 'REVIEW...' / 'fallback').
3. Open that CSV and hand-correct the `label` column wherever the decision
   wasn't 'exact' or 'auto' (or wherever you disagree with an auto match).
4. Call apply_codebook() to apply the corrected labels back onto every row of 
   your full dataframe, aligned by normalized value (not row order). 
   By default a new coded column is created; pass overwrite=True to replace 
   the original free-text column.

Any matching gap in step 4 prints a warning listing the unmatched distinct
values and their counts, so you can add them to the codebook and re-run.

USAGE
-----
    from codebook_validator import CodebookConfig, run, apply_codebook

    orientation_cfg = CodebookConfig(
        id_col='Application.ID',
        raw_col='Sexual.Orientation.Other',
        codebook=orientation_groups,
        out_path='data/processed/sexuality_review.csv',
    )

    # Step 1: Build the review CSV.
    review = run(df, orientation_cfg)

    # Step 2: Open the CSV and hand-correct the `label` column.

    # Step 3: Apply the reviewed labels back to the dataframe.
    # By default this creates a new <raw_col>_coded column.
    # Pass overwrite=True to replace the original free-text column instead.
    coded_df = apply_codebook(df, orientation_cfg)
    # or:
    # coded_df = apply_codebook(df, orientation_cfg, overwrite=True)

Repeat this workflow for each free-text field (e.g. gender, borough,
sexual orientation, sector, etc.) by creating a new `CodebookConfig`
with the appropriate `raw_col`, `codebook`, and `out_path`.

See the runnable starter template at the bottom of this file
(`python codebook_validator.py`) for a complete end-to-end example.

DEPENDENCIES
------------
numpy, pandas, rapidfuzz

STATUS & PROVENANCE
-------------------
Tested: wide-form use (text in `raw_col`, `field_col=None`). Exercised
end-to-end on gender, sexual-orientation, and borough codebooks against
real SYEP application data.

Untested: long-form use (`field_col` / `field_value` set). The filtering
path runs but has never been validated against real long-form data —
check your row counts after `run()` before trusting the output.

Drafted & debugged with Claude (Opus 5 / Sonnet 5); 
specified, reviewed, and tested by Sonia Tyburczy, 
summer 2026 analytics intern.

Revised 2026-07-30 (bugfix pass): the overwrite=True path was silently
skipping the "unmatched" warning; the codebook duplicate check compared
raw values instead of normalized ones; a mistyped field_col failed
silently instead of raising; an empty codebook could crash extractOne
with an unhelpful error; id_col wasn't actually showing up in the review
CSV despite the docstring claiming it would.

Contact for questions/concerns:
github.com/soniatyburczy · me@soniatyburczy.com
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process


# =============================================================================
# CONFIG
# =============================================================================

DEFAULT_NULL_TOKENS = {'', 'n/a', 'na', 'none', 'null', '-', '.', '--', '?'}


@dataclass
class CodebookConfig:
    """
    Everything the pipeline needs to validate one free-text column.

    Note on `fallback_label`: values that score below `low` (and non-string
    values) are assigned this label. It does NOT have to be a key in your
    codebook — it's written to the review CSV as plain text for you to
    correct by hand. If your codebook has no natural "I couldn't categorize
    this" bucket, either add one or set `fallback_label` to whatever your
    codebook calls it, so the review output doesn't contain a label that
    isn't in your taxonomy.
    """
    id_col: str                      # retained for traceability in the review file;
                                     # row alignment is guaranteed by the value_norm
                                     # join in apply_codebook(), NOT by this column
    raw_col: str                     # column holding the RAW free-text answer
    codebook: dict                   # label -> list of canonical values
    out_path: str                    # where the review CSV gets written
    field_col: Optional[str] = None  # long form only: column to filter on
    field_value: Optional[str] = None  # long form only: value field_col must equal
    coded_col: Optional[str] = None  # name of the new coded column; defaults to <raw_col>_coded
    high: float = 88.0               # >=high: auto-assign
    low: float = 68.0                # high>..>=low: review; <low: fallback
    fallback_label: str = 'ambiguous'  # label given to sub-`low` scores; see docstring
    null_tokens: set = field(default_factory=lambda: set(DEFAULT_NULL_TOKENS))
    heavy_labels: set = field(default_factory=lambda: {'hostile'})  # never auto-assign

    def __post_init__(self):
        if self.coded_col is None:
            self.coded_col = f'{self.raw_col}_coded'
        _validate_codebook(self.codebook, self.null_tokens)


# =============================================================================
# VALIDATION / NORMALIZATION HELPERS
# =============================================================================

def normalize(s, null_tokens=DEFAULT_NULL_TOKENS):
    if pd.isna(s):
        return np.nan
    s = ' '.join(str(s).strip().lower().split())
    return np.nan if s in null_tokens else s


def _validate_codebook(codebook: dict, null_tokens: set = DEFAULT_NULL_TOKENS) -> None:
    """
    Catch a value accidentally listed in two groups before it causes silent
    mislabeling. Checks NORMALIZED values (not raw strings) since that's
    what actually gets used as the dict key in _build_canon() -- two raw
    entries like 'Non-Binary' and 'non-binary ' look different but collapse
    to the same canon key, and a raw-string check alone would miss that
    collision. (Uses ValueError rather than assert so the check survives
    `python -O`, which strips asserts.)
    """
    all_vals = [normalize(v, null_tokens) for vals in codebook.values() for v in vals]
    all_vals = [v for v in all_vals if isinstance(v, str)]
    dupes = sorted({v for v, c in Counter(all_vals).items() if c > 1})
    if dupes:
        raise ValueError(f"normalized value(s) in multiple codebook groups: {dupes}")


def _build_canon(codebook: dict, null_tokens: set):
    """value_norm -> label lookup, plus the flat list fuzzy-matching needs."""
    canon = {}
    for label, vals in codebook.items():
        for v in vals:
            nv = normalize(v, null_tokens)
            if isinstance(nv, str):
                canon[nv] = label
    return canon, list(canon.keys())


def is_multitoken(nv: str) -> bool:
    # a word + a pronoun-set, or 3+ tokens => matcher may silently ignore part
    return (' ' in nv and '/' in nv) or len(nv.split()) >= 3


# =============================================================================
# CLASSIFICATION
# =============================================================================

def classify(nv, canon: dict, canon_choices: list, cfg: CodebookConfig):
    """
    Return (label, matched_canonical, score, decision).

    The three guards below exist because a high fuzzy score is not the same
    thing as a correct match. Each one catches a failure mode observed in
    real response data — please don't remove them without checking that the
    underlying problem is actually gone.
    """
    if not isinstance(nv, str):
        return cfg.fallback_label, '', 0.0, 'fallback'
    if nv in canon:                                   # exact -> done
        return canon[nv], nv, 100.0, 'exact'

    if not canon_choices:
        # Every codebook value normalized away to nothing (e.g. an empty
        # codebook, or a codebook whose only entries collide with
        # null_tokens). Nothing to fuzzy-match against.
        raise ValueError(
            "codebook has no usable canonical values after normalization -- "
            "check that cfg.codebook isn't empty and that its values don't "
            "all fall into cfg.null_tokens"
        )

    match, score, _ = process.extractOne(nv, canon_choices, scorer=fuzz.WRatio)
    label = canon[match]

    # GUARD 1: multi-token entries never auto-assign.
    # WRatio scores against the best-matching SUBSTRING, so a compound answer
    # like "nonbinary she/they" scores ~95 against the canonical "she/they"
    # and the identity term is silently dropped — the response gets coded
    # 'schema' (pronouns) when it should be 'true_other'. Any multi-token
    # value therefore gets human eyes regardless of how high it scored.
    if is_multitoken(nv):
        return label, match, score, 'REVIEW(multi)'

    # GUARD 2: partial_ratio >> ratio means the same substring problem as
    # guard 1, but for values that aren't obviously multi-token. A large gap
    # between "matches somewhere inside" and "matches as a whole" is the
    # signature of the matcher latching onto a fragment and ignoring the
    # rest. 25 points was tuned by hand against real data; it's a heuristic,
    # not a derived threshold.
    if score >= cfg.high and (fuzz.partial_ratio(nv, match) - fuzz.ratio(nv, match)) >= 25:
        return label, match, score, 'REVIEW(partial)'

    # GUARD 3: heavy labels are never auto-assigned. This is about asymmetric
    # ERROR COST, not accuracy — the matcher is no worse at 'hostile' than at
    # anything else, but wrongly tagging a sincere response as hostile is a
    # much more damaging mistake than wrongly tagging it 'redundant', and it
    # is the kind of mistake that ends up in a published table. Cheap
    # insurance: make a human confirm every one.
    if score >= cfg.high and label in cfg.heavy_labels:
        return label, match, score, f'REVIEW({label})'

    if score >= cfg.high:
        return label, match, score, 'auto'
    if score >= cfg.low:
        return label, match, score, 'REVIEW'
    return cfg.fallback_label, match, score, 'fallback'   # uncategorizable


# =============================================================================
# PIPELINE
# =============================================================================

_DECISION_ORDER = {
    'fallback': 0, 'REVIEW': 1, 'REVIEW(multi)': 2,
    'REVIEW(partial)': 3, 'auto': 5, 'exact': 6,
}


def _decision_sort_key(decision: str) -> int:
    # any REVIEW(<heavy_label>) variant not explicitly listed sorts with the
    # other REVIEW buckets rather than falling through to the 9-catchall
    if decision in _DECISION_ORDER:
        return _DECISION_ORDER[decision]
    if decision.startswith('REVIEW'):
        return 4
    return 9


def _filter_field(df: pd.DataFrame, cfg: CodebookConfig) -> pd.DataFrame:
    if not cfg.field_col:
        return df
    if cfg.field_col not in df.columns:
        # Fail loudly. A silent no-op here would mean apply_codebook() runs
        # against the WRONG rows in long-form use with no signal that
        # anything went wrong -- worse than a KeyError.
        raise KeyError(
            f"cfg.field_col={cfg.field_col!r} not found in dataframe columns; "
            "check for a typo, or unset field_col for wide-form use"
        )
    out = df[df[cfg.field_col] == cfg.field_value]
    if out.empty:
        print(
            f"WARNING: filtering on {cfg.field_col}=={cfg.field_value!r} "
            "produced 0 rows. Double-check cfg.field_value against the "
            "actual values in that column."
        )
    return out


def run(filtered_df: pd.DataFrame, cfg: CodebookConfig) -> pd.DataFrame:
    """
    Build the review CSV: one row per DISTINCT normalized value, with a
    suggested label, match score, and decision. Rows needing attention
    (fallback / REVIEW*) float to the top.
    """
    df = _filter_field(filtered_df, cfg)

    canon, canon_choices = _build_canon(cfg.codebook, cfg.null_tokens)

    work = df[[c for c in (cfg.id_col, cfg.raw_col) if c in df.columns]].copy()
    work['value_norm'] = work[cfg.raw_col].map(lambda s: normalize(s, cfg.null_tokens))
    work = work.dropna(subset=['value_norm'])

    has_id = cfg.id_col in work.columns
    agg_kwargs = dict(n=('value_norm', 'size'), example_raw=(cfg.raw_col, 'first'))
    if has_id:
        agg_kwargs['example_id'] = (cfg.id_col, 'first')
    distinct = work.groupby('value_norm').agg(**agg_kwargs).reset_index()

    rows = []
    for _, r in distinct.iterrows():
        label, matched, score, decision = classify(r['value_norm'], canon, canon_choices, cfg)
        row = {'value_norm': r['value_norm'],
               'example_raw': r['example_raw'],
               'n': r['n'],
               'matched_canonical': matched,
               'label': label,
               'score': round(score, 1),
               'decision': decision}
        if has_id:
            # One example id per distinct value, purely so a human reviewer
            # can jump back to a real row -- NOT used for row alignment.
            # apply_codebook() aligns strictly by value_norm.
            row['example_id'] = r['example_id']
        rows.append(row)

    review = pd.DataFrame(rows)
    review['_ord'] = review['decision'].map(_decision_sort_key)
    review = (review.sort_values(['_ord', 'score'])
                    .drop(columns='_ord')
                    .reset_index(drop=True))

    review.to_csv(cfg.out_path, index=False, encoding='utf-8-sig')  # Excel-safe
    print(f"Wrote {len(review)} distinct value(s) to {cfg.out_path} "
          f"({(review['decision'] != 'exact').sum()} need a look).")
    return review


def apply_codebook(
    filtered_df: pd.DataFrame,
    cfg: CodebookConfig,
    corrected_csv: Optional[str] = None,
    overwrite: bool = False,
) -> pd.DataFrame:
    """
    Apply a reviewed codebook to every row of `filtered_df`, aligned by
    normalized value (NOT by row order — the join key guarantees alignment,
    so every id keeps its correct label).

    `corrected_csv` is your hand-reviewed review file (defaults to
    `cfg.out_path`). Its `label` column is treated as the source of truth.

    By default, the reviewed labels are written to a new column
    (`cfg.coded_col`). Pass `overwrite=True` to replace the original
    free-text column (`cfg.raw_col`) with the reviewed labels instead.
    """
    corrected_csv = corrected_csv or cfg.out_path
    df = _filter_field(filtered_df, cfg).copy()

    df["value_norm"] = df[cfg.raw_col].map(
        lambda s: normalize(s, cfg.null_tokens)
    )

    cb = (
        pd.read_csv(corrected_csv, encoding="utf-8-sig")[["value_norm", "label"]]
        .dropna(subset=["value_norm"])
        .drop_duplicates("value_norm")
    )

    # Left join: rows with a real answer get a label; blanks/nulls stay NaN.
    df = df.merge(cb, on="value_norm", how="left")

    # Any non-null response that didn't get a label? (codebook gap)
    # This MUST be computed here, off the freshly-merged `label` column --
    # not after the overwrite branch below, since overwrite only touches
    # rows that DID get a label and leaves raw_col (with real text still
    # sitting in it) untouched for the ones that didn't. Checking
    # raw_col.isna() post-overwrite would never catch those rows.
    unmatched_mask = df["value_norm"].notna() & df["label"].isna()

    if overwrite:
        # Only overwrite rows that actually received a label.
        mask = df["label"].notna()
        df.loc[mask, cfg.raw_col] = df.loc[mask, "label"]
        df = df.drop(columns="label")
        label_col = cfg.raw_col
    else:
        df = df.rename(columns={"label": cfg.coded_col})
        label_col = cfg.coded_col

    unmatched = df[unmatched_mask]
    if len(unmatched):
        print(
            f"WARNING: {len(unmatched)} row(s) have text but no label "
            f"({unmatched['value_norm'].nunique()} distinct). "
            "Add them to the codebook (or the review CSV) and re-run."
        )
        print(unmatched["value_norm"].value_counts().head(20).to_string())

    return df.drop(columns="value_norm")

# ---- 1. Define your codebook: label -> list of raw values that mean it ----
EXAMPLE_CODEBOOK = {
    'male': ['male', 'm'],
    'female': ['female', 'f'],
    'nonbinary': ['nonbinary', 'non-binary', 'enby'],
    'true_other': ['genderfluid', 'genderqueer'],
    'hostile': ['attack helicopter'],
    'declined': ['none of your business', 'prefer not to say', 'rather not say'],
}


def example() -> None:
    """Runnable end-to-end demo on a small in-memory frame."""
    # ---- 2. Point the config at your data ---------------------------------
    cfg = CodebookConfig(
        id_col='Application.ID',              # your unique row id
        raw_col='Gender.Other',               # the free-text column
        codebook=EXAMPLE_CODEBOOK,            # the dict from step 1
        out_path='gender_review.csv',         # where the review file lands
        # field_col='Question.Name',          # long-form only (UNTESTED — see
        # field_value='Preferred.Gender.Other',  # STATUS block at top of file)
    )

    # ---- 3. Swap this for your real dataframe ------------------------------
    df = pd.DataFrame({
        'Application.ID': range(1, 9),
        'Gender.Other': ['Male', 'nonbinary', 'attack helicopter', 'she/they',
                         'genderfluid she/her', 'None of your business',
                         'asdfgh', None],
    })

    # Step A: build the review file.
    review = run(df, cfg)
    print('\n--- review file (correct the `label` column by hand) ---')
    print(review.to_string(index=False))

    # Step B: (in real use, STOP HERE, open the CSV, fix the labels, then
    # continue). Rows marked fallback/REVIEW* are sorted to the top.
    print('\n--- after apply_codebook() ---')

    # Creates a new <raw_col>_coded column.
    coded = apply_codebook(df, cfg)
    print(coded.to_string(index=False))

    # Or, replace the original free-text column:
    # coded = apply_codebook(df, cfg, overwrite=True)

if __name__ == '__main__':
    example()