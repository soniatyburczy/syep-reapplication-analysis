from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Keys carried through every derived frame
# --------------------------------------------------------------------------

KEYS: List[str] = ["Participant.Unique.ID", "Year", "SurveyType"]


# --------------------------------------------------------------------------
# Text normalization
# --------------------------------------------------------------------------

def _norm(s: str) -> str:
    """Normalize a column name or label for robust matching.

    Handles the usual survey-export noise: non-breaking spaces (U+00A0),
    smart quotes, doubled spaces, trailing whitespace, case.
    """
    if s is None:
        return ""
    s = unicodedata.normalize("NFKC", str(s))   # NBSP -> plain space
    s = s.replace("\u2019", "'").replace("\u2018", "'")
    s = s.replace("\u201c", '"').replace("\u201d", '"')
    s = re.sub(r"\s+", " ", s)
    return s.strip().casefold()


# --------------------------------------------------------------------------
# Value coercion
# --------------------------------------------------------------------------

_TRUE_TOKENS = {"1", "1.0", "true", "t", "yes", "y", "x", "checked", "selected"}
_FALSE_TOKENS = {"0", "0.0", "false", "f", "no", "n", "unchecked", "not selected", ""}


def coerce_checkbox(series: pd.Series, label: Optional[str] = None) -> pd.Series:
    """Coerce one checkbox column to float 0/1/NaN.

    `label` is the selection text; some exports store the label itself in the
    cell when checked, so we treat a match on it as True.
    """
    label_norm = _norm(label) if label else None

    def _one(v):
        if pd.isna(v):
            return np.nan
        if isinstance(v, (bool, np.bool_)):
            return float(v)
        if isinstance(v, (int, float, np.integer, np.floating)):
            if v in (0, 1):
                return float(v)
            return np.nan
        t = _norm(v)
        if t in _TRUE_TOKENS:
            return 1.0
        if t in _FALSE_TOKENS:
            return 0.0
        if label_norm and t == label_norm:
            return 1.0
        return np.nan

    return series.map(_one).astype(float)


def coerce_likert(series: pd.Series, mapping: Optional[Dict[str, float]] = None) -> pd.Series:
    """Coerce a Likert column to numeric.

    If the column is already numeric (e.g. a `|weight` column), pass through.
    Otherwise map response text via `mapping`, defaulting to a 4-point
    agreement scale. Unmapped values become NaN.
    """
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(float)

    default = {
        "strongly disagree": 1.0,
        "disagree": 2.0,
        "agree": 3.0,
        "strongly agree": 4.0,
        "somewhat disagree": 2.0,
        "somewhat agree": 3.0,
        "neither agree nor disagree": np.nan,
        "not applicable": np.nan,
        "n/a": np.nan,
    }
    m = {_norm(k): v for k, v in (mapping or default).items()}
    return series.map(lambda v: np.nan if pd.isna(v) else m.get(_norm(v), np.nan)).astype(float)


# --------------------------------------------------------------------------
# Block specification
# --------------------------------------------------------------------------

@dataclass
class BlockSpec:
    """Declarative description of one checkbox battery.

    Parameters
    ----------
    prefix
        Short name for output columns, e.g. "cf" -> cf_answered, cf_n_selected.
    stem
        The question text preceding the " - " separator. Matched on a
        normalized prefix basis, so minor whitespace drift is tolerated.
    nota_label
        Label of the "None of the above" item, if present. Used for the
        answered flag; excluded from counts and groups automatically.
    groups
        Mapping of group name -> list of selection labels. Produces a binary
        "any of these" indicator per group.
    singles
        Mapping of output name -> single selection label. Produces a binary
        indicator for that one item.
    exclude_from_count
        Labels to leave out of `<prefix>_n_selected` (beyond NOTA).
    separator
        String between stem and label. Default " - ".
    """

    prefix: str
    stem: str
    nota_label: Optional[str] = "None of the above"
    groups: Dict[str, Sequence[str]] = field(default_factory=dict)
    singles: Dict[str, str] = field(default_factory=dict)
    exclude_from_count: Sequence[str] = field(default_factory=tuple)
    separator: str = " - "


# --------------------------------------------------------------------------
# Column discovery
# --------------------------------------------------------------------------

def discover_items(df: pd.DataFrame, spec: BlockSpec) -> Dict[str, str]:
    """Return {selection_label: actual_column_name} for one block.

    Matches any column whose normalized text starts with the normalized
    stem + separator. The returned label is the ORIGINAL (un-normalized)
    text after the separator, so it stays human-readable.
    """
    key = _norm(spec.stem + spec.separator)
    found: Dict[str, str] = {}
    sep_norm = _norm(spec.separator)

    for col in df.columns:
        n = _norm(col)
        if not n.startswith(key):
            continue
        # Recover the label from the original string, not the normalized one.
        raw = unicodedata.normalize("NFKC", str(col))
        idx = raw.find(spec.separator)
        label = raw[idx + len(spec.separator):].strip() if idx >= 0 else raw
        if not label:
            continue
        found[label] = col

    if not found and sep_norm:  # fall back: stem match without separator
        for col in df.columns:
            if _norm(col).startswith(_norm(spec.stem)):
                found[str(col)] = col
    return found


def _resolve(labels: Sequence[str], items: Dict[str, str], where: str) -> List[str]:
    """Map declared labels to actual columns, raising on typos."""
    lookup = {_norm(k): v for k, v in items.items()}
    out, missing = [], []
    for lab in labels:
        col = lookup.get(_norm(lab))
        (out.append(col) if col else missing.append(lab))
    if missing:
        raise KeyError(
            f"{where}: could not match {missing!r}. "
            f"Available labels: {sorted(items.keys())!r}"
        )
    return out


# --------------------------------------------------------------------------
# Auditing (run this BEFORE building features)
# --------------------------------------------------------------------------

def audit_block(df: pd.DataFrame, spec: BlockSpec, by: str = "Year") -> pd.DataFrame:
    """Report per-item coverage and coercion problems for one block.

    Returns a frame with, per item: how many rows are non-null, how many
    coerced to 1, and how many values failed coercion (these are the ones
    that would silently become NaN).
    """
    items = discover_items(df, spec)
    if not items:
        raise KeyError(f"[{spec.prefix}] no columns matched stem: {spec.stem!r}")

    rows = []
    for label, col in items.items():
        raw = df[col]
        coerced = coerce_checkbox(raw, label)
        bad = raw.notna() & coerced.isna()
        rows.append({
            "prefix": spec.prefix,
            "label": label,
            "column": col,
            "n_nonnull": int(raw.notna().sum()),
            "n_selected": int((coerced == 1).sum()),
            "n_uncoercible": int(bad.sum()),
            "example_bad": raw[bad].iloc[0] if bad.any() else None,
        })
    out = pd.DataFrame(rows)

    if by in df.columns:
        # Per-group presence: is this block asked in every year / survey type?
        pres = (
            df[list(items.values())].notna().any(axis=1)
            .groupby(df[by]).mean().rename("frac_rows_with_any_item")
        )
        print(f"[{spec.prefix}] block presence by {by}:")
        print(pres.to_string(), "\n")

    return out


def block_year_matrix(df: pd.DataFrame, specs: Sequence[BlockSpec],
                      by: Sequence[str] = ("Year", "SurveyType")) -> pd.DataFrame:
    """Which blocks exist in which year / survey type.

    Run this first. It tells you what is actually buildable before you
    invest in any of the rest.
    """
    by = [c for c in by if c in df.columns]
    out = {}
    for spec in specs:
        items = discover_items(df, spec)
        if not items:
            out[spec.prefix] = pd.Series(dtype=float)
            continue
        present = df[list(items.values())].notna().any(axis=1)
        out[spec.prefix] = present.groupby([df[c] for c in by]).mean()
    return pd.DataFrame(out).round(3)


# --------------------------------------------------------------------------
# Feature construction
# --------------------------------------------------------------------------

def build_block(df: pd.DataFrame, spec: BlockSpec,
                keys: Sequence[str] = KEYS) -> pd.DataFrame:
    """Build derived features for one block. Returns keys + features."""
    missing_keys = [k for k in keys if k not in df.columns]
    if missing_keys:
        raise KeyError(f"missing key columns: {missing_keys}")

    items = discover_items(df, spec)
    if not items:
        raise KeyError(f"[{spec.prefix}] no columns matched stem: {spec.stem!r}")

    coerced = pd.DataFrame(
        {label: coerce_checkbox(df[col], label) for label, col in items.items()},
        index=df.index,
    )

    nota_cols = []
    if spec.nota_label:
        hit = {_norm(k): k for k in items}.get(_norm(spec.nota_label))
        if hit:
            nota_cols = [hit]

    substantive = [c for c in coerced.columns if c not in nota_cols]
    excl = {_norm(x) for x in spec.exclude_from_count}
    count_cols = [c for c in substantive if _norm(c) not in excl]

    out = df[list(keys)].copy()

    # Answered = any item non-null AND (something selected OR NOTA selected).
    any_nonnull = coerced.notna().any(axis=1)
    any_selected = (coerced[substantive].fillna(0).sum(axis=1) > 0)
    nota_selected = (coerced[nota_cols].fillna(0).sum(axis=1) > 0) if nota_cols else False
    answered = any_nonnull & (any_selected | nota_selected)
    out[f"{spec.prefix}_answered"] = answered

    out[f"{spec.prefix}_n_selected"] = coerced[count_cols].fillna(0).sum(axis=1)

    for gname, labels in spec.groups.items():
        cols = _resolve(labels, items, f"[{spec.prefix}] group {gname!r}")
        col_labels = [lab for lab, c in items.items() if c in cols]
        out[f"{spec.prefix}_{gname}"] = coerced[col_labels].fillna(0).max(axis=1)

    for sname, label in spec.singles.items():
        (col,) = _resolve([label], items, f"[{spec.prefix}] single {sname!r}")
        col_label = next(lab for lab, c in items.items() if c == col)
        out[f"{spec.prefix}_{sname}"] = coerced[col_label].fillna(0)

    if nota_cols:
        out[f"{spec.prefix}_nota"] = coerced[nota_cols].fillna(0).max(axis=1)

    # Mask everything derived where the block wasn't answered.
    derived = [c for c in out.columns
               if c not in keys and c != f"{spec.prefix}_answered"]
    out.loc[~answered, derived] = np.nan

    return out


def build_block_frame(df: pd.DataFrame, specs: Sequence[BlockSpec],
                      keys: Sequence[str] = KEYS,
                      validate_unique: bool = True) -> pd.DataFrame:
    """Build all blocks and merge into a single feature frame."""
    if validate_unique:
        dup = df.duplicated(subset=list(keys)).sum()
        if dup:
            raise ValueError(
                f"{dup} duplicate rows on {list(keys)}. Resolve before building "
                "features, or the merge will fan out."
            )

    frame = df[list(keys)].copy()
    for spec in specs:
        block = build_block(df, spec, keys)
        frame = frame.merge(block, on=list(keys), how="left", validate="1:1")
    return frame


def add_likert_scale(df: pd.DataFrame, out: pd.DataFrame, columns: Sequence[str],
                     name: str, min_items: int = 2,
                     mapping: Optional[Dict[str, float]] = None) -> pd.DataFrame:
    """Add a mean-score scale from Likert items, plus its item count.

    Requires at least `min_items` non-missing responses; otherwise NaN.
    Check Cronbach's alpha (see `cronbach_alpha`) before trusting the mean.
    """
    num = pd.DataFrame({c: coerce_likert(df[c], mapping) for c in columns}, index=df.index)
    n_ok = num.notna().sum(axis=1)
    out = out.copy()
    out[f"{name}_mean"] = num.mean(axis=1).where(n_ok >= min_items)
    out[f"{name}_n_items"] = n_ok
    return out


def cronbach_alpha(df: pd.DataFrame, columns: Sequence[str],
                   mapping: Optional[Dict[str, float]] = None) -> float:
    """Cronbach's alpha over Likert items, complete cases only."""
    num = pd.DataFrame({c: coerce_likert(df[c], mapping) for c in columns}).dropna()
    k = num.shape[1]
    if k < 2 or num.empty:
        return np.nan
    item_var = num.var(axis=0, ddof=1).sum()
    total_var = num.sum(axis=1).var(ddof=1)
    return float((k / (k - 1)) * (1 - item_var / total_var))


# --------------------------------------------------------------------------
# Example specs -- edit labels to match your actual export
# --------------------------------------------------------------------------

COUNTERFACTUAL = BlockSpec(
    prefix="cf",
    stem="What would you have done this summer if you had not been in SYEP? [Check all that apply]",
    groups={
        "paid_alternative": [
            "A different part-time job",
            "A different full-time job",
            "Paid summer internship",
        ],
        "unpaid_activity": [
            "Volunteered",
            "Unpaid summer internship",
            "Summer camp",
            "Summer school",
        ],
        "leisure_only": [
            "Spending time with friends",
            "Spending time with family",
        ],
    },
    singles={"no_plans": "I had no other summer plans"},
    exclude_from_count=["I had no other summer plans"],
)

MOTIVATIONS = BlockSpec(
    prefix="motiv",
    stem="Besides earning money, what were other top reasons you wanted to participate in SYEP? [Check up to THREE responses]",
    groups={
        "career": [
            "Learn more about career options",
            "Find a summer job or internship",
            "Learn new skills",
            "Receive work readiness training",
        ],
        "social": ["Meet new people while participating in the programming"],
        "structure": ["Have safe and productive activity this summer"],
    },
    singles={"credit": "To earn course credit"},
)

ADULTS = BlockSpec(
    prefix="adults",
    stem="Which statements do you agree with regarding the adults you met in SYEP? [Check all that apply]",
    groups={
        "basic_respect": [
            "They made me feel welcome",
            "They treated me with respect",
            "They were knowledgeable",
        ],
        "developmental": [
            "They included someone I would consider a mentor",
            "They made me aware of jobs I could consider",
            "They encouraged me to believe in myself",
            "They valued my opinions and concerns",
        ],
    },
)

BENEFITS = BlockSpec(
    prefix="benefit",
    stem="Besides earning money, what were the five biggest benefits you received from SYEP this past summer? [Select up to 5]",
    groups={
        "career_clarity": [
            "Identified careers I am interested in",
            "Developed a career plan with clear next steps",
        ],
        "job_readiness": [
            "Understood how to look for and get a job",
            "Understood what employers are looking for",
            "Developed a resume or other job application materials",
            "Understood how to interact with professionals",
        ],
        "education": [
            "Decided to go to college",
            "Made plans to enroll in a college or training program",
            "Earned course credit",
        ],
    },
    singles={"mentor_relationship": "Developed relationships with mentors/supervisors"},
)

ALL_SPECS = [COUNTERFACTUAL, MOTIVATIONS, ADULTS, BENEFITS]


if __name__ == "__main__":
    print(__doc__)