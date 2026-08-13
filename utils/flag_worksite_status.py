"""
flag_worksite_status.py — tag administrative placeholders in the worksite
bridge table, and collapse the no-shows into a single site.

Input: worksite bridge, one row per (Year, worksite_id):
    Year, worksite_id, worksite_name, worksite_name_canonical,
    worksite_cluster_id, fuzzy_match_score

Outputs (to OUTDIR):
    worksite_bridge_status.csv     bridge + status, collapsed cluster
    worksite_status_matches.csv    every distinct name that matched
    worksite_status_review.csv     near misses that did NOT match
    worksite_status_conflicts.csv  clusters mixing placeholder and real sites
    worksite_status_summary.csv    records by Year x status

Added columns:
    worksite_status               no_show | exited | unplaced | test | active
    is_no_show                    bool
    is_placeholder                bool  (status != "active")
    worksite_cluster_id           collapsed: no-shows -> NO_SHOW_CLUSTER
    worksite_name_canonical       collapsed: no-shows -> NO_SHOW_LABEL
    worksite_cluster_id_pre       original cluster id, kept for audit
    worksite_name_canonical_pre   original canonical name, kept for audit

No rows are dropped and no keys change, so the output remerges onto the
analysis frame on (Year, worksite_id) exactly as the original bridge did.

Counts here are WORKSITE RECORDS, not youth. How many participants sit behind
each placeholder is only knowable after the remerge.
"""

from __future__ import annotations

import argparse
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/Users/sonia/Documents/SYEP")
BRIDGE = ROOT / "data/to_use/worksite_bridge.csv"
OUTDIR = ROOT / "data/to_use"

YEAR = "Year"
WS_ID = "worksite_id"
RAW = "worksite_name"
CANON = "worksite_name_canonical"
CLUSTER = "worksite_cluster_id"
SCORE = "fuzzy_match_score"

# Sentinel clusters for collapsed placeholders. Negative so they cannot
# collide with a real cluster id, and stay visible if they ever leak into a
# model. One per status, so collapsing two statuses does not merge them.
NO_SHOW_CLUSTER = -1
SENTINEL = {"no_show": -1, "exited": -2, "unplaced": -3, "test": -4}
SENTINEL_LABEL = {
    "no_show": "NO SHOW (administrative)",
    "exited": "EXITED (administrative)",
    "unplaced": "NOT PLACED (administrative)",
    "test": "TEST RECORD (administrative)",
}
NO_SHOW_LABEL = SENTINEL_LABEL["no_show"]

# Names are normalised (lowercased, punctuation -> space) before matching, so
# patterns need no punctuation classes. Order matters: the first status to
# match wins, so "NO SHOW - TERMINATED" lands in no_show, not exited.
STATUS_PATTERNS: dict[str, list[str]] = {
    "no_show": [
        r"\bno\s*show(s|ed|ing)?\b",
        r"\bnoshow(s|ed)?\b",
        r"\bno\s*call\s*no\s*show\b",
        r"\bdid\s*not\s*show\b",
        r"\bnever\s*show(ed)?\b",
        r"\bnever\s*reported\b",
    ],
    "exited": [
        r"\bterminat(ed|ion|e)\b",
        r"\bdeclin(ed|e)\b",
        r"\bwithdrew\b",
        r"\bwithdrawn\b",
        r"\bdropped?\s*out\b",
        r"\bcancell?ed\b",
        r"\bquit\b",
        r"\bdismissed\b",
        r"\bremoved\s*from\b",
    ],
    "unplaced": [
        r"\bnot\s*placed\b",
        r"\bno\s*placement\b",
        r"\bno\s*work\s*site\b",
        r"\bno\s*site\b",
        r"\bunassigned\b",
        r"\bunplaced\b",
        r"\bpending\b",
        r"\btbd\b",
        r"\bto\s*be\s*(determined|assigned)\b",
        r"\bplaceholder\b",
        r"\bunknown\b",
        r"^n\s*a$",
        r"^none$",
    ],
    "test": [
        r"^test\b",
        r"\btest\s*(site|worksite|record|entry|account)\b",
        r"\bdummy\b",
        r"\bdo\s*not\s*use\b",
        r"^x+$",
        r"^z+$",
    ],
}

STATUS_ORDER = ["no_show", "exited", "unplaced", "test"]

# Unmatched names get pulled into the review file if they trip any of these.
REVIEW_PATTERNS = [
    (r"show", "contains 'show'"),
    (r"^no\b", "starts with 'no'"),
    (r"\bnot\b", "contains 'not'"),
    (r"\bn\s*s\b", "possible 'NS' abbreviation"),
    (r"\bvoid\b|\binvalid\b|\berror\b|\bdelete|\bmisc\b|\bother\b",
     "administrative wording"),
    (r"^.{1,4}$", "very short name — possible code"),
    (r"^\d+$", "numeric name"),
]

_COMPILED = {s: [re.compile(p) for p in v] for s, v in STATUS_PATTERNS.items()}
_REVIEW = [(re.compile(p), why) for p, why in REVIEW_PATTERNS]


# --------------------------------------------------------------------------
# Matching
# --------------------------------------------------------------------------

def normalize(name: object) -> str:
    """Lowercase, strip accents, punctuation -> space, collapse whitespace."""
    if name is None or (isinstance(name, float) and np.isnan(name)):
        return ""
    s = unicodedata.normalize("NFKD", str(name))
    s = s.encode("ascii", "ignore").decode("ascii").lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def classify(*names: object) -> str:
    """Status for a site, checking every name variant it has. Any variant
    matching is enough: canonicalisation sometimes eats the marker, and
    sometimes only the canonical form carries it.
    """
    norms = [n for n in (normalize(x) for x in names) if n]
    for status in STATUS_ORDER:
        for pat in _COMPILED[status]:
            if any(pat.search(n) for n in norms):
                return status
    return "active"


def review_reason(*names: object) -> str:
    """Why an unmatched name is worth a human look. Empty if nothing trips."""
    norms = [n for n in (normalize(x) for x in names) if n]
    hits = [why for pat, why in _REVIEW if any(pat.search(n) for n in norms)]
    return "; ".join(dict.fromkeys(hits))


# --------------------------------------------------------------------------
# Build
# --------------------------------------------------------------------------

def flag(bridge: pd.DataFrame,
         collapse: tuple[str, ...] = ("no_show",)) -> pd.DataFrame:
    """Add status columns and collapse the requested statuses into the
    sentinel cluster. Originals are preserved in `*_pre` columns.
    """
    df = bridge.copy()

    df["worksite_status"] = [classify(r, c) for r, c in zip(df[RAW], df[CANON])]
    df["is_no_show"] = df["worksite_status"].eq("no_show")
    df["is_placeholder"] = df["worksite_status"].ne("active")

    df[f"{CLUSTER}_pre"] = df[CLUSTER]
    df[f"{CANON}_pre"] = df[CANON]

    is_str = pd.api.types.is_string_dtype(df[CLUSTER]) or df[CLUSTER].dtype == object

    for status in collapse:
        hit = df["worksite_status"].eq(status)
        df.loc[hit, CLUSTER] = str(SENTINEL[status]) if is_str else SENTINEL[status]
        df.loc[hit, CANON] = SENTINEL_LABEL[status]
    return df


def matched_names(df: pd.DataFrame) -> pd.DataFrame:
    """Every distinct name that matched, with how often and which years.
    This is the table to actually read before trusting the flag.
    """
    hit = df[df["is_placeholder"]]
    if hit.empty:
        return pd.DataFrame()
    return (
        hit.groupby([f"{CANON}_pre", "worksite_status"], dropna=False)
        .agg(records=(WS_ID, "size"),
             worksite_ids=(WS_ID, "nunique"),
             years=(YEAR, lambda s: ", ".join(map(str, sorted(s.unique())))),
             example_raw=(RAW, "first"))
        .reset_index()
        .sort_values("records", ascending=False)
    )


def review_candidates(df: pd.DataFrame) -> pd.DataFrame:
    """Names that did NOT match but look adjacent — false-negative hunting."""
    live = df[~df["is_placeholder"]].copy()
    live["review_reason"] = [
        review_reason(r, c) for r, c in zip(live[RAW], live[f"{CANON}_pre"])
    ]
    out = live[live["review_reason"].ne("")]
    if out.empty:
        return pd.DataFrame()
    cols = [YEAR, WS_ID, RAW, f"{CANON}_pre", f"{CLUSTER}_pre", "review_reason"]
    if SCORE in out.columns:
        cols.append(SCORE)
    return out[cols].sort_values(["review_reason", RAW])


def cluster_conflicts(df: pd.DataFrame, score_cut: float = 85.0) -> pd.DataFrame:
    """Clusters that held both placeholder and real sites before collapsing.

    This is the failure mode that matters. If the fuzzy matcher pulled a
    "NO SHOW" record into a real organisation's cluster, that org's placement
    count and return rate were absorbing no-shows. A high fuzzy_match_score on
    a placeholder row is the same warning from the other direction.
    """
    g = df.groupby(f"{CLUSTER}_pre", dropna=True).agg(
        n_records=(WS_ID, "size"),
        n_placeholder=("is_placeholder", "sum"),
        n_active=("is_placeholder", lambda s: int((~s).sum())),
        statuses=("worksite_status", lambda s: ", ".join(sorted(set(s)))),
        names=(f"{CANON}_pre", lambda s: " | ".join(sorted(set(map(str, s)))[:6])),
    )
    mixed = g[(g["n_placeholder"] > 0) & (g["n_active"] > 0)].reset_index()

    if SCORE in df.columns and not mixed.empty:
        absorbed = set(
            df.loc[df["is_placeholder"]
                   & pd.to_numeric(df[SCORE], errors="coerce").ge(score_cut),
                   f"{CLUSTER}_pre"]
        )
        mixed["high_score_placeholder"] = mixed[f"{CLUSTER}_pre"].isin(absorbed)

    return mixed.sort_values("n_records", ascending=False)


def summary(df: pd.DataFrame) -> pd.DataFrame:
    """Worksite records by Year x status, with a pooled block."""
    per_year = (df.groupby([YEAR, "worksite_status"], dropna=False)
                .size().rename("records").reset_index())
    per_year[YEAR] = per_year[YEAR].astype(str)
    pooled = (df.groupby("worksite_status", dropna=False)
              .size().rename("records").reset_index().assign(**{YEAR: "Pooled"}))
    out = pd.concat([per_year, pooled], ignore_index=True)
    out["% of records"] = (
        out["records"] / out.groupby(YEAR)["records"].transform("sum")).round(3)
    return out[[YEAR, "worksite_status", "records", "% of records"]]


# --------------------------------------------------------------------------

def main(bridge_path: Path = BRIDGE,
         outdir: Path = OUTDIR,
         collapse: tuple[str, ...] = ("no_show",),
         write: bool = True) -> pd.DataFrame:
    bridge = pd.read_csv(bridge_path)

    missing = [c for c in (YEAR, WS_ID, RAW, CANON, CLUSTER)
               if c not in bridge.columns]
    if missing:
        raise KeyError(f"bridge is missing {missing}; got {list(bridge.columns)}")

    df = flag(bridge, collapse=collapse)

    tables = {
        "worksite_bridge_status": df,
        "worksite_status_matches": matched_names(df),
        "worksite_status_review": review_candidates(df),
        "worksite_status_conflicts": cluster_conflicts(df),
        "worksite_status_summary": summary(df),
    }

    if write:
        outdir.mkdir(parents=True, exist_ok=True)
        for name, t in tables.items():
            path = outdir / f"{name}.csv"
            t.to_csv(path, index=False, encoding="utf-8-sig")
            print(f"wrote {path} ({len(t)} rows)")

    n_hit = int(df["is_placeholder"].sum())
    print(f"\n{n_hit} of {len(df)} worksite records flagged ({n_hit / len(df):.1%})")
    print(f"{df.loc[df['is_no_show'], WS_ID].nunique()} distinct worksite_ids "
          f"are no-shows, collapsed to cluster {NO_SHOW_CLUSTER}")
    print(f"clusters mixing placeholder and real sites: "
          f"{len(tables['worksite_status_conflicts'])}  <- review these")
    print(f"unmatched near-misses to review: {len(tables['worksite_status_review'])}")
    print()
    print(tables["worksite_status_summary"].to_string(index=False))

    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bridge", type=Path, default=BRIDGE)
    ap.add_argument("--outdir", type=Path, default=OUTDIR)
    ap.add_argument("--collapse", nargs="*", default=["no_show"],
                    choices=STATUS_ORDER,
                    help="statuses to fold into the sentinel cluster")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the summary without writing anything")
    a = ap.parse_args()
    main(a.bridge, a.outdir, tuple(a.collapse), write=not a.dry_run)