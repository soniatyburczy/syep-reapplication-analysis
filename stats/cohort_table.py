"""
cohort_table.py — descriptive tables for the reapplication analysis.

Writes to data/to_use/descriptives/:
    waterfall.csv               attrition from enrolled -> risk set
    cohort_table.csv            N, returns, rate + Wilson CI, by index year
    service_option_flow.csv     t x t+1 service option, returners only
    worksite_status_by_year.csv risk set share + return rate by status, by year
    worksite_distribution.csv   coverage + concentration of sites, by year
    worksite_size_bands.csv     sites and youth by site-size band, by year
    worksite_continuity.csv     sites appearing in 1/2/3 index years
    top_worksites.csv           largest sites, per-year N, return rate + CI
    return_by_<col>.csv         univariate return rate + CI, one per cut
    descriptives.md             all of the above, for prog.qmd / slides

Assumes an analysis frame with one row per (Participant.Unique.ID, Year)
already restricted to: enrolled, Older Youth, Community-Based, deduped,
age-eligible at t+1. Outcome `returned` = applied to any SYEP service
option in Year+1.

Worksite fields come from worksite_bridge_status.csv, so `worksite_status`
marks rows whose "worksite" is really an intake status (NO SHOW and friends).
Those rows stay in the risk set — they enrolled — but every site-level table
runs on `active_sites()` only. `is_no_show` is a covariate, not a site.

service_option is constant (Older Youth) in the index frame, so it is not
a composition column and not a subgroup cut. It reappears only in
service_option_flow, where the t+1 side can be Ladders for Leaders.

Return = reapplication
"""

from __future__ import annotations

from pathlib import Path
import sys
sys.path.insert(0, "/Users/sonia/Documents/SYEP")

import numpy as np
import pandas as pd

ROOT = Path("/Users/sonia/Documents/SYEP")
PANEL = ROOT / "data/to_use/analysis_frame_worksite.csv"
LOOKUP = ROOT / "data/to_use/merged.csv"
OUTDIR = ROOT / "data/to_use/descriptives"

ID = "Participant.Unique.ID"
INDEX_YEARS = [2022, 2023, 2024]

# Worksite identity, finest to coarsest. `worksite_name_y` is the raw string
# that came out of a merge with a suffix still attached; canonical is the
# cleaned form; cluster_id groups canonical names judged to be the same site.
WS_ID = "worksite_id"
WS_RAW = "worksite_name_y"
WS_CANON = "worksite_name_canonical"
WS_CLUSTER = "worksite_cluster_id"
WS_STATUS = "worksite_status"
WS_NO_SHOW = "is_no_show"

# Site-size bands, right-closed: 1, 2-5, 6-20, 21-50, 51+.
SIZE_BINS = [0, 1, 5, 20, 50, np.inf]
SIZE_LABELS = ["1", "2-5", "6-20", "21-50", "51+"]

_TABLES: list[tuple[str, pd.DataFrame]] = []


def save(df: pd.DataFrame, name: str, title: str | None = None,
         index: bool = False) -> pd.DataFrame:
    """Write one table to CSV and register it for the markdown dump."""
    OUTDIR.mkdir(parents=True, exist_ok=True)
    path = OUTDIR / f"{name}.csv"
    df.to_csv(path, index=index, encoding="utf-8-sig")
    _TABLES.append((title or name, df if not index else df.reset_index()))
    print(f"wrote {path} ({len(df)} rows)")
    return df


def write_markdown(name: str = "descriptives") -> None:
    """One file with every table, for pasting into prog.qmd or slides."""
    path = OUTDIR / f"{name}.md"
    parts = ["# Descriptive tables\n"]
    for title, df in _TABLES:
        parts.append(f"\n## {title}\n\n{df.to_markdown(index=False)}\n")
    path.write_text("\n".join(parts), encoding="utf-8")
    print(f"wrote {path} ({len(_TABLES)} tables)")


def _year_slices(df: pd.DataFrame,
                 index_years: list[int] = INDEX_YEARS
                 ) -> list[tuple[str, pd.DataFrame]]:
    """(label, subframe) for each index year, with a pooled row last."""
    return [(str(y), df[df["Year"] == y]) for y in index_years] \
         + [("Pooled", df[df["Year"].isin(index_years)])]


def active_sites(df: pd.DataFrame) -> pd.DataFrame:
    """Rows placed at a real worksite.

    Placeholder rows (no_show, exited, unplaced, test) are intake statuses
    written into the worksite name field. They stay in the risk set — those
    youth enrolled, and whether they reapply is the outcome — but they cannot
    be treated as sites, so every site-level table runs on this subset.
    """
    if WS_STATUS not in df.columns:
        return df
    return df[df[WS_STATUS].eq("active")]


def _have(df: pd.DataFrame, *cols: str, what: str = "tables") -> bool:
    """True if every column is present; otherwise print what is missing."""
    missing = [c for c in cols if c not in df.columns]
    if missing:
        print(f"  skipped {what}: not in frame -> {', '.join(missing)}")
    return not missing


# --------------------------------------------------------------------------
# Wilson score interval
# --------------------------------------------------------------------------

def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def build_outcome(panel: pd.DataFrame,
                  lookup: pd.DataFrame,
                  index_years: list[int] = INDEX_YEARS) -> pd.DataFrame:
    """Attach `returned` by looking (person, Year+1) up in the full
    all-options SYEP application frame.
    """
    applied = (
        lookup.loc[lookup["Year"].isin([y + 1 for y in index_years]),
                   [ID, "Year"]]
        .dropna(subset=[ID])
        .drop_duplicates()
        .assign(_applied=1)
        .rename(columns={"Year": "_next_year"})
    )

    out = panel[panel["Year"].isin(index_years)].copy()
    out["_next_year"] = out["Year"] + 1
    out = out.merge(applied, on=[ID, "_next_year"], how="left", validate="1:1")
    out["returned"] = out.pop("_applied").fillna(0).astype(int)
    return out.drop(columns="_next_year")


def cohort_table(df: pd.DataFrame,
                 outcome: str = "returned",
                 index_years: list[int] = INDEX_YEARS) -> pd.DataFrame:
    """N at risk, returns, rate + CI, and composition, by index year."""
    rows = []

    for label, sub in _year_slices(df, index_years):
        n = len(sub)
        k = int(sub[outcome].sum())
        lo, hi = wilson(k, n)

        age = pd.to_numeric(sub["age_on_start"], errors="coerce")
        hrs = pd.to_numeric(sub["total_hours_paid"], errors="coerce")

        rows.append({
            "Index year": label,
            "N at risk": n,
            "N returned": k,
            "Return rate": round(k / n, 3) if n else np.nan,
            "95% CI": f"[{lo:.3f}, {hi:.3f}]" if n else "",
            "Unique people": sub[ID].nunique(),
            "Mean age": round(age.mean(), 2),
            "% age 16-17": round(age.between(16, 17).mean(), 3),
            "% age 21+": round(age.ge(21).mean(), 3),
            "Median hours": round(hrs.median(), 1),
            "% at cap": round(hrs.ge(150).mean(), 3),
            "% zero hours": round(hrs.eq(0).mean(), 3),
            "% no show": (round(float(sub[WS_NO_SHOW].mean()), 3)
                          if WS_NO_SHOW in sub.columns else np.nan),
            "N worksites": active_sites(sub)[WS_CLUSTER].nunique(),
            "% at real worksite": (round(float(sub[WS_STATUS].eq("active").mean()), 3)
                                   if WS_STATUS in sub.columns else np.nan),
            "% cluster known": round(active_sites(sub)[WS_CLUSTER].notna().mean(), 3),
            "% provider known": round(sub["provider"].notna().mean(), 3),
        })

    return pd.DataFrame(rows)


def by_subgroup(df: pd.DataFrame,
                col: str,
                outcome: str = "returned",
                min_n: int = 30) -> pd.DataFrame:
    """Return rate + Wilson CI within levels of `col`, pooled across index
    years. Feeds the univariate panels of the descriptives figure. 
    """
    rows = []
    for level, sub in df.groupby(col, dropna=False, observed=True):
        n, k = len(sub), int(sub[outcome].sum())
        lo, hi = wilson(k, n)
        rows.append({
            col: level,
            "n": n,
            "returned": k,
            "rate": round(k / n, 3) if n else np.nan,
            "ci_lo": round(lo, 3),
            "ci_hi": round(hi, 3),
            "thin": n < min_n,
        })
    return pd.DataFrame(rows).sort_values("rate", ascending=False)


def service_option_flow(df: pd.DataFrame,
                        lookup: pd.DataFrame,
                        index_years: list[int] = INDEX_YEARS) -> pd.DataFrame:
    """t service option x t+1 service option, among returners only.
    """
    nxt = (
        lookup.loc[lookup["Year"].isin([y + 1 for y in index_years]),
                   [ID, "Year", "service_option"]]
        .dropna(subset=[ID])
        .sort_values("service_option")
        .drop_duplicates(subset=[ID, "Year"], keep="first")
        .rename(columns={"Year": "_next_year",
                         "service_option": "service_option_next"})
    )

    ret = df[df["returned"].eq(1)].copy()
    ret["_next_year"] = ret["Year"] + 1
    ret = ret.merge(nxt, on=[ID, "_next_year"], how="left")

    return pd.crosstab(ret["service_option"],
                       ret["service_option_next"],
                       dropna=False)


# --------------------------------------------------------------------------
# Worksite distribution
# --------------------------------------------------------------------------

def attach_worksite_size(df: pd.DataFrame,
                         within_year: bool = True) -> pd.DataFrame:
    """Add `worksite_n_youth` (size of this row's cluster) and
    `worksite_size_band`.

    Sized within the index year by default, so a site that took 40 youth in
    2022 and 5 in 2024 is not reported as one 45-youth site. Rows with a null
    cluster get NaN in both columns rather than being folded into a band.
    """
    keys = ["Year", WS_CLUSTER] if within_year else [WS_CLUSTER]
    live = active_sites(df)
    size = live.groupby(keys, dropna=True)[WS_CLUSTER].transform("size")
    # placeholders and unknown clusters get NaN, not a spurious band
    size = size.reindex(df.index)
    df["worksite_n_youth"] = size
    df["worksite_size_band"] = pd.cut(size, bins=SIZE_BINS, labels=SIZE_LABELS)
    return df


def placeholder_table(df: pd.DataFrame,
                      index_years: list[int] = INDEX_YEARS,
                      outcome: str = "returned") -> pd.DataFrame:
    """Share of the risk set and return rate by worksite status, per year.

    This is the substantive reason to do the cleaning, not just a QC table.
    "No show" is a participant-level fact — enrolled, never appeared — that
    was only recoverable from the worksite name field. If its return rate
    separates from the placed youth, it belongs in the model as a covariate,
    not as a site.
    """
    if WS_STATUS not in df.columns:
        return pd.DataFrame()

    rows = []
    for label, sub in _year_slices(df, index_years):
        total = len(sub)
        for status, grp in sub.groupby(WS_STATUS, dropna=False, observed=True):
            n, k = len(grp), int(grp[outcome].sum())
            lo, hi = wilson(k, n)
            rows.append({
                "Index year": label,
                "Worksite status": status,
                "N": n,
                "% of risk set": round(n / total, 3) if total else np.nan,
                "N returned": k,
                "Return rate": round(k / n, 3) if n else np.nan,
                "ci_lo": round(lo, 3),
                "ci_hi": round(hi, 3),
                "thin": n < 30,
            })

    return pd.DataFrame(rows)


def worksite_distribution(df: pd.DataFrame,
                          index_years: list[int] = INDEX_YEARS,
                          top: int = 10) -> pd.DataFrame:
    """Coverage and concentration of worksites, by index year.

    Placeholder rows are counted (`% no show`, `% placeholder`) and then
    dropped: every column from `N at real sites` onward describes youth placed
    at an actual site. Mixing the two makes the no-show pseudo-site the largest
    "worksite" in the data and wrecks every concentration statistic.

    All three identity levels are reported side by side so the effect of
    canonicalisation is visible: raw string -> canonical name -> cluster.
    "Names per cluster" above 1 means the clustering is collapsing spelling
    variants; "Effective N sites" is 1/HHI, the number of equal-sized sites
    that would give the same concentration.
    """
    rows = []

    for label, full in _year_slices(df, index_years):
        sub = active_sites(full)
        sizes = sub[WS_CLUSTER].value_counts()
        n_sites = int(sizes.size)
        placed = int(sizes.sum())
        shares = sizes / placed if placed else sizes

        rows.append({
            "Index year": label,
            "N in risk set": len(full),
            "% no show": (round(float(full[WS_NO_SHOW].mean()), 3)
                          if WS_NO_SHOW in full.columns else np.nan),
            "% placeholder": (round(float(full[WS_STATUS].ne("active").mean()), 3)
                              if WS_STATUS in full.columns else np.nan),
            "N at real sites": len(sub),
            "% worksite_id known": round(sub[WS_ID].notna().mean(), 3),
            "% raw name known": round(sub[WS_RAW].notna().mean(), 3),
            "% canonical known": round(sub[WS_CANON].notna().mean(), 3),
            "% cluster known": round(sub[WS_CLUSTER].notna().mean(), 3),
            "Distinct ids": sub[WS_ID].nunique(),
            "Distinct raw names": sub[WS_RAW].nunique(),
            "Distinct canonical": sub[WS_CANON].nunique(),
            "Distinct clusters": n_sites,
            "Names per cluster": (round(sub[WS_CANON].nunique() / n_sites, 2)
                                  if n_sites else np.nan),
            "Median youth/site": float(sizes.median()) if n_sites else np.nan,
            "Mean youth/site": round(float(sizes.mean()), 1) if n_sites else np.nan,
            "P90 youth/site": float(sizes.quantile(0.9)) if n_sites else np.nan,
            "Max youth/site": int(sizes.max()) if n_sites else 0,
            "% sites singleton": (round(float(sizes.eq(1).mean()), 3)
                                  if n_sites else np.nan),
            "% youth at singletons": (round(int(sizes[sizes.eq(1)].sum()) / placed, 3)
                                      if placed else np.nan),
            f"% youth in top {top}": (round(float(shares.nlargest(top).sum()), 3)
                                      if placed else np.nan),
            "Effective N sites": (round(1 / float((shares ** 2).sum()), 1)
                                  if placed else np.nan),
        })

    return pd.DataFrame(rows)


def worksite_size_bands(df: pd.DataFrame,
                        index_years: list[int] = INDEX_YEARS) -> pd.DataFrame:
    """Long table: how many sites, and how many youth, sit in each size band
    per index year. Long rather than wide so it plots straight as stacked or
    grouped bars. Denominators are placements with a known cluster.
    """
    rows = []

    for label, full in _year_slices(df, index_years):
        sub = active_sites(full)
        sizes = sub[WS_CLUSTER].value_counts()
        if sizes.empty:
            continue
        band = pd.cut(sizes, bins=SIZE_BINS, labels=SIZE_LABELS)
        n_sites, n_youth = int(sizes.size), int(sizes.sum())

        for b in SIZE_LABELS:
            mask = (band == b).to_numpy()
            youth = int(sizes[mask].sum())
            rows.append({
                "Index year": label,
                "Site size": b,
                "Worksites": int(mask.sum()),
                "% of worksites": round(int(mask.sum()) / n_sites, 3),
                "Youth": youth,
                "% of youth": round(youth / n_youth, 3),
            })

    return pd.DataFrame(rows)


def worksite_continuity(df: pd.DataFrame,
                        index_years: list[int] = INDEX_YEARS) -> pd.DataFrame:
    """How many clusters appear in 1, 2, ... of the index years, and what
    share of placements they absorb.

    Relevant to the model, not just the deck: a site seen in a single year
    cannot support a site-level effect estimated across years.
    """
    d = active_sites(df[df["Year"].isin(index_years)]).dropna(subset=[WS_CLUSTER])
    years_seen = d.groupby(WS_CLUSTER)["Year"].nunique()
    placements = d[WS_CLUSTER].value_counts()

    rows = []
    for k in range(1, len(index_years) + 1):
        ids = years_seen.index[years_seen.eq(k)]
        youth = int(placements.reindex(ids).fillna(0).sum())
        rows.append({
            "Years present": k,
            "Worksites": len(ids),
            "% of worksites": (round(len(ids) / len(years_seen), 3)
                               if len(years_seen) else np.nan),
            "Youth": youth,
            "% of youth": round(youth / len(d), 3) if len(d) else np.nan,
        })

    return pd.DataFrame(rows)


def top_worksites(df: pd.DataFrame,
                  index_years: list[int] = INDEX_YEARS,
                  n_top: int = 20,
                  outcome: str = "returned",
                  min_n: int = 30) -> pd.DataFrame:
    """Largest clusters pooled across index years, with per-year headcount and
    the return rate of the youth placed there. Cluster label is the modal
    canonical name. `thin` flags sites too small to read the rate off.
    """
    d = active_sites(df[df["Year"].isin(index_years)]).dropna(subset=[WS_CLUSTER])
    if d.empty:
        return pd.DataFrame()

    counts = pd.crosstab(d[WS_CLUSTER], d["Year"])
    counts.columns = [f"N {c}" for c in counts.columns]

    agg = d.groupby(WS_CLUSTER).agg(
        worksite=(WS_CANON, lambda s: s.mode().iat[0] if not s.mode().empty else ""),
        names=(WS_CANON, "nunique"),
        n=(outcome, "size"),
        returned=(outcome, "sum"),
    )

    out = agg.join(counts).sort_values("n", ascending=False).head(n_top)
    ci = [wilson(int(k), int(n)) for k, n in zip(out["returned"], out["n"])]
    out["rate"] = (out["returned"] / out["n"]).round(3)
    out["ci_lo"] = [round(lo, 3) for lo, _ in ci]
    out["ci_hi"] = [round(hi, 3) for _, hi in ci]
    out["thin"] = out["n"] < min_n
    return out.reset_index()


def waterfall(steps: list[tuple[str, int]]) -> pd.DataFrame:
    """Attrition table. Pass (label, row_count) in the order the filters
    actually ran, e.g.:

        waterfall([
            ("Enrolled, 2022-2024",            len(enrolled)),
            ("Older Youth",                    len(oy)),
            ("Deduplicated (person, year)",    len(dedup)),
            ("Non-null Participant.Unique.ID", len(has_id)),
            ("Age-eligible at t+1",            len(risk_set)),
        ])
    """
    t = pd.DataFrame(steps, columns=["Step", "Rows"])
    t["Dropped"] = t["Rows"].shift().sub(t["Rows"]).fillna(0).astype(int)
    t["% of start"] = (t["Rows"] / t["Rows"].iloc[0]).round(3)
    return t


def main(panel_path: Path = PANEL, tag: str = "") -> pd.DataFrame:
    """Build and save every descriptive table.
    """
    panel = pd.read_csv(panel_path)
    lookup = pd.read_csv(LOOKUP, usecols=[ID, "Year", "service_option"])

    df = build_outcome(panel, lookup)

    save(cohort_table(df), f"cohort_table{tag}",
         "Cohort table, by index year")

    save(service_option_flow(df, lookup), f"service_option_flow{tag}",
         "Service option at t vs t+1, returners only", index=True)

    hours = df["total_hours_paid"]
    df["hours_band"] = pd.cut(
        hours,
        bins=[-0.001, 0, 25, 75, 125, 149.99, np.inf],
        labels=["0", "1-25", "26-75", "76-125", "126-149", "150 (cap)"],
    )

    if _have(df, WS_STATUS, what="placeholder table"):
        save(placeholder_table(df), f"worksite_status_by_year{tag}",
             "Risk set share and return rate by worksite status, by year")

    if _have(df, WS_ID, WS_RAW, WS_CANON, WS_CLUSTER, what="worksite tables"):
        df = attach_worksite_size(df)

        save(worksite_distribution(df), f"worksite_distribution{tag}",
             "Worksite distribution, by index year")

        save(worksite_size_bands(df), f"worksite_size_bands{tag}",
             "Worksites and youth by site size band, by index year")

        save(worksite_continuity(df), f"worksite_continuity{tag}",
             "Worksites by number of index years present")

        save(top_worksites(df), f"top_worksites{tag}",
             "Largest worksites, pooled, with return rate")

    # worksite_status is covered per-year by placeholder_table above; only
    # the binary no-show cut is repeated here, as a model-facing covariate.
    for col in ["borough", "age_on_start", "hours_band", "provider",
                WS_NO_SHOW, "worksite_size_band"]:
        if col not in df.columns:
            print(f"  skipped {col}: not in frame")
            continue
        save(by_subgroup(df, col), f"return_by_{col}{tag}",
             f"Return rate by {col}")

    write_markdown(f"descriptives{tag}")
    return df


if __name__ == "__main__":
    main()