"""
plots_o_plenty.py — slide figures for the reapplication descriptives.

Writes PNG + PDF to data/plots/descriptives/:

    fig_dose_response   return rate by paid hours and by age, with CIs
    fig_funnel          site-level return rate vs site size, binomial limits
    fig_concentration   Lorenz curve of youth across worksites
    fig_univariate      every return_by_*.csv as one dot-and-CI panel
    fig_year_trend      return rate by index year, all rows vs matched only

Reads the CSVs written by cohort_table.py, and imports that module for the
site-level frames the CSVs do not carry. Matplotlib only, no seaborn.

Run after cohort_table.py:
    python figures.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cohort_table as ct

DESC = ct.OUTDIR
FIGS = ct.ROOT / "plots/descriptives"
INDEX_YEARS = ct.INDEX_YEARS

BLUE, ORANGE, GREY, DARK = "#2f6db3", "#e08214", "#9aa5b1", "#22303f"

try:
    plt.style.use("seaborn-v0_8-whitegrid")
except OSError:
    plt.style.use("ggplot")

plt.rcParams.update({
    "figure.dpi": 130,
    "savefig.dpi": 200,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "axes.edgecolor": "#c8ced6",
    "legend.frameon": False,
})


def _pct(ax, axis: str = "y") -> None:
    fmt = mticker.FuncFormatter(lambda v, _: f"{v:.0%}")
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)


def _finish(fig, name: str, note: str = "") -> Path:
    """Strip chart junk, add the source note, write PNG + PDF."""
    for ax in fig.axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    if note:
        fig.text(0.005, 0.005, note, fontsize=8, color=GREY, ha="left")
    FIGS.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIGS / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {FIGS / name}.png")
    return FIGS / f"{name}.png"


def _load_frame() -> pd.DataFrame:
    """Rebuild the analysis frame with the outcome and worksite size."""
    panel = pd.read_csv(ct.PANEL)
    lookup = pd.read_csv(ct.LOOKUP, usecols=[ct.ID, "Year", "service_option"])
    df = ct.build_outcome(panel, lookup)
    return ct.attach_worksite_size(df)

# --------------------------------------------------------------------------
# 1. Dose response: hours and age
# --------------------------------------------------------------------------

def fig_dose_response(desc: Path = DESC):
    hrs = pd.read_csv(desc / "return_by_hours_band.csv")
    age = pd.read_csv(desc / "return_by_age_on_start.csv")

    order = ["0", "1-25", "26-75", "76-125", "126-149", "150 (cap)"]
    hrs["hours_band"] = pd.Categorical(hrs["hours_band"], order, ordered=True)
    hrs = hrs.sort_values("hours_band")
    age = age.sort_values("age_on_start")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.6))

    x = np.arange(len(hrs))
    a1.vlines(x, hrs["ci_lo"], hrs["ci_hi"], color=BLUE, lw=2, alpha=.55)
    a1.plot(x, hrs["rate"], "o-", color=BLUE, lw=2, ms=7)
    a1.set_xticks(x, hrs["hours_band"].astype(str))
    a1.set_title("Paid hours track reapplication")
    a1.set_xlabel("Paid hours")
    a1.set_ylabel("Reapplied the next year")

    for xi, r, n in zip(x, hrs["rate"], hrs["n"]):
        a1.annotate(f"n={n:,}", (xi, r), textcoords="offset points",
                    xytext=(0, -16), ha="center", fontsize=8, color=GREY)

    a2.vlines(age["age_on_start"], age["ci_lo"], age["ci_hi"],
              color=ORANGE, lw=2, alpha=.55)
    a2.plot(age["age_on_start"], age["rate"], "o-", color=ORANGE, lw=2, ms=7)
    a2.set_title("Reapplication falls with age")
    a2.set_xlabel("Age at program start")
    a2.set_ylabel("")

    for ax in (a1, a2):
        _pct(ax)
        ax.set_ylim(0, 0.85)

    return _finish(fig, "fig_dose_response",
                   "Bars are 95% Wilson intervals. Pooled index years "
                   f"{min(INDEX_YEARS)}-{max(INDEX_YEARS)}.")


# --------------------------------------------------------------------------
# 2. Funnel plot: is there a worksite effect at all?
# --------------------------------------------------------------------------

def funnel_data(df: pd.DataFrame, index_years=INDEX_YEARS) -> pd.DataFrame:
    """Per-cluster n and return rate, real worksites only, pooled."""
    d = ct.active_sites(df[df["Year"].isin(index_years)]).dropna(
        subset=[ct.WS_CLUSTER])
    g = d.groupby(ct.WS_CLUSTER).agg(n=("returned", "size"),
                                     k=("returned", "sum"))
    g["rate"] = g["k"] / g["n"]
    return g.reset_index()


def fig_funnel(df: pd.DataFrame, index_years=INDEX_YEARS, min_n: int = 5,
               label_top: int = 6):
    """Site return rate against site size, with binomial control limits.
    """
    g = funnel_data(df, index_years)
    g = g[g["n"] >= min_n]
    p = g["k"].sum() / g["n"].sum()

    grid = np.arange(min_n, max(g["n"].max(), min_n + 1) + 1)
    se = np.sqrt(p * (1 - p) / grid)

    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    for z, alpha, lab in [(1.96, .18, "95%"), (3.09, .12, "99.8%")]:
        ax.fill_between(grid, np.clip(p - z * se, 0, 1),
                        np.clip(p + z * se, 0, 1),
                        color=BLUE, alpha=alpha, lw=0,
                        label=f"{lab} limits")
    ax.axhline(p, color=DARK, lw=1.4, ls="--",
               label=f"pooled rate {p:.1%}")

    out = (g["rate"] > p + 3.09 * np.sqrt(p * (1 - p) / g["n"])) | \
          (g["rate"] < p - 3.09 * np.sqrt(p * (1 - p) / g["n"]))
    ax.scatter(g.loc[~out, "n"], g.loc[~out, "rate"], s=14, color=GREY,
               alpha=.45, lw=0, label="within limits")
    ax.scatter(g.loc[out, "n"], g.loc[out, "rate"], s=26, color=ORANGE,
               alpha=.85, lw=0, label="outside 99.8%")

    for _, r in g[out].nlargest(label_top, "n").iterrows():
        ax.annotate(f"{int(r['n'])}", (r["n"], r["rate"]),
                    textcoords="offset points", xytext=(4, 4),
                    fontsize=8, color=DARK)

    ax.set_xscale("log")
    ax.set_xlabel("Youth placed at the worksite (log scale)")
    ax.set_ylabel("Reapplied the next year")
    ax.set_ylim(0, 1)
    _pct(ax)
    ax.legend(loc="lower right", ncol=2)

    n_out, expected = int(out.sum()), 0.002 * len(g)
    ax.set_title(
        f"{n_out} of {len(g):,} worksites sit outside the 99.8% limits "
        f"({expected:.1f} expected if site identity did not matter)"
        if n_out else
        f"No worksite sits outside the 99.8% limits: site-level differences "
        f"are consistent with sampling noise alone", fontsize=12)

    return _finish(fig, "fig_funnel",
                   f"Real worksites only, n>={min_n} youth, pooled "
                   f"{min(index_years)}-{max(index_years)}. "
                   f"{len(g):,} worksites.")


# --------------------------------------------------------------------------
# 3. Concentration
# --------------------------------------------------------------------------

def fig_concentration(df: pd.DataFrame, index_years=INDEX_YEARS):
    """Lorenz curve: what share of youth sit at what share of worksites."""
    fig, ax = plt.subplots(figsize=(6.4, 5.4))

    for year, color in zip(index_years, [GREY, ORANGE, BLUE]):
        sizes = np.sort(funnel_data(df, [year])["n"].to_numpy())
        if not sizes.size:
            continue
        cx = np.insert(np.arange(1, sizes.size + 1) / sizes.size, 0, 0)
        cy = np.insert(np.cumsum(sizes) / sizes.sum(), 0, 0)
        gini = 1 - 2 * np.trapezoid(cy, cx)
        ax.plot(cx, cy, lw=2, color=color, label=f"{year}  (Gini {gini:.2f})")

    ax.plot([0, 1], [0, 1], ls=":", lw=1.2, color=DARK)
    ax.annotate("equal placement", (.55, .58), rotation=38, fontsize=9,
                color=GREY)
    ax.set_xlabel("Share of worksites, smallest first")
    ax.set_ylabel("Share of youth placed")
    ax.set_title("A small minority of worksites absorb most placements")
    _pct(ax); _pct(ax, "x")
    ax.legend(loc="upper left")

    return _finish(fig, "fig_concentration", "Real worksites only.")


# --------------------------------------------------------------------------
# 4. Every univariate cut as one figure
# --------------------------------------------------------------------------

def fig_univariate(desc: Path = DESC, cuts: list[str] | None = None,
                   drop_missing: bool = True, max_levels: int = 12):
    """Dot-and-interval panel per return_by_*.csv. Thin cells are hollow."""
    files = ([desc / f"return_by_{c}.csv" for c in cuts] if cuts
             else sorted(desc.glob("return_by_*.csv")))
    files = [f for f in files if f.exists()]
    if not files:
        print("  no return_by_*.csv found"); return None

    ncol = min(3, len(files))
    nrow = -(-len(files) // ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 3.6 * nrow))
    axes = np.atleast_1d(axes).ravel()

    for ax, f in zip(axes, files):
        t = pd.read_csv(f)
        col = t.columns[0]
        # astype(str) leaves pd.NA intact on pandas 3 string dtype, so the
        # missing level has to be materialised before it can be filtered.
        t[col] = t[col].astype("string").fillna("(missing)").astype(str)
        if drop_missing:
            t = t[~t[col].str.lower().isin(["nan", "none", "(missing)", ""])]
        if t.empty:
            ax.set_visible(False)
            continue
        t = t.nlargest(max_levels, "n").sort_values("rate")
        y = np.arange(len(t))
        thin = t["thin"].fillna(False).to_numpy(dtype=bool)

        ax.hlines(y, t["ci_lo"], t["ci_hi"], color=BLUE, lw=2, alpha=.5)
        ax.scatter(t.loc[~thin, "rate"], y[~thin], color=BLUE, s=34, zorder=3)
        ax.scatter(t.loc[thin, "rate"], y[thin], facecolors="white",
                   edgecolors=BLUE, s=34, zorder=3)
        ax.set_yticks(y, [s[:26] for s in t[col]], fontsize=9)
        ax.set_title(col.replace("_", " "))
        _pct(ax, "x")
        ax.set_xlim(0, 1)

    for ax in axes[len(files):]:
        ax.set_visible(False)

    fig.suptitle("Reapplication rate by participant and placement characteristics",
                 fontsize=14, fontweight="bold", y=1.01)
    return _finish(fig, "fig_univariate",
                   "95% Wilson intervals. Hollow points are cells under n=30. "
                   f"Top {max_levels} levels by n.")


# --------------------------------------------------------------------------
# 5. Year trend, with and without the unmatched rows
# --------------------------------------------------------------------------

def fig_year_trend(desc: Path = DESC):
    """Overall rate by index year against the matched-worksite subset.
    """
    cohort = pd.read_csv(desc / "cohort_table.csv")
    cohort = cohort[cohort["Index year"] != "Pooled"].copy()
    cohort["year"] = cohort["Index year"].astype(int)

    lo_hi = cohort["95% CI"].str.strip("[]").str.split(",", expand=True)
    cohort["lo"] = lo_hi[0].astype(float)
    cohort["hi"] = lo_hi[1].astype(float)

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.errorbar(cohort["year"], cohort["Return rate"],
                yerr=[cohort["Return rate"] - cohort["lo"],
                      cohort["hi"] - cohort["Return rate"]],
                fmt="o-", color=BLUE, lw=2, ms=8, capsize=4,
                label="all youth at risk")

    status = desc / "worksite_status_by_year.csv"
    if status.exists():
        s = pd.read_csv(status)
        s = s[(s["Worksite status"] == "active") & (s["Index year"] != "Pooled")]
        s = s.assign(year=s["Index year"].astype(int))
        ax.errorbar(s["year"], s["Return rate"],
                    yerr=[s["Return rate"] - s["ci_lo"],
                          s["ci_hi"] - s["Return rate"]],
                    fmt="s--", color=ORANGE, lw=2, ms=7, capsize=4,
                    label="matched to a real worksite")

    ax.set_xticks(sorted(cohort["year"]))
    ax.set_xlabel("Index year")
    ax.set_ylabel("Reapplied the next year")
    ax.set_title("Reapplication rose across index years")
    _pct(ax)
    ax.legend(loc="lower right")

    return _finish(fig, "fig_year_trend",
                   "95% Wilson intervals. Worksite coverage differs by year, "
                   "so the two series cover different denominators.")


def main() -> None:
    fig_dose_response()
    fig_univariate()
    fig_year_trend()

    df = _load_frame()
    fig_funnel(df)
    fig_concentration(df)


if __name__ == "__main__":
    main()