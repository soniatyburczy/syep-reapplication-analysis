from __future__ import annotations

import importlib.util
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.container import BarContainer


# ---------------------------------------------------------------------
# Import lg.py so plots use the exact Frame A / Frame B construction
# ---------------------------------------------------------------------

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

_LG = ROOT / "models" / "lg.py"
_spec = importlib.util.spec_from_file_location("lg", _LG)
lg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lg)

PLOTDIR = ROOT / "plots" / "descriptives"
PLOTDIR.mkdir(parents=True, exist_ok=True)

OUTDIR = ROOT / "data" / "models"
OUTDIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

# Bar charts always start at zero. The question is the TOP of the axis.
#
# False (default): every reapplication-rate plot shares one upper limit,
#   computed as the next multiple of 10 above the highest confidence
#   bound across all rate plots. Panels stay mutually comparable without
#   crushing the bars into the bottom of an empty 0-100 canvas.
#
# True: pin every rate axis to a literal 0-100. Use this if a reviewer
#   specifically wants "percent of a whole" framing everywhere.
FORCE_FULL_SCALE = False

# Age cells below this n are drawn hollow and flagged in the footnote.
MIN_CELL_N = 30

# Survey bars at or above this percent carry their value label inside
# the bar in white; shorter bars get it outside, past the CI cap.
LABEL_INSIDE_MIN = 18.0

SAVE_FORMATS = ("png",)  # add "pdf" or "svg" for vector slides

Z = 1.96  # 95% normal quantile for Wilson intervals


# ---------------------------------------------------------------------
# Palette and typography
# ---------------------------------------------------------------------

INK = "#1B1F24"
SUBTLE = "#5C6670"
GRID = "#DCE1E6"

ADMIN = "#2F5B7C"     # blue
SURVEY = "#C2703D"    # orange
MUTED = "#9AA6B1"

FRAME_COLORS = {
    "Administrative": ADMIN,
    "Survey": SURVEY,
}

sns.set_theme(style="white", context="talk")

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "text.color": INK,
    "axes.labelcolor": SUBTLE,
    "axes.edgecolor": GRID,
    "axes.labelsize": 13,
    "xtick.color": SUBTLE,
    "ytick.color": SUBTLE,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "legend.frameon": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
})


def savefig(name: str) -> None:
    stem = Path(name).stem

    for ext in SAVE_FORMATS:
        plt.savefig(
            PLOTDIR / f"{stem}.{ext}",
            bbox_inches="tight",
            facecolor="white",
        )

    plt.close()


# ---------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------

def _barplot(**kwargs):
    """sns.barplot with error bars disabled, across seaborn versions."""
    try:
        return sns.barplot(errorbar=None, **kwargs)
    except TypeError:
        return sns.barplot(ci=None, **kwargs)


def titles(ax, title: str, subtitle: str | None = None) -> None:
    ax.set_title(
        title,
        loc="left",
        fontsize=17,
        color=INK,
        pad=30 if subtitle else 14,
    )

    if subtitle:
        ax.annotate(
            subtitle,
            xy=(0, 1),
            xycoords="axes fraction",
            xytext=(0, 10),
            textcoords="offset points",
            ha="left",
            va="bottom",
            fontsize=12.5,
            color=SUBTLE,
        )


def footnote(fig, text: str) -> None:
    fig.text(
        0.0,
        -0.02,
        text,
        ha="left",
        va="top",
        fontsize=10.5,
        color=SUBTLE,
    )


def value_grid(ax, orient: str = "v") -> None:
    """Gridlines on the value axis only; drop the category-side spine."""
    ax.grid(False)

    if orient == "v":
        ax.yaxis.grid(True)
        ax.set_axisbelow(True)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
    else:
        ax.xaxis.grid(True)
        ax.set_axisbelow(True)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)


def pct_axis(ax, orient: str = "v", limit: float | None = None) -> None:
    axis = ax.yaxis if orient == "v" else ax.xaxis
    axis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))

    if limit is not None:
        if orient == "v":
            ax.set_ylim(0, limit)
        else:
            ax.set_xlim(0, limit)


def label_bars(ax, fmt: str = "{:.0f}%", fontsize: float = 11.5) -> None:
    """Label bar ends. Skips ErrorbarContainers, which have no
    datavalues attribute."""
    for container in ax.containers:
        if not isinstance(container, BarContainer):
            continue

        labels = [
            "" if not np.isfinite(v) else fmt.format(v)
            for v in container.datavalues
        ]

        ax.bar_label(
            container,
            labels=labels,
            padding=4,
            fontsize=fontsize,
            color=SUBTLE,
        )


def label_bars_h_inside(ax, values, ci_low, ci_high,
                        inside_min: float = LABEL_INSIDE_MIN,
                        fontsize: float = 11.5,
                        whisker_pad: float = 1.5) -> None:
    """Horizontal-bar labels that never collide with CI whiskers.

    Long bars carry the label inside in white, positioned to the LEFT of
    the lower CI bound. Short bars place the label outside to the RIGHT of
    the upper CI bound. This clears the entire whisker rather than only its
    outer cap.
    """
    for y, (pct, lo, hi) in enumerate(zip(values, ci_low, ci_high)):
        if not np.isfinite(pct):
            continue

        inside = pct >= inside_min and np.isfinite(lo)

        x = (float(lo) - whisker_pad) if inside else (float(hi) + whisker_pad)

        ax.text(
            x,
            y,
            f"{pct:.1f}%",
            va="center",
            ha="right" if inside else "left",
            fontsize=fontsize,
            color="white" if inside else SUBTLE,
            fontweight="medium" if inside else "normal",
            zorder=6,
        )


def headroom(ax, orient: str = "v", frac: float = 0.10) -> None:
    """Leave room for value labels without clipping."""
    if orient == "v":
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo, hi + (hi - lo) * frac)
    else:
        lo, hi = ax.get_xlim()
        ax.set_xlim(lo, hi + (hi - lo) * frac)


def wrap_labels(labels, width: int = 32):
    return ["\n".join(textwrap.wrap(str(x), width)) for x in labels]


# ---------------------------------------------------------------------
# Rate estimation
# ---------------------------------------------------------------------

def wilson_ci(k, n, z: float = Z):
    """Wilson score interval. Behaves sanely at small n and p near 0/1."""
    k = np.asarray(k, dtype=float)
    n = np.asarray(n, dtype=float)

    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(n > 0, k / n, np.nan)
        denom = 1 + z**2 / n
        centre = (p + z**2 / (2 * n)) / denom
        half = (
            z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
        )

    return centre - half, centre + half


def rate_table(df: pd.DataFrame, by: str, outcome: str = "returned",
               order=None) -> pd.DataFrame:
    """Cell-level rate with n and a 95% Wilson interval, in percent."""
    tab = (
        df
        .groupby(by, observed=True)[outcome]
        .agg(n="size", k="sum")
        .reset_index()
    )

    if order is not None:
        tab = (
            tab.set_index(by)
            .reindex(order)
            .rename_axis(by)
            .reset_index()
        )

    tab["rate"] = tab["k"] / tab["n"]

    lo, hi = wilson_ci(tab["k"], tab["n"])

    tab["ci_low"] = lo
    tab["ci_high"] = hi

    for col in ("rate", "ci_low", "ci_high"):
        tab[f"{col}_pct"] = 100 * tab[col]

    return tab


def shared_rate_limit(*tabs) -> float:
    """Common upper bound across rate plots."""
    if FORCE_FULL_SCALE:
        return 100.0

    hi = np.nanmax([t["ci_high_pct"].max() for t in tabs])

    return float(min(100.0, np.ceil((hi + 4) / 10.0) * 10.0))


def draw_ci(ax, x, tab, orient: str = "v") -> None:
    err_lo = np.clip(tab["rate_pct"] - tab["ci_low_pct"], 0, None)
    err_hi = np.clip(tab["ci_high_pct"] - tab["rate_pct"], 0, None)

    ax.errorbar(
        x=x,
        y=tab["rate_pct"],
        yerr=np.vstack([err_lo, err_hi]),
        fmt="none",
        ecolor=INK,
        elinewidth=1.4,
        capsize=4,
        alpha=0.55,
        zorder=5,
    )


def cell_ticks(values, ns) -> list[str]:
    return [
        f"{v}\nn = {int(n):,}" if np.isfinite(n) else f"{v}\n—"
        for v, n in zip(values, ns)
    ]


# ---------------------------------------------------------------------
# Build EXACT analysis frames used by lg.py
# ---------------------------------------------------------------------

def build_frames():
    dfA = lg.prep()
    dfB = lg.prep_survey(dfA)
    dfA_survey = lg.survey_admin_subsample(dfA, dfB)

    print(f"Frame A person-years: {len(dfA):,}")
    print(f"Frame A unique people: {dfA[lg.ID].nunique():,}")

    print(f"\nFrame B rows: {len(dfB):,}")
    print(f"Frame B unique people: {dfB[lg.ID].nunique():,}")

    print(f"\nA survey-key subsample: {len(dfA_survey):,}")

    return dfA, dfB, dfA_survey


# ---------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------

def frame_summary(dfA, dfB):
    rows = []

    for name, df in [
        ("Administrative frame (A)", dfA),
        ("Survey frame (B)", dfB),
    ]:
        rows.append({
            "frame": name,
            "rows": len(df),
            "unique_participants": df[lg.ID].nunique(),
            "year_min": df["Year"].min(),
            "year_max": df["Year"].max(),
            "mean_age": df["age_on_start"].mean(),
            "median_age": df["age_on_start"].median(),
            "mean_hours": df["total_hours_paid"].mean(),
            "median_hours": df["total_hours_paid"].median(),
            "return_rate": df["returned"].mean(),
            "no_show_rate": df["is_no_show"].mean(),
            "worksite_matched_rate": df["worksite_matched"].mean(),
        })

    out = pd.DataFrame(rows)

    out.to_csv(
        OUTDIR / "descriptive_frame_summary.csv",
        index=False,
    )

    print("\n--- Frame summary ---")
    print(out.to_string(index=False))

    return out


# ---------------------------------------------------------------------
# 1. AGE DISTRIBUTION
# ---------------------------------------------------------------------

def plot_age_distribution(dfA, dfB):
    """Percent-of-frame, not counts: the frames differ enormously in n,
    so raw counts make the survey frame look like a rounding error."""
    parts = []

    for label, df in [("Administrative", dfA), ("Survey", dfB)]:
        age = (
            pd.to_numeric(df["age_on_start"], errors="coerce")
            .round()
            .dropna()
            .astype(int)
        )

        tab = (
            age.value_counts(normalize=True)
            .mul(100)
            .rename("percent")
            .rename_axis("age")
            .reset_index()
        )

        tab["frame"] = label
        parts.append(tab)

    plot_df = (
        pd.concat(parts, ignore_index=True)
        .sort_values(["frame", "age"])
    )

    # Pull the youngest-age gap out of the data so the subtitle cannot
    # drift out of sync with the plot.
    youngest = int(plot_df["age"].min())

    def share(frame: str) -> float:
        row = plot_df[
            plot_df["frame"].eq(frame) & plot_df["age"].eq(youngest)
        ]
        return float(row["percent"].iloc[0]) if len(row) else np.nan

    fig, ax = plt.subplots(figsize=(12, 6.5))

    _barplot(
        data=plot_df,
        x="age",
        y="percent",
        hue="frame",
        palette=FRAME_COLORS,
        ax=ax,
        saturation=1.0,
    )

    ax.set_xlabel("Age at program start")
    ax.set_ylabel("Percent of frame")

    pct_axis(ax, "v")
    value_grid(ax, "v")
    headroom(ax, "v", 0.08)

    ax.legend(title=None, loc="upper right")

    titles(
        ax,
        "Survey respondents skew young",
        f"Age {youngest} is {share('Survey'):.1f}% of the survey frame "
        f"vs {share('Administrative'):.1f}% of administrative person-years",
    )

    footnote(
        fig,
        f"Administrative n = {len(dfA):,} person-years   |   "
        f"Survey n = {len(dfB):,} participants",
    )

    savefig("age_distribution_frames.png")


# ---------------------------------------------------------------------
# 2. BOROUGH DISTRIBUTION
# ---------------------------------------------------------------------

def plot_borough_distribution(dfA, dfB):
    parts = []

    for label, df in [("Administrative", dfA), ("Survey", dfB)]:
        tab = (
            df["borough"]
            .fillna("Unknown")
            .value_counts(normalize=True, dropna=False)
            .mul(100)
            .rename("percent")
            .rename_axis("borough")
            .reset_index()
        )

        tab["frame"] = label
        parts.append(tab)

    plot_df = pd.concat(parts, ignore_index=True)

    # Rank by administrative share so the reader has one stable ordering.
    order = (
        plot_df[plot_df["frame"].eq("Administrative")]
        .sort_values("percent", ascending=False)["borough"]
        .tolist()
    )

    extra = [b for b in plot_df["borough"].unique() if b not in order]
    order = order + extra

    fig, ax = plt.subplots(figsize=(11.5, 6.5))

    _barplot(
        data=plot_df,
        y="borough",
        x="percent",
        hue="frame",
        order=order,
        palette=FRAME_COLORS,
        ax=ax,
        saturation=1.0,
    )

    ax.set_xlabel("Percent of frame")
    ax.set_ylabel("")

    pct_axis(ax, "h")
    value_grid(ax, "h")
    label_bars(ax, "{:.1f}%", fontsize=10.5)
    headroom(ax, "h", 0.12)

    ax.legend(title=None, loc="lower right")

    titles(
        ax,
        "Borough composition: administrative vs survey frame",
        "Ordered by administrative share; gaps indicate survey selection",
    )

    savefig("borough_distribution_frames.png")


# ---------------------------------------------------------------------
# 3. HOURS BAND DISTRIBUTION
# ---------------------------------------------------------------------

def plot_hours_distribution(dfA):
    tab = (
        dfA["hours_band"]
        .value_counts(normalize=True, sort=False)
        .mul(100)
        .reindex(lg.HOURS_LABELS)
        .rename("percent")
        .rename_axis("hours_band")
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(11, 6.5))

    _barplot(
        data=tab,
        x="hours_band",
        y="percent",
        color=ADMIN,
        ax=ax,
        saturation=1.0,
    )

    ax.set_xlabel("Hours worked")
    ax.set_ylabel("Percent of person-years")

    pct_axis(ax, "v")
    value_grid(ax, "v")
    label_bars(ax, "{:.1f}%")
    headroom(ax, "v")

    titles(
        ax,
        "Distribution of hours worked",
        f"{len(dfA):,} person-years, administrative frame",
    )

    savefig("hours_band_distribution.png")


# ---------------------------------------------------------------------
# 4-6. REAPPLICATION RATES (shared vertical scale)
# ---------------------------------------------------------------------

def plot_return_by_hours(dfA, tab, limit):
    tab.to_csv(OUTDIR / "descriptive_return_by_hours.csv", index=False)

    fig, ax = plt.subplots(figsize=(11.5, 6.5))

    _barplot(
        data=tab,
        x="hours_band",
        y="rate_pct",
        color=ADMIN,
        ax=ax,
        saturation=1.0,
    )

    draw_ci(ax, np.arange(len(tab)), tab)

    ax.set_xlabel("Hours worked")
    ax.set_ylabel("Reapplication rate")

    ax.set_xticks(np.arange(len(tab)))
    ax.set_xticklabels(
        cell_ticks(tab["hours_band"], tab["n"]),
        fontsize=11,
    )

    pct_axis(ax, "v", limit)
    value_grid(ax, "v")
    label_bars(ax, "{:.1f}%")

    titles(
        ax,
        "Unadjusted reapplication rate by hours worked",
    )

    savefig("return_rate_by_hours.png")


def plot_return_by_year(dfA, tab, limit):
    tab.to_csv(OUTDIR / "descriptive_return_by_year.csv", index=False)

    fig, ax = plt.subplots(figsize=(11, 6.5))

    ax.fill_between(
        tab["Year"],
        tab["ci_low_pct"],
        tab["ci_high_pct"],
        color=ADMIN,
        alpha=0.14,
        linewidth=0,
    )

    ax.plot(
        tab["Year"],
        tab["rate_pct"],
        marker="o",
        markersize=9,
        linewidth=2.6,
        color=ADMIN,
        markerfacecolor="white",
        markeredgewidth=2.4,
        zorder=4,
    )

    for _, row in tab.iterrows():
        ax.annotate(
            f"{row['rate_pct']:.1f}%",
            xy=(row["Year"], row["rate_pct"]),
            xytext=(0, 14),
            textcoords="offset points",
            ha="center",
            fontsize=11.5,
            color=SUBTLE,
        )

    ax.set_xlabel("Index year")
    ax.set_ylabel("Reapplication rate")

    ax.set_xticks(tab["Year"])
    ax.set_xticklabels(
        cell_ticks(tab["Year"].astype(int), tab["n"]),
        fontsize=11,
    )

    # Same zero-based scale as the bar panels, so the three are comparable.
    pct_axis(ax, "v", limit)
    value_grid(ax, "v")

    titles(
        ax,
        "Reapplication rate by index year",
        "Shaded band is the 95% Wilson interval",
    )

    savefig("return_rate_by_year.png")


def plot_return_by_age(dfA, tab, limit):
    tab.to_csv(OUTDIR / "descriptive_return_by_age.csv", index=False)

    thin = tab["n"] < MIN_CELL_N

    fig, ax = plt.subplots(figsize=(11, 6.5))

    ax.fill_between(
        tab["age"],
        tab["ci_low_pct"],
        tab["ci_high_pct"],
        color=ADMIN,
        alpha=0.14,
        linewidth=0,
    )

    ax.plot(
        tab["age"],
        tab["rate_pct"],
        linewidth=2.6,
        color=ADMIN,
        zorder=3,
    )

    # Well-populated cells: filled. Thin cells: hollow, so nobody reads
    # a 12-person cell as a finding.
    ax.scatter(
        tab.loc[~thin, "age"],
        tab.loc[~thin, "rate_pct"],
        s=70,
        color=ADMIN,
        zorder=4,
        label=f"n \u2265 {MIN_CELL_N}",
    )

    ax.scatter(
        tab.loc[thin, "age"],
        tab.loc[thin, "rate_pct"],
        s=70,
        facecolors="white",
        edgecolors=ADMIN,
        linewidths=2.2,
        zorder=4,
        label=f"n < {MIN_CELL_N}",
    )

    # Age 24 is structurally important in age_analysis.py: participants
    # cannot return after ageing out.
    ax.axvline(23.5, linestyle="--", linewidth=1.6, color=MUTED, zorder=1)

    ax.annotate(
        "Age-out boundary",
        xy=(23.5, limit),
        xytext=(-8, -14),
        textcoords="offset points",
        ha="right",
        va="top",
        fontsize=11,
        color=SUBTLE,
    )

    ax.set_xlabel("Age at program start")
    ax.set_ylabel("Reapplication rate")
    ax.set_xticks(tab["age"])

    pct_axis(ax, "v", limit)
    value_grid(ax, "v")

    if thin.any():
        ax.legend(loc="upper right", title=None)

    titles(
        ax,
        "Unadjusted reapplication rate by age",
        "Shaded band is the 95% Wilson interval",
    )

    footnote(
        fig,
        "Eligibility to return is mechanically constrained near the "
        "age-out boundary; rates there are not behavioural.",
    )

    savefig("return_rate_by_age.png")


def return_rate_plots(dfA):
    """Build all three rate tables first so they can share an axis."""
    hours = rate_table(dfA, "hours_band", order=lg.HOURS_LABELS)

    year = rate_table(dfA, "Year").sort_values("Year")

    temp = dfA.copy()
    temp["age"] = (
        pd.to_numeric(temp["age_on_start"], errors="coerce")
        .round()
        .astype("Int64")
    )

    age = (
        rate_table(temp.dropna(subset=["age"]), "age")
        .sort_values("age")
    )
    age["age"] = age["age"].astype(int)

    limit = shared_rate_limit(hours, year, age)

    print(f"\nShared reapplication-rate axis: 0-{limit:.0f}%")

    plot_return_by_hours(dfA, hours, limit)
    plot_return_by_year(dfA, year, limit)
    plot_return_by_age(dfA, age, limit)

    return hours, year, age


# ---------------------------------------------------------------------
# 7. SURVEY ITEMS
# ---------------------------------------------------------------------

# Labels are kept short enough to sit on one line at the default wrap
# width. The plot title carries the sentence stem ("What participants
# would have done otherwise"), so the items need not repeat it.
SURVEY_LABELS = {
    # Counterfactuals
    "cf_paid_alternative":
        "Another paid activity",
    "cf_unpaid_activity":
        "An unpaid activity",
    "cf_leisure_only":
        "Leisure time only",
    "cf_no_plans":
        "No alternative plans",

    # Expectations
    "expect_app_clarity":
        "Application process was clear",
    "expect_first_choice":
        "Received first-choice placement",
    "expect_knew_what_to_expect":
        "Knew what to expect",

    # Benefits
    "benefit_career_clarity":
        "Greater career clarity",
    "benefit_job_readiness":
        "Improved job readiness",
    "benefit_self_efficacy":
        "Improved confidence",
    "benefit_mentor_relationship":
        "Built a mentor relationship",
    "benefit_money_management":
        "Improved money management",
}


def survey_block_table(dfB, block_name):
    items = lg.BLOCKS[block_name]
    gate = f"{block_name}_answered"

    # Restrict descriptive percentages to people who actually answered
    # this block.
    answered = dfB.loc[dfB[gate].eq(1)]

    rows = []

    for item in items:
        k = int(answered[item].sum())
        n = len(answered)

        lo, hi = wilson_ci(k, n)

        rows.append({
            "block": block_name,
            "item": item,
            "label": SURVEY_LABELS.get(item, item),
            "n_block_answered": n,
            "n_selected": k,
            "percent_selected": 100 * k / n if n else np.nan,
            "ci_low": 100 * float(lo),
            "ci_high": 100 * float(hi),
            "block_response_rate": 100 * dfB[gate].mean(),
        })

    return pd.DataFrame(rows)


def plot_survey_block(dfB, block_name, title, filename, limit=None):
    tab = survey_block_table(dfB, block_name)

    # Largest at the top.
    tab = tab.sort_values("percent_selected", ascending=False)

    n_answered = int(tab["n_block_answered"].iloc[0])
    response_rate = tab["block_response_rate"].iloc[0]

    height = 2.6 + 0.72 * len(tab)

    fig, ax = plt.subplots(figsize=(12, height))

    _barplot(
        data=tab,
        y="label",
        x="percent_selected",
        color=SURVEY,
        ax=ax,
        saturation=1.0,
    )

    err_lo = np.clip(tab["percent_selected"] - tab["ci_low"], 0, None)
    err_hi = np.clip(tab["ci_high"] - tab["percent_selected"], 0, None)

    ax.errorbar(
        x=tab["percent_selected"],
        y=np.arange(len(tab)),
        xerr=np.vstack([err_lo, err_hi]),
        fmt="none",
        ecolor=INK,
        elinewidth=1.4,
        capsize=4,
        alpha=0.55,
        zorder=5,
    )

    ax.set_yticks(np.arange(len(tab)))
    ax.set_yticklabels(wrap_labels(tab["label"]), fontsize=12)

    ax.set_xlabel("Percent selecting response")
    ax.set_ylabel("")

    # Survey items are shares of a fixed denominator and can plausibly
    # approach 100, so these DO get the full scale.
    pct_axis(ax, "h", limit if limit is not None else 100)
    value_grid(ax, "h")

    # Labels sit inside the bar, or outside past the CI cap for short
    # bars, so they never land on top of a whisker.
    label_bars_h_inside(
        ax,
        tab["percent_selected"],
        tab["ci_low"],
        tab["ci_high"],
        fontsize=11.5,
    )

    titles(
        ax,
        title,
        f"Among the {n_answered:,} participants who answered this block "
        f"({response_rate:.1f}% of the survey frame)",
    )

    savefig(filename)

    return tab


# ---------------------------------------------------------------------
# 7b. SURVEY ITEMS — combined two-panel figure
# ---------------------------------------------------------------------
#
# Replaces the separate per-block slides with one figure. Two stacked
# panels share a single 0-100% x-axis, so items are directly comparable
# across blocks, but the blocks stay visually separate because they have
# different denominators and different meanings: `cf` is what someone
# would have done instead of SYEP, `benefit` is what they say they got
# out of it. Putting all nine bars on one continuous axis with no break
# would invite reading "83.6% job readiness" against "60.5% another paid
# activity" as if they were the same quantity.
#
# Panel heights are proportional to item count so a bar is the same
# physical height in both panels; unequal panels make the block with
# fewer items look inflated.

PANEL_BLOCKS = [
    ("cf", "Counterfactual — what they would have done instead"),
    ("benefit", "Reported benefits — what they say they got"),
]


def plot_survey_panels(
    dfB,
    blocks=PANEL_BLOCKS,
    title="What respondents say about the program",
    subtitle=None,
    filename="survey_items_panels.png",
    limit=100,
    bar_height=0.62,
):
    tables = []

    for block, _ in blocks:
        tab = survey_block_table(dfB, block)
        tab = tab.sort_values("percent_selected", ascending=False)
        tables.append(tab)

    counts = [len(t) for t in tables]

    # 1.9" of fixed chrome (titles, axis, footnote) plus the bars.
    height = 1.9 + bar_height * sum(counts)

    fig, axes = plt.subplots(
        len(tables),
        1,
        figsize=(13, height),
        sharex=True,
        gridspec_kw={"height_ratios": counts, "hspace": 0.30},
    )

    axes = np.atleast_1d(axes)

    for ax, tab, (block, header) in zip(axes, tables, blocks):
        _barplot(
            data=tab,
            y="label",
            x="percent_selected",
            color=SURVEY,
            ax=ax,
            saturation=1.0,
        )

        err_lo = np.clip(tab["percent_selected"] - tab["ci_low"], 0, None)
        err_hi = np.clip(tab["ci_high"] - tab["percent_selected"], 0, None)

        ax.errorbar(
            x=tab["percent_selected"],
            y=np.arange(len(tab)),
            xerr=np.vstack([err_lo, err_hi]),
            fmt="none",
            ecolor=INK,
            elinewidth=1.4,
            capsize=4,
            alpha=0.55,
            zorder=5,
        )

        ax.set_yticks(np.arange(len(tab)))
        ax.set_yticklabels(wrap_labels(tab["label"]), fontsize=12)
        ax.set_ylabel("")

        pct_axis(ax, "h", limit)
        value_grid(ax, "h")

        label_bars_h_inside(
            ax,
            tab["percent_selected"],
            tab["ci_low"],
            tab["ci_high"],
            fontsize=11.5,
        )

        # Each block has its own denominator, so it is stated on the
        # block rather than once for the figure.
        n_answered = int(tab["n_block_answered"].iloc[0])
        rate = float(tab["block_response_rate"].iloc[0])

        ax.annotate(
            f"{header}  ·  n = {n_answered:,} answered "
            f"({rate:.1f}% of the survey frame)",
            xy=(0, 1),
            xycoords="axes fraction",
            xytext=(0, 9),
            textcoords="offset points",
            ha="left",
            va="bottom",
            fontsize=12,
            color=INK,
            fontweight="medium",
        )

        # The shared axis draws a spine between the panels; without this
        # the two blocks read as one grid with a stray rule through it.
        ax.spines["bottom"].set_visible(ax is axes[-1])

    axes[-1].set_xlabel("Percent selecting response")

    # Three stacked lines above the top panel: title, optional subtitle,
    # then that panel's own block header at +9pt (set in the loop).
    axes[0].set_title(
        title,
        loc="left",
        fontsize=17,
        color=INK,
        pad=58 if subtitle else 36,
    )

    if subtitle:
        axes[0].annotate(
            subtitle,
            xy=(0, 1),
            xycoords="axes fraction",
            xytext=(0, 36),
            textcoords="offset points",
            ha="left",
            va="bottom",
            fontsize=12.5,
            color=SUBTLE,
        )

    footnote(
        fig,
        "Items are not mutually exclusive; a respondent may select any "
        "number within a block. Bars are 95% Wilson intervals.",
    )

    savefig(filename)

    return pd.concat(tables, ignore_index=True)


def survey_descriptives(dfB):
    """Two-panel slide figure for cf + benefit, plus the expectations
    block kept as a standalone appendix chart so the item-level CSV
    still covers all twelve items."""
    panel = plot_survey_panels(dfB)

    appendix = plot_survey_block(
        dfB,
        "expect",
        "Participant expectations and placement experience",
        "survey_expectations.png",
    )

    out = pd.concat([panel, appendix], ignore_index=True)

    out.to_csv(
        OUTDIR / "descriptive_survey_items.csv",
        index=False,
    )

    return out


# ---------------------------------------------------------------------
# 8. SURVEY RESPONDER SELECTION COMPARISON
# ---------------------------------------------------------------------

def selection_descriptives(dfA, dfB):
    sel = lg.selection_table(dfA, dfB)

    sel.to_csv(
        OUTDIR / "descriptive_survey_selection.csv",
        index=False,
    )

    print("\n--- Survey responder comparison ---")
    print(sel.to_string(index=False))

    return sel


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main():
    dfA, dfB, dfA_survey = build_frames()

    frame_summary(dfA, dfB)

    plot_age_distribution(dfA, dfB)
    plot_borough_distribution(dfA, dfB)

    plot_hours_distribution(dfA)
    return_rate_plots(dfA)

    survey_descriptives(dfB)
    selection_descriptives(dfA, dfB)

    print(f"\nPlots written to: {PLOTDIR}")
    print(f"Tables written to: {OUTDIR}")


if __name__ == "__main__":
    main()