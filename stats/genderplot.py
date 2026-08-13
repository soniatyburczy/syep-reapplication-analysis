
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from patsy import build_design_matrices
from scipy.special import expit

_HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_HERE))

import cohort_table as ct
import models.lg as lg

FIGS = ct.ROOT / "plots"
N_DRAWS = 2000
DRAW_CHUNK = 100
SEED = 20260811
KEEP = ["Female", "Male"]

FEMALE_COLOR = "#9ecae1"
MALE_COLOR = "#3182bd"


def adjusted_rates(res, frame: pd.DataFrame, rng) -> pd.DataFrame:
    """Counterfactual adjusted rates and their clustered sampling intervals."""
    design_info = res.model.data.design_info

    designs = {}
    for level in KEEP:
        cf = frame.copy()
        cf["gender"] = level
        designs[level] = build_design_matrices(
            [design_info], cf, return_type="dataframe"
        )[0].to_numpy()

    # Guard: the two counterfactual designs must differ only in gender columns.
    differing = np.where(
        ~np.isclose(designs["Female"], designs["Male"]).all(axis=0)
    )[0]
    names = [design_info.column_names[i] for i in differing]
    if any("gender" not in nm for nm in names):
        raise RuntimeError(
            "Setting gender changed non-gender design columns, so the "
            f"contrast is not a clean counterfactual: {names}"
        )

    draws = rng.multivariate_normal(
        res.params.values, res.cov_params().values, size=N_DRAWS
    )

    sims = {}
    for level, X in designs.items():
        chunks = [
            expit(X @ draws[i:i + DRAW_CHUNK].T).mean(axis=0)
            for i in range(0, N_DRAWS, DRAW_CHUNK)
        ]
        sims[level] = np.concatenate(chunks)

    rows = []
    for level, X in designs.items():
        point = float(expit(X @ res.params.values).mean())
        lo, hi = np.percentile(sims[level], [2.5, 97.5])
        rows.append({
            "gender": level,
            "percent": 100 * point,
            "lo": 100 * float(lo),
            "hi": 100 * float(hi),
        })

    out = pd.DataFrame(rows).set_index("gender").loc[KEEP].reset_index()
    gap_sims = 100 * (sims["Female"] - sims["Male"])
    out.attrs["gap"] = out.loc[0, "percent"] - out.loc[1, "percent"]
    out.attrs["gap_lo"] = float(np.percentile(gap_sims, 2.5))
    out.attrs["gap_hi"] = float(np.percentile(gap_sims, 97.5))
    return out


def main() -> None:
    rng = np.random.default_rng(SEED)

    dfA = lg.prep()
    dfA_prov = dfA.dropna(subset=[lg.CLUSTER]).copy()
    demo_terms, _ = lg.resolve_demographic_terms(dfA_prov, "A_full")
    res, used = lg.fit(dfA_prov, lg.CORE + lg.QUALITY + demo_terms, "A_full")

    # Standardise over the male/female person-years only, so the adjusted rate
    # describes a population the audience can name. Rows with other or
    # unrecorded gender still informed the fit.
    frame = used.loc[used["gender"].isin(KEEP)].copy()
    n_male = int((frame["gender"] == "Male").sum())
    print(f"standardising over {len(frame):,} of {len(used):,} fitted "
          f"person-years ({n_male:,} male)")

    rate = adjusted_rates(res, frame, rng)
    gap = rate.attrs["gap"]

    raw = frame.groupby("gender")["returned"].mean().mul(100)
    raw_gap = raw["Female"] - raw["Male"]
    print(f"raw gap      : {raw_gap:.2f} pp")
    print(f"adjusted gap : {gap:.2f} pp "
          f"[{rate.attrs['gap_lo']:.2f}, {rate.attrs['gap_hi']:.2f}]")

    shortfall = round(n_male * gap / 100)

    FIGS.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.0, 3.4))

    y = {"Female": 1, "Male": 0}
    ax.plot(rate["percent"], [y[g] for g in rate["gender"]],
            color="#bdbdbd", linewidth=2.5, zorder=1, solid_capstyle="round")

    for _, r in rate.iterrows():
        ax.plot([r["lo"], r["hi"]], [y[r["gender"]]] * 2,
                color="#666666", linewidth=1.4, zorder=2)
        ax.scatter(r["percent"], y[r["gender"]], s=190, zorder=3,
                   color=FEMALE_COLOR if r["gender"] == "Female" else MALE_COLOR,
                   edgecolor="white", linewidth=1.5)
        ax.annotate(f"{r['percent']:.1f}%",
                    xy=(r["percent"], y[r["gender"]]), xytext=(0, 16),
                    textcoords="offset points", ha="center",
                    fontsize=12, fontweight="bold", color="#222222")

    mid = rate["percent"].mean()
    ax.annotate(f"{gap:.1f} point gap", xy=(mid, 0.5), ha="center",
                va="center", fontsize=11, color="#444444",
                bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                          edgecolor="#dddddd"))

    span = rate["hi"].max() - rate["lo"].min()
    ax.set_xlim(rate["lo"].min() - 0.45 * span, rate["hi"].max() + 0.45 * span)
    ax.set_ylim(-0.55, 1.75)
    ax.set_yticks([1, 0])
    ax.set_yticklabels(["Female", "Male"], fontsize=12)
    ax.set_xlabel("Adjusted reapplication rate (%)")

    ax.set_title(
        "Young men reapply less often than young women with the same\n"
        "age, hours worked, borough, and placement outcome",
        loc="left", fontsize=13, pad=28,
    )
    ax.text(0.0, 1.03,
            f"About {shortfall:,} fewer returning male participants across "
            f"the study period",
            transform=ax.transAxes, fontsize=10, color="#555555")
    ax.annotate(
        "Bars show 95% intervals with standard errors clustered by provider.\n"
        f"Unadjusted gap: {raw_gap:.1f} points. First observed person-year "
        "only, 2022–2025.",
        xy=(0, -0.30), xycoords="axes fraction", fontsize=8,
        color="#666666", va="top")

    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color="#eeeeee", linewidth=0.8)
    ax.set_axisbelow(True)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIGS / f"gender_adjusted_dot.{ext}", dpi=300,
                    bbox_inches="tight")
    print(f"wrote {FIGS / 'gender_adjusted_dot.png'}")


if __name__ == "__main__":
    main()