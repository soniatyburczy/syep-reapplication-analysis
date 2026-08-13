"""
provider_stability.py -- is a provider's adjusted excess a stable property?

Add-on to lg.py and provider_analysis.py. The caterpillar shows providers
departing from expectation. It cannot say whether a given provider departs
the same way next year. If the ranking reshuffles between periods, then
"providers differ" survives but "this provider is good" does not, and no
provider can be named in a recommendation.

METHOD
------
Fit the participant model once on all provider-complete person-years (Year
is already a covariate, so period shifts are absorbed), then compute each
provider's mean residual separately in each half and correlate the two.

Two splits, because they fail in different directions:

* by_year   -- 2022-2023 vs 2024. The natural test, but a participant who
               returns appears in both halves at the same provider, which
               correlates the halves by construction and inflates r.
* random    -- people split at random into two groups. Person-disjoint,
               and free of the tenure confound that a first-year split
               would carry, but it tests reproducibility rather than
               persistence over time. If by_year is weak and random is
               strong, the spread is real but moving; if both are weak,
               it is noise.

DISATTENUATION
--------------
Each half's excess is measured with error, which drags the observed
correlation toward zero. A modest r can therefore mean "unstable" or
"noisy" and the raw number cannot tell them apart. Reliability is estimated
from the bootstrap sampling variance and used to report a corrected
correlation alongside the observed one. The corrected figure is an upper
bound on stability, not a replacement for it.

Run:  python provider_stability.py
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

_LG = Path(__file__).resolve().parent / "lg.py"
_spec = importlib.util.spec_from_file_location("lg", _LG)
lg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lg)

ROOT = Path(__file__).resolve().parent.parent
lg.OUT = ROOT / "data" / "models"
PLOTDIR = Path("plots")

EARLY_YEARS = (2022, 2023)
LATE_YEARS = (2024,)

# A provider needs enough person-years in BOTH halves for the pair to mean
# anything; below this the point is mostly sampling noise in at least one
# half and will flatten the correlation for no good reason.
MIN_HALF_N = 150

N_BOOT = 2000
SEED = 20260811
TOP_K = 10

INK, SUBTLE, GRID = "#1B1F24", "#5C6670", "#DCE1E6"
ABOVE, BELOW, FLAT = "#2F5B7C", "#C2703D", "#B8C0C8"


def base_fit(dfA: pd.DataFrame):
    """The participant model, without provider, on provider-complete rows."""
    demo, _ = lg.resolve_demographic_terms(dfA, "stability_base")
    return lg.fit(dfA, lg.CORE + lg.QUALITY + demo,
                  "stability_base", cluster=lg.ID)


def add_residuals(res, used: pd.DataFrame) -> pd.DataFrame:
    out = used.copy()
    out["_p_hat"] = res.predict(out)
    out["_resid"] = out["returned"].to_numpy() - out["_p_hat"].to_numpy()
    return out


def assign_halves(df: pd.DataFrame, how: str) -> pd.Series:
    """Label each person-year 'early' or 'late'."""
    if how == "by_year":
        return np.where(df["Year"].isin(EARLY_YEARS), "early", "late")

    if how == "random":
        # Split people, not rows: a participant's person-years must land
        # together or the halves share individuals and the correlation is
        # measuring the same people twice.
        people = pd.Index(pd.unique(df[lg.ID]))
        rng = np.random.default_rng(SEED)
        side = pd.Series(
            np.where(rng.random(len(people)) < 0.5, "early", "late"),
            index=people,
        )
        return side.reindex(df[lg.ID]).to_numpy()

    raise ValueError(f"unknown split: {how!r}")


def half_excess(df: pd.DataFrame, how: str,
                n_boot: int = N_BOOT) -> pd.DataFrame:
    """Adjusted excess per provider within each half, with bootstrap SEs.

    The SE is what makes the reliability correction possible, so it is
    computed here rather than left to the caller.
    """
    d = df.copy()
    d["_half"] = assign_halves(d, how)

    rng = np.random.default_rng(SEED)
    rows = []

    for half in ("early", "late"):
        sub = d.loc[d["_half"].eq(half)]
        if not len(sub):
            continue

        grp = sub.groupby(lg.CLUSTER, observed=True)["_resid"]
        point = grp.mean().mul(100)
        n = grp.size()

        # Person-level resample within the half: a participant's several
        # person-years at one provider are not independent draws.
        people = sub[lg.ID].to_numpy()
        uniq = pd.unique(people)
        code = pd.Series(np.arange(len(uniq)),
                         index=uniq).reindex(people).to_numpy()
        order = np.argsort(code, kind="stable")
        resid = sub["_resid"].to_numpy()[order]
        cats = sorted(sub[lg.CLUSTER].dropna().unique())
        pcodes = pd.Categorical(sub[lg.CLUSTER].to_numpy()[order],
                                categories=cats).codes
        starts = np.searchsorted(code[order], np.arange(len(uniq)))
        ends = np.searchsorted(code[order], np.arange(len(uniq)),
                               side="right")

        draws = np.full((n_boot, len(cats)), np.nan)
        for b in range(n_boot):
            pick = rng.integers(0, len(uniq), len(uniq))
            idx = np.concatenate([np.arange(starts[i], ends[i])
                                  for i in pick])
            s = np.bincount(pcodes[idx], weights=resid[idx],
                            minlength=len(cats))
            c = np.bincount(pcodes[idx], minlength=len(cats))
            with np.errstate(invalid="ignore", divide="ignore"):
                draws[b] = np.where(c > 0, 100 * s / np.maximum(c, 1), np.nan)

        se = np.nanstd(draws, axis=0)

        rows.append(pd.DataFrame({
            lg.CLUSTER: cats,
            "half": half,
            "excess_pp": point.reindex(cats).to_numpy(),
            "se_pp": se,
            "n": n.reindex(cats).to_numpy(),
        }))

    return pd.concat(rows, ignore_index=True)


def reliability(excess: np.ndarray, se: np.ndarray) -> float:
    """Share of observed between-provider variance that is signal.

    var(observed) = var(true) + mean(sampling variance), so subtracting the
    mean squared standard error leaves an estimate of the true spread.
    Clipped at zero: a negative estimate means the observed spread is
    entirely consistent with noise.
    """
    obs = float(np.nanvar(excess, ddof=1))
    noise = float(np.nanmean(se ** 2))
    return float(np.clip((obs - noise) / obs, 0.0, 1.0)) if obs > 0 else 0.0


def compare_halves(tab: pd.DataFrame, how: str) -> dict:
    wide = tab.pivot(index=lg.CLUSTER, columns="half",
                     values=["excess_pp", "se_pp", "n"])
    wide.columns = [f"{a}_{b}" for a, b in wide.columns]
    wide = wide.dropna(subset=["excess_pp_early", "excess_pp_late"])

    keep = wide.loc[(wide["n_early"] >= MIN_HALF_N)
                    & (wide["n_late"] >= MIN_HALF_N)].copy()

    lg.log(f"\n--- split: {how} ---")
    lg.log(f"{len(wide)} providers in both halves; {len(keep)} clear the "
           f"{MIN_HALF_N}-person-year floor in both")

    if len(keep) < 8:
        lg.log("  too few providers to correlate; reporting the pairs only.")
        return {"split": how, "n_providers": len(keep), "pearson": np.nan,
                "spearman": np.nan, "corrected": np.nan,
                "rel_early": np.nan, "rel_late": np.nan,
                "top_overlap": np.nan, "sign_agree": np.nan, "frame": keep}

    r, p_r = stats.pearsonr(keep["excess_pp_early"], keep["excess_pp_late"])
    rho, p_rho = stats.spearmanr(keep["excess_pp_early"],
                                 keep["excess_pp_late"])

    rel_e = reliability(keep["excess_pp_early"].to_numpy(),
                        keep["se_pp_early"].to_numpy())
    rel_l = reliability(keep["excess_pp_late"].to_numpy(),
                        keep["se_pp_late"].to_numpy())
    denom = np.sqrt(rel_e * rel_l)
    corrected = float(np.clip(r / denom, -1, 1)) if denom > 0 else np.nan

    k = min(TOP_K, len(keep) // 2)
    top_e = set(keep["excess_pp_early"].nlargest(k).index)
    top_l = set(keep["excess_pp_late"].nlargest(k).index)
    overlap = len(top_e & top_l)

    sign_agree = float(
        np.mean(np.sign(keep["excess_pp_early"])
                == np.sign(keep["excess_pp_late"]))
    )

    lg.log(f"Pearson r  = {r:.3f} (p={p_r:.4g})")
    lg.log(f"Spearman   = {rho:.3f} (p={p_rho:.4g})")
    if denom > 0:
        lg.log(f"reliability: early={rel_e:.2f} late={rel_l:.2f} "
               f"-> disattenuated r = {corrected:.3f} (upper bound)")
    else:
        lg.log(f"reliability: early={rel_e:.2f} late={rel_l:.2f} -- the "
               f"observed spread in at least one half is fully consistent "
               f"with sampling noise, so no correction is possible and the "
               f"raw r should be read as an upper bound already")
    lg.log(f"top-{k} overlap = {overlap}/{k}")
    lg.log(f"same side of zero in both halves = {100*sign_agree:.0f}% "
           f"of providers")

    if r >= 0.6:
        lg.log("  -> stable. A provider's position persists, so naming "
               "specific providers in a recommendation is defensible.")
    elif r >= 0.3:
        lg.log("  -> partially stable. Report the spread as real but treat "
               "individual provider rankings as indicative; say so before "
               "anyone builds a league table from the caterpillar.")
    else:
        lg.log("  -> NOT stable. The between-provider spread does not "
               "persist across periods. Report that providers vary and "
               "that this data cannot identify which ones are good; do "
               "not name providers.")

    return {"split": how, "n_providers": len(keep), "pearson": r,
            "spearman": rho, "corrected": corrected, "rel_early": rel_e,
            "rel_late": rel_l, "top_overlap": overlap, "sign_agree":
            sign_agree, "frame": keep}


def plot_stability(result: dict) -> None:
    keep = result["frame"]
    if len(keep) < 8:
        return

    PLOTDIR.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.6, 8.0))

    lim = float(np.nanmax(np.abs(
        np.r_[keep["excess_pp_early"], keep["excess_pp_late"]]))) * 1.15

    ax.axhline(0, color=GRID, linewidth=1.2, zorder=1)
    ax.axvline(0, color=GRID, linewidth=1.2, zorder=1)
    ax.plot([-lim, lim], [-lim, lim], linestyle="--", linewidth=1.4,
            color=SUBTLE, alpha=0.7, zorder=2)

    agree = np.sign(keep["excess_pp_early"]) == np.sign(keep["excess_pp_late"])
    size = 18 + 190 * (keep["n_early"] + keep["n_late"]) / \
        (keep["n_early"] + keep["n_late"]).max()

    ax.scatter(keep["excess_pp_early"], keep["excess_pp_late"],
               s=size, c=np.where(agree, ABOVE, FLAT),
               edgecolors="white", linewidths=1.1, zorder=4)

    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("Adjusted excess, 2022\u20132023 (pp)", fontsize=12.5)
    ax.set_ylabel("Adjusted excess, 2024 (pp)", fontsize=12.5)
    ax.tick_params(labelsize=11.5, colors=SUBTLE)
    ax.grid(color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    r = result["pearson"]
    verdict = ("holds up" if r >= 0.6
               else "partly holds up" if r >= 0.3 else "does not hold up")
    ax.set_title(f"Provider position {verdict} across years",
                 loc="left", fontsize=17, fontweight="bold", pad=34)
    ax.annotate(
        f"r = {r:.2f} \u00b7 {int(result['top_overlap'])} of the top "
        f"{TOP_K} repeat \u00b7 dot size is person-years \u00b7 "
        f"grey = switched sides",
        xy=(0, 1), xycoords="axes fraction", xytext=(0, 8),
        textcoords="offset points", ha="left", va="bottom",
        fontsize=11, color=SUBTLE,
    )

    fig.savefig(PLOTDIR / "provider_stability.png", bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    lg.log(f"\nwrote {PLOTDIR / 'provider_stability.png'}")


def main() -> None:
    lg.OUT.mkdir(parents=True, exist_ok=True)

    dfA = lg.prep().dropna(subset=[lg.CLUSTER]).copy()
    lg.log(f"provider-complete person-years: {len(dfA):,}")

    res, used = base_fit(dfA)
    scored = add_residuals(res, used)

    early_people = set(scored.loc[scored["Year"].isin(EARLY_YEARS), lg.ID])
    late_people = set(scored.loc[scored["Year"].isin(LATE_YEARS), lg.ID])
    both = len(early_people & late_people)
    lg.log(f"\n{both:,} people appear in both periods "
           f"({100 * both / len(late_people):.1f}% of the 2024 group). "
           f"That overlap is why the by_person split is reported alongside.")

    results, frames = [], []
    for how in ("by_year", "random"):
        tab = half_excess(scored, how)
        out = compare_halves(tab, how)
        frames.append(out.pop("frame").assign(split=how).reset_index())
        results.append(out)

    summary = pd.DataFrame(results)
    lg.log("\n=== STABILITY SUMMARY ===")
    lg.log(summary.to_string(index=False,
                             float_format=lambda v: f"{v:,.3f}"))

    if summary["pearson"].notna().all():
        gap = abs(summary["pearson"].iloc[0] - summary["pearson"].iloc[1])
        if gap > 0.15:
            lg.log(f"\nthe two splits disagree by {gap:.2f}. A high random "
                   f"split with a low year split means the spread is real "
                   f"but not persistent -- report variance, do not name "
                   f"providers.")

    plot_stability({**results[0], "frame": frames[0].set_index(lg.CLUSTER)})

    pd.concat(frames, ignore_index=True).to_csv(
        lg.OUT / "provider_stability_pairs.csv", index=False)
    summary.to_csv(lg.OUT / "provider_stability.csv", index=False)
    (lg.OUT / "provider_stability_log.txt").write_text(
        "\n".join(lg._LOG), encoding="utf-8")

    print(f"\nwrote {lg.OUT}")


if __name__ == "__main__":
    main()