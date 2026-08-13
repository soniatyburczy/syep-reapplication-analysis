"""
provider_analysis.py -- does the provider matter, net of who they serve?

Add-on to lg.py. Import-and-reuse, same pattern as age_analysis.py, so
frame construction, missing-value policy and the covariate set all come
from lg rather than being restated here.

THE QUESTION
------------
`provider` has so far entered only as a clustering variable. That answers
"are outcomes correlated within provider" but not "does any provider do
better than its participant mix predicts". This script asks the second.

The distinction matters because a provider serving mostly 16-year-olds who
work full hours will post a high raw return rate without doing anything
well. The adjusted excess below is the raw rate minus what the participant
model already expects from that provider's mix.

WHAT IT PRODUCES
----------------
1. provider_summary.csv      per provider: n, raw rate, adjusted excess pp,
                             CI, fixed-effect OR, small-cell flag
2. provider_tests.csv        Wald and LR tests on the provider block
3. provider_characteristics.csv
                             adjusted excess regressed on provider-level
                             composition (size, hours, no-show, borough)
4. provider_variance.txt     how much of the raw spread survives adjustment
5. plots/provider_caterpillar.png

INFERENCE NOTE
--------------
You cannot cluster on `provider` while `provider` is in the design, so the
fixed-effect fits cluster on person instead. That handles repeated
observations of the same participant, which is the dependence that remains
once provider is absorbed.

Run:  python provider_analysis.py
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

# Providers below this are estimated but flagged: with a binary outcome and
# a ~60% base rate, a 50-participant provider has a raw-rate standard error
# near 7 points, so it will appear at the extremes of any ranking by noise
# alone.
MIN_PROVIDER_N = 100

# Bootstrap resamples for the adjusted-excess intervals. Resampling is at
# the person level, not the row level, because the same participant
# contributes several person-years to the same provider.
N_BOOT = 2000
SEED = 20260811

SIG, MUTED, REF_C = "#1F5FA8", "#9AA6B1", "#C0392B"


# --------------------------------------------------------------- descriptives
def provider_descriptives(df: pd.DataFrame) -> pd.DataFrame:
    """Raw composition and outcome per provider, before any adjustment."""
    lg.log("\n=== 1. PROVIDER COMPOSITION ===")

    hrs = pd.to_numeric(df["total_hours_paid"], errors="coerce")

    tab = (
        df.assign(_hours=hrs, _zero=hrs.eq(0).astype(float))
        .groupby(lg.CLUSTER, observed=True)
        .agg(
            n=("returned", "size"),
            n_people=(lg.ID, "nunique"),
            k=("returned", "sum"),
            mean_hours=("_hours", "mean"),
            zero_hours_share=("_zero", "mean"),
            no_show_share=("is_no_show", "mean"),
            mean_age=("age_on_start", "mean"),
            n_years=("Year", "nunique"),
        )
    )
    tab["raw_rate"] = tab["k"] / tab["n"]
    tab["small"] = tab["n"] < MIN_PROVIDER_N

    lg.log(f"{len(tab):,} providers over {int(tab['n'].sum()):,} "
           f"person-years")
    lg.log(f"size: min={tab['n'].min():,} median={tab['n'].median():,.0f} "
           f"max={tab['n'].max():,}")
    lg.log(f"{int(tab['small'].sum())} provider(s) under {MIN_PROVIDER_N} "
           f"person-years, flagged not dropped")
    lg.log(f"raw return rate: min={tab['raw_rate'].min():.3f} "
           f"median={tab['raw_rate'].median():.3f} "
           f"max={tab['raw_rate'].max():.3f} "
           f"sd={tab['raw_rate'].std():.4f}")

    return tab.reset_index()


# ------------------------------------------------------------ adjusted excess
def adjusted_excess(res, used: pd.DataFrame,
                    n_boot: int = N_BOOT) -> pd.DataFrame:
    """Mean response residual per provider, in percentage points.

    This is the provider's realised return rate minus the rate the
    participant-level model predicts from its mix. Positive means the
    provider's participants came back more often than their ages, hours,
    boroughs, years and placement outcomes would suggest.

    Reported on the probability scale rather than as a log-odds because
    "three points better than expected" is a sentence a program officer can
    act on and "0.13 log-odds" is not.

    Intervals come from a person-level cluster bootstrap. Resampling rows
    would treat a participant's three person-years as three independent
    observations and give intervals that are too narrow.
    """
    lg.log("\n=== 3. ADJUSTED EXCESS RETURN BY PROVIDER ===")

    frame = used.copy()
    frame["_p_hat"] = res.predict(frame)
    frame["_resid"] = frame["returned"].to_numpy() - frame["_p_hat"].to_numpy()

    point = (
        frame.groupby(lg.CLUSTER, observed=True)["_resid"]
        .mean()
        .mul(100)
        .rename("excess_pp")
    )

    # Person-level bootstrap. Persons are drawn with replacement from the
    # whole frame; a provider's estimate in a given draw uses whichever of
    # its people were drawn.
    rng = np.random.default_rng(SEED)
    people = frame[lg.ID].to_numpy()
    uniq = pd.unique(people)
    code = pd.Series(np.arange(len(uniq)), index=uniq).reindex(people).to_numpy()

    order = np.argsort(code, kind="stable")
    resid_sorted = frame["_resid"].to_numpy()[order]
    prov_sorted = pd.Categorical(
        frame[lg.CLUSTER].to_numpy()[order],
        categories=sorted(frame[lg.CLUSTER].dropna().unique()),
    )
    starts = np.searchsorted(code[order], np.arange(len(uniq)))
    ends = np.searchsorted(code[order], np.arange(len(uniq)), side="right")

    n_prov = len(prov_sorted.categories)
    prov_codes = prov_sorted.codes
    draws = np.full((n_boot, n_prov), np.nan)

    for b in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        idx = np.concatenate([np.arange(starts[i], ends[i]) for i in pick])
        s = np.bincount(prov_codes[idx], weights=resid_sorted[idx],
                        minlength=n_prov)
        c = np.bincount(prov_codes[idx], minlength=n_prov)
        with np.errstate(invalid="ignore", divide="ignore"):
            draws[b] = np.where(c > 0, 100 * s / np.maximum(c, 1), np.nan)

    lo = np.nanpercentile(draws, 2.5, axis=0)
    hi = np.nanpercentile(draws, 97.5, axis=0)

    out = pd.DataFrame({
        lg.CLUSTER: list(prov_sorted.categories),
        "excess_pp": point.reindex(prov_sorted.categories).to_numpy(),
        "excess_lo_pp": lo,
        "excess_hi_pp": hi,
    })
    out["separates"] = (out["excess_lo_pp"] > 0) | (out["excess_hi_pp"] < 0)

    n_sep = int(out["separates"].sum())
    lg.log(f"{n_sep} of {len(out)} providers have an adjusted excess whose "
           f"95% interval excludes zero")
    lg.log(f"excess spread: min={out['excess_pp'].min():.1f}pp "
           f"max={out['excess_pp'].max():.1f}pp "
           f"sd={out['excess_pp'].std():.2f}pp")

    if n_sep == 0:
        lg.log("  -> no provider separates from zero. Once participant mix "
               "is accounted for, this data cannot distinguish providers "
               "from one another. That is a finding, not a failure: it "
               "says the raw spread in provider return rates is about who "
               "they serve.")

    return out


# ------------------------------------------------------------- formal testing
def provider_block_tests(dfA: pd.DataFrame, base_terms: list[str]):
    """Is the provider block jointly non-zero, net of participant mix?

    Two tests, because neither alone is clean:

    * Wald, using the person-clustered covariance. Robust to the repeated-
      person dependence that survives provider absorption. This is the one
      to quote.
    * Likelihood ratio, from the plain MLE log-likelihoods. Its reference
      chi-square assumes independent observations, which is false here, so
      it will be anti-conservative. Reported for comparison only.
    """
    lg.log("\n=== 2. IS THE PROVIDER BLOCK JOINTLY NON-ZERO? ===")

    counts = dfA[lg.CLUSTER].value_counts()
    ref = str(counts.idxmax())
    if '"' in ref:
        raise ValueError(f"Provider name contains a quote: {ref!r}")
    lg.log(f"reference provider = {ref!r} (n={int(counts.max()):,})")

    prov_term = f'C({lg.CLUSTER}, Treatment(reference="{ref}"))'

    res_base, used_base = lg.fit(
        dfA, base_terms, "provider_base", cluster=lg.ID)
    res_fe, used_fe = lg.fit(
        dfA, base_terms + [prov_term], "provider_fe", cluster=lg.ID)

    if int(res_base.nobs) != int(res_fe.nobs):
        raise RuntimeError(
            "base and provider-FE fits used different samples "
            f"({int(res_base.nobs):,} vs {int(res_fe.nobs):,}); the block "
            "test is not interpretable."
        )

    prov_params = [t for t in res_fe.params.index
                   if t.startswith(f"C({lg.CLUSTER}")]
    if not prov_params:
        raise RuntimeError("no provider coefficients in the FE fit.")

    wald = res_fe.wald_test(
        np.eye(len(res_fe.params))[
            [res_fe.params.index.get_loc(t) for t in prov_params]
        ],
        scalar=True,
    )
    w_chi2 = float(np.squeeze(wald.statistic))
    w_p = float(np.squeeze(wald.pvalue))
    w_df = len(prov_params)

    lr_chi2 = 2 * (res_fe.llf - res_base.llf)
    lr_p = float(stats.chi2.sf(lr_chi2, w_df))

    tests = pd.DataFrame([
        {"test": "Wald (person-clustered)", "df": w_df,
         "chi2": w_chi2, "p": w_p,
         "note": "quote this one"},
        {"test": "Likelihood ratio (independence)", "df": w_df,
         "chi2": lr_chi2, "p": lr_p,
         "note": "assumes independent rows; anti-conservative here"},
    ])

    lg.log(tests.to_string(index=False, float_format=lambda v: f"{v:,.4g}"))
    lg.log(f"\npseudo-R2: {res_base.prsquared:.4f} without provider, "
           f"{res_fe.prsquared:.4f} with "
           f"(+{res_fe.prsquared - res_base.prsquared:.4f})")

    if w_p >= 0.05:
        lg.log("  -> the provider block does not clear conventional "
               "significance. Report that provider identity adds nothing "
               "detectable once participant mix is adjusted for.")
    else:
        lg.log("  -> providers differ jointly. The caterpillar plot shows "
               "whether that is a few outliers or a broad spread; a "
               "significant block test with no individually separating "
               "provider means the former is not established either.")

    fe = pd.DataFrame({
        lg.CLUSTER: [
            t.split("[T.")[1].rstrip("]") for t in prov_params
        ],
        "fe_or": np.exp(res_fe.params[prov_params].to_numpy()),
        "fe_or_lo": np.exp(res_fe.conf_int().loc[prov_params, 0].to_numpy()),
        "fe_or_hi": np.exp(res_fe.conf_int().loc[prov_params, 1].to_numpy()),
        "fe_p": res_fe.pvalues[prov_params].to_numpy(),
    })
    fe.loc[len(fe)] = {lg.CLUSTER: ref, "fe_or": 1.0, "fe_or_lo": np.nan,
                       "fe_or_hi": np.nan, "fe_p": np.nan}

    return tests, fe, res_base, used_base


# --------------------------------------------------- what explains the spread
def variance_accounted(summary: pd.DataFrame) -> str:
    """How much of the raw between-provider spread survives adjustment."""
    raw_sd = float(summary["raw_rate"].mul(100).std())
    adj_sd = float(summary["excess_pp"].std())
    shrink = 100 * (1 - adj_sd / raw_sd) if raw_sd > 0 else np.nan

    rho = summary[["raw_rate", "excess_pp"]].corr(method="spearman").iloc[0, 1]

    # The reading depends on which way the numbers came out, so it is
    # derived rather than asserted -- a hardcoded sentence here would be
    # wrong roughly half the time.
    if shrink >= 40 and rho < 0.6:
        verdict = (
            "Most of the raw spread is participant mix, and the raw "
            "ranking does not survive adjustment: the providers that look "
            "best on unadjusted return rates are largely not the ones "
            "doing best for a comparable participant. Any published "
            "provider league table based on raw rates is misleading."
        )
    elif shrink >= 40:
        verdict = (
            "Most of the raw spread is participant mix, but the ranking is "
            "broadly preserved -- adjustment compresses the differences "
            "without reordering them."
        )
    elif rho < 0.6:
        verdict = (
            "Adjustment barely narrows the spread yet substantially "
            "reorders it, which is unusual; check whether a few small "
            "providers are driving the rank correlation before reporting."
        )
    else:
        verdict = (
            "Adjustment changes neither the spread nor the ordering much, "
            "so differences between providers are not explained by the "
            "participant characteristics in the model. That does not make "
            "them provider effects -- it means the cause is something not "
            "measured here."
        )

    text = (
        f"raw provider return rate sd      = {raw_sd:.2f} pp\n"
        f"adjusted excess sd               = {adj_sd:.2f} pp\n"
        f"spread accounted for by mix      = {shrink:.1f}%\n"
        f"Spearman(raw rate, adj excess)   = {rho:.3f}\n\n"
        + verdict + "\n"
    )
    lg.log("\n=== 4. VARIANCE ACCOUNTED FOR BY PARTICIPANT MIX ===")
    lg.log(text)
    return text


def provider_characteristics(summary: pd.DataFrame) -> pd.DataFrame:
    """Does provider composition predict its adjusted excess?

    Weighted least squares of excess on provider-level aggregates. This is
    descriptive only, and specifically NOT causal: mean hours is computed
    from the same participants whose residuals form the outcome, so a
    provider whose participants work more will tend to show up on both
    sides. Read it as "what kind of provider sits where", not "raise hours
    and the excess follows".
    """
    lg.log("\n=== 5. PROVIDER COMPOSITION vs ADJUSTED EXCESS ===")

    import statsmodels.formula.api as smf

    d = summary.dropna(subset=["excess_pp"]).copy()
    d["log_n"] = np.log(d["n"])

    # Five predictors on ~50 providers is already thin; below 30 the
    # intervals are so wide the table invites overreading a point estimate.
    if len(d) < 30:
        lg.log(f"only {len(d)} providers: five predictors cannot be "
               f"estimated usefully here. Reporting for completeness; do "
               f"not put this table on a slide.")

    model = smf.wls(
        "excess_pp ~ mean_hours + zero_hours_share + no_show_share "
        "+ mean_age + log_n",
        data=d,
        weights=d["n"],
    ).fit(cov_type="HC3")

    out = pd.DataFrame({
        "term": model.params.index,
        "coef_pp": model.params.to_numpy(),
        "lo": model.conf_int()[0].to_numpy(),
        "hi": model.conf_int()[1].to_numpy(),
        "p": model.pvalues.to_numpy(),
    })

    lg.log(f"n providers = {int(model.nobs)}, R2 = {model.rsquared:.3f}")
    lg.log(out.to_string(index=False, float_format=lambda v: f"{v:,.4f}"))
    lg.log("Descriptive only: the right-hand side is aggregated from the "
           "same participants that produced the residuals.")

    return out


# --------------------------------------------------------------------- plot
def caterpillar(summary: pd.DataFrame) -> None:
    PLOTDIR.mkdir(exist_ok=True)

    d = summary.dropna(subset=["excess_pp"]).sort_values("excess_pp")
    y = np.arange(len(d))

    fig, ax = plt.subplots(figsize=(11, max(5.0, 0.24 * len(d) + 2.4)))

    ax.axvline(0, linestyle="--", linewidth=1.6, color=REF_C, zorder=1)

    for i, row in enumerate(d.itertuples()):
        sep = bool(row.separates)
        small = bool(row.small)
        colour = MUTED if (small or not sep) else SIG
        ax.plot([row.excess_lo_pp, row.excess_hi_pp], [i, i],
                color=colour, linewidth=2.0,
                alpha=0.45 if small else 1.0, zorder=3)
        ax.plot(row.excess_pp, i, "o", markersize=6 if small else 7.5,
                markerfacecolor="white" if small else colour,
                markeredgecolor=colour, markeredgewidth=1.8, zorder=4)

    ax.set_yticks([])
    ax.set_ylim(-1, len(d))
    ax.set_xlabel("Adjusted excess return, percentage points "
                  "(provider minus its expected rate)")
    ax.set_title("Do any providers beat their participant mix?",
                 loc="left", fontsize=17, pad=26)
    ax.annotate(
        f"Each line is one provider · 95% person-bootstrap interval · "
        f"hollow = under {MIN_PROVIDER_N} person-years",
        xy=(0, 1), xycoords="axes fraction", xytext=(0, 8),
        textcoords="offset points", ha="left", va="bottom",
        fontsize=12, color="#5C6670",
    )
    ax.grid(axis="x", color="#DCE1E6", linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)

    fig.tight_layout()
    fig.savefig(PLOTDIR / "provider_caterpillar.png",
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    lg.log(f"\nwrote {PLOTDIR / 'provider_caterpillar.png'}")


# ---------------------------------------------------------------------- main
def main() -> None:
    lg.OUT.mkdir(parents=True, exist_ok=True)

    dfA = lg.prep()
    dfA = dfA.dropna(subset=[lg.CLUSTER]).copy()
    lg.log(f"\nprovider-complete person-years: {len(dfA):,}")
    lg.log("Provider is unrecorded on ~12% of A, concentrated in 2022, so "
           "this analysis is lighter on 2022 than the raw frame.")

    demo, _ = lg.resolve_demographic_terms(dfA, "provider_analysis")
    base_terms = lg.CORE + lg.QUALITY + demo

    desc = provider_descriptives(dfA)

    tests, fe, res_base, used_base = provider_block_tests(dfA, base_terms)

    excess = adjusted_excess(res_base, used_base)

    summary = (
        desc.merge(excess, on=lg.CLUSTER, how="left")
        .merge(fe, on=lg.CLUSTER, how="left")
        .sort_values("excess_pp", ascending=False)
    )

    var_text = variance_accounted(summary)
    chars = provider_characteristics(summary)

    # The movers table is the presentable output: who looks good raw but
    # is only average once you adjust, and vice versa.
    summary["rank_raw"] = summary["raw_rate"].rank(ascending=False)
    summary["rank_adj"] = summary["excess_pp"].rank(ascending=False)
    summary["rank_shift"] = summary["rank_raw"] - summary["rank_adj"]

    big = summary.loc[~summary["small"]]
    lg.log("\n=== 6. BIGGEST RANK MOVERS (providers above the size floor) ===")
    lg.log(
        big.reindex(big["rank_shift"].abs().sort_values(ascending=False).index)
        .head(10)[[lg.CLUSTER, "n", "raw_rate", "excess_pp",
                   "rank_raw", "rank_adj", "rank_shift"]]
        .to_string(index=False, float_format=lambda v: f"{v:,.3f}")
    )

    caterpillar(summary)

    summary.to_csv(lg.OUT / "provider_summary.csv", index=False)
    tests.to_csv(lg.OUT / "provider_tests.csv", index=False)
    chars.to_csv(lg.OUT / "provider_characteristics.csv", index=False)
    (lg.OUT / "provider_variance.txt").write_text(var_text, encoding="utf-8")
    (lg.OUT / "provider_analysis_log.txt").write_text(
        "\n".join(lg._LOG), encoding="utf-8")

    print(f"\nwrote {lg.OUT}")


if __name__ == "__main__":
    main()