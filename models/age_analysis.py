"""
age_analysis.py -- eligibility censoring fix, and age heterogeneity.

Add-on to lg.py. Import-and-reuse, same as block_tests.py, so clustering,
frame construction and missing-value policy all come from lg.

WHAT IT PRODUCES
----------------
1. age_eligibility.csv        return rate by single year of age + cliff check
2. primary_refit.csv          A_full before vs after the restriction
3. interaction_or.csv         interaction coefficients as ORs (appendix)
4. adjusted_probs_*.csv       marginal-standardised predicted probabilities
5. contrast_*.csv             within-group gaps, in percentage points
6. did_*.csv                  differences between those gaps across age groups
7. plots/hours_by_age.png, plots/firstchoice_by_age.png

Run:  python age_analysis.py
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from patsy import build_design_matrices
import importlib.util
from pathlib import Path

_LG = Path(__file__).resolve().parent / "lg.py"
_spec = importlib.util.spec_from_file_location("lg", _LG)
lg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lg)

ROOT = Path(__file__).resolve().parent.parent
lg.OUT = ROOT / "data" / "models"

AGE_MAX = 24                 # oldest age served by Older Youth
AGE_BINS = [15.5, 17.5, 20.5, 24.5]
AGE_LABELS = ["16-17", "18-20", "21-23"]
N_BOOT = 2000
SEED = 20260810
PLOTDIR = Path("plots")

HOURS = 'C(hours_band, Treatment(reference="126-149"))'
AGE_SPLINE = "bs(age_on_start, df=3)"
HEADLINE = "expect_first_choice"

SIG, ALT, THIRD = "#2166AC", "#D6604D", "#4D9221"
COLOURS = {"16-17": SIG, "18-20": ALT, "21-23": THIRD}


# --------------------------------------------------------------- derivations
def add_age_fields(df: pd.DataFrame) -> pd.DataFrame:
    """eligible_next_year and a coarse age_group, from age_on_start."""
    out = df.copy()
    age = pd.to_numeric(out["age_on_start"], errors="coerce")
    out["eligible_next_year"] = (age <= AGE_MAX - 1).astype("Int64")
    out["age_group"] = pd.cut(age, bins=AGE_BINS, labels=AGE_LABELS)
    return out


def eligibility_diagnostic(dfA: pd.DataFrame) -> pd.DataFrame:
    """Return rate by single year of age, and whether the cliff is real."""
    lg.log("\n=== 1. ELIGIBILITY CENSORING CHECK ===")
    age = pd.to_numeric(dfA["age_on_start"], errors="coerce").round().astype("Int64")
    tab = (dfA.assign(age=age)
              .groupby("age", observed=True)["returned"]
              .agg(n="size", returned="sum"))
    tab["rate"] = tab["returned"] / tab["n"]
    tab["eligible_next_year"] = (tab.index <= AGE_MAX - 1).astype(int)
    lg.log(tab.to_string(float_format=lambda v: f"{v:,.4f}"))

    top = tab.loc[tab.index >= AGE_MAX, "rate"]
    below = tab.loc[tab.index == AGE_MAX - 1, "rate"]
    if not len(top):
        lg.log(f"\nno age-{AGE_MAX} rows in the frame, so there is nothing to "
               "censor and the eligibility restriction is a no-op. "
               "primary_refit.csv will show identical columns. Worth "
               "confirming upstream why the top eligible age is absent.")
    elif len(below):
        t, b = float(top.iloc[0]), float(below.iloc[0])
        lg.log(f"\nage {AGE_MAX} return rate = {t:.4f}; "
               f"age {AGE_MAX-1} = {b:.4f}")
        if t < 0.01:
            lg.log(f"  -> cliff confirmed: age {AGE_MAX} is structurally "
                   "censored. Restricting to eligible_next_year == 1.")
        elif t < 0.5 * b:
            lg.log(f"  -> partial cliff: age {AGE_MAX} returns at some rate, "
                   "so eligibility is not absolute in this data (exceptions, "
                   "or age measured at a different reference date). Restrict "
                   "as primary, but report the unrestricted fit alongside.")
        else:
            lg.log(f"  !! NO cliff at age {AGE_MAX}. The eligibility rule is "
                   "not what this data does -- do NOT drop these rows on my "
                   "say-so. Check how age_on_start is defined first.")
    return tab.reset_index()


# ------------------------------------------------- marginal standardisation
def _set_value(frame: pd.DataFrame, var: str, val) -> pd.DataFrame:
    """Set one column to a constant, preserving categorical levels."""
    out = frame.copy()
    if isinstance(frame[var].dtype, pd.CategoricalDtype):
        if val not in list(frame[var].cat.categories):
            raise ValueError(
                f"{val!r} is not a level of {var!r}; levels are "
                f"{list(frame[var].cat.categories)}. Assigning it would "
                "produce an all-NaN column and a silently wrong design "
                "matrix.")
        out[var] = pd.Categorical([val] * len(out),
                                  categories=frame[var].cat.categories,
                                  ordered=frame[var].cat.ordered)
    else:
        out[var] = val
    return out


def _draws(res, n_boot: int) -> tuple[np.ndarray, np.ndarray]:
    """Parametric bootstrap draws from the fitted (clustered) covariance."""
    beta, cov = res.params.to_numpy(), np.asarray(res.cov_params())
    rng = np.random.default_rng(SEED)
    return beta, rng.multivariate_normal(beta, cov, size=n_boot, method="svd")


def _design(res, frame: pd.DataFrame, var: str, val) -> np.ndarray:
    cf = _set_value(frame, var, val)
    return np.asarray(build_design_matrices([res.model.data.design_info], cf,
                                            return_type="dataframe")[0])


def adjusted_probs(res, frame: pd.DataFrame, var: str, values: list,
                   by: str | None = None, n_boot: int = N_BOOT) -> pd.DataFrame:
    """
    Marginal-standardised predicted probability of `var` = each value,
    optionally computed separately within each level of `by`.

    Averaging predictions over the real covariate distribution, rather than
    predicting once at covariate means, because with categorical adjusters the
    "mean person" is 0.31 Brooklyn and does not exist.
    """
    beta, draws = _draws(res, n_boot)
    groups = ([(None, frame)] if by is None
              else list(frame.groupby(by, observed=True)))
    rows = []
    for gname, gframe in groups:
        if not len(gframe):
            continue
        for val in values:
            X = _design(res, gframe, var, val)
            boot = _expit(draws @ X.T).mean(axis=1)
            lo, hi = np.percentile(boot, [2.5, 97.5])
            rows.append({by or "group": gname, var: val, "n": len(gframe),
                         "prob": float(_expit(X @ beta).mean()),
                         "lo": float(lo), "hi": float(hi)})
    return pd.DataFrame(rows)


def prob_contrast(res, frame: pd.DataFrame, var: str, lo_val, hi_val,
                  by: str = "age_group", n_boot: int = N_BOOT):
    """
    Difference in adjusted probability between two values of `var`, within
    each level of `by`.

    Returns the table and the per-group bootstrap vectors. The vectors are
    kept so the difference of differences can be formed draw by draw, which
    preserves the correlation between two gaps estimated from the same fit --
    differencing the summary intervals instead would throw that away and give
    an interval that is too wide.
    """
    beta, draws = _draws(res, n_boot)
    rows, boots = [], {}
    for gname, gframe in frame.groupby(by, observed=True):
        if not len(gframe):
            continue
        X_lo = _design(res, gframe, var, lo_val)
        X_hi = _design(res, gframe, var, hi_val)
        point = float(_expit(X_hi @ beta).mean() - _expit(X_lo @ beta).mean())
        boot = (_expit(draws @ X_hi.T).mean(axis=1)
                - _expit(draws @ X_lo.T).mean(axis=1))
        lo, hi = np.percentile(boot, [2.5, 97.5])
        rows.append({by: gname, "from": lo_val, "to": hi_val,
                     "n": len(gframe), "diff_pp": 100 * point,
                     "lo_pp": 100 * float(lo), "hi_pp": 100 * float(hi)})
        boots[gname] = 100 * boot
    return pd.DataFrame(rows), boots


def contrast_difference(contrast: pd.DataFrame, boots: dict,
                        by: str = "age_group") -> pd.DataFrame:
    """
    Difference of differences: does the gap differ across age groups?

    This is the heterogeneity test. Every pair of groups is compared, so the
    p-values are not adjusted -- with three groups that is three comparisons,
    and if only one of them clears the bar it should be described as
    suggestive rather than established.
    """
    points = contrast.set_index(by)["diff_pp"].to_dict()
    keys = [k for k in contrast[by] if k in boots]
    rows = []
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            d = boots[a] - boots[b]
            lo, hi = np.percentile(d, [2.5, 97.5])
            # Two-sided bootstrap p: how much of the distribution sits on the
            # other side of zero from the point estimate.
            p = 2 * min((d <= 0).mean(), (d >= 0).mean())
            rows.append({"group_a": a, "group_b": b,
                         "gap_a_pp": points[a], "gap_b_pp": points[b],
                         "did_pp": points[a] - points[b],
                         "lo_pp": float(lo), "hi_pp": float(hi),
                         "p_boot": float(min(p, 1.0)),
                         "separates": bool(lo > 0 or hi < 0)})
    return pd.DataFrame(rows)


def _expit(x):
    return 1.0 / (1.0 + np.exp(-x))


# ------------------------------------------------------------------- fitting
def refit_primary(dfA_all: pd.DataFrame, dfA_elig: pd.DataFrame):
    """A_full before and after the eligibility restriction."""
    lg.log("\n=== 2. PRIMARY MODEL, BEFORE vs AFTER RESTRICTION ===")
    res_before, _ = lg.fit(dfA_all, lg.CORE + lg.QUALITY, "A_full_unrestricted")
    res_after, _ = lg.fit(dfA_elig, lg.CORE + lg.QUALITY, "A_full_eligible")

    common = [t for t in res_before.params.index if t in res_after.params.index]
    out = pd.DataFrame({
        "term": common,
        "or_unrestricted": np.exp(res_before.params[common]).to_numpy(),
        "or_eligible_only": np.exp(res_after.params[common]).to_numpy(),
        "p_eligible_only": res_after.pvalues[common].to_numpy(),
    })
    out["pct_change_log_or"] = 100 * (
        res_after.params[common].to_numpy() - res_before.params[common].to_numpy()
    ) / np.where(np.abs(res_before.params[common]) < 1e-8, np.nan,
                 np.abs(res_before.params[common]))
    lg.log(out.to_string(index=False, float_format=lambda v: f"{v:,.4f}"))

    moved = out.loc[out["term"].str.startswith("C(hours_band"),
                    "pct_change_log_or"].abs().max()
    lg.log(f"\nlargest hours-band shift: {moved:.1f}% on the log-OR scale "
           + ("-- gradient is robust to the censoring fix."
              if moved < 15 else
              "-- material. Every hours number in the deck needs updating."))
    return out, res_after


def interaction(df: pd.DataFrame, label: str, var: str,
                extra_terms: list[str] | None = None):
    """
    Fit `var * age_group`, swapping the age spline for the binned age.

    The spline stays in the primary model. Here it is replaced, because a
    3-df spline crossed with a 3-level factor is neither estimable at this n
    nor presentable, and keeping both a spline and bins of the same variable
    is near-collinear.
    """
    base = [t for t in lg.CORE if t != AGE_SPLINE] + lg.QUALITY
    base = [t for t in base if t != var]
    terms = base + [f"{var}*C(age_group)"] + (extra_terms or [])
    return lg.fit(df, terms, label)


# --------------------------------------------------------------------- plots
def plot_probs(tab: pd.DataFrame, var: str, order: list, title: str,
               subtitle: str, outfile: str, xlabel: str,
               group_col: str = "age_group") -> None:
    PLOTDIR.mkdir(exist_ok=True)
    plt.rcParams.update({"figure.dpi": 200, "savefig.dpi": 200,
                         "font.size": 13.5, "axes.titlesize": 18,
                         "axes.titleweight": "bold", "axes.labelsize": 14})
    fig, ax = plt.subplots(figsize=(11.5, 6.4))
    x = np.arange(len(order))
    offs = np.linspace(-0.13, 0.13, tab[group_col].nunique())

    for off, (g, sub) in zip(offs, tab.groupby(group_col, observed=True)):
        sub = sub.set_index(var).reindex(order).reset_index()
        c = COLOURS.get(str(g), "#666666")
        ax.plot(x + off, sub["prob"], color=c, lw=2.0, alpha=0.6, zorder=2)
        ax.vlines(x + off, sub["lo"], sub["hi"], color=c, lw=2.8, zorder=3)
        ax.plot(x + off, sub["prob"], "o", ms=10, color=c, zorder=4,
                label=f"{g}  (n={int(sub['n'].iloc[0]):,})")

    ax.set_xticks(x)
    ax.set_xticklabels(order)
    ax.set_xlim(-0.6, len(order) - 0.4)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Adjusted probability of reapplying")
    ax.set_title(title, pad=32)
    ax.text(0.0, 1.035, subtitle, transform=ax.transAxes, fontsize=12,
            color="#555555")
    ax.legend(frameon=False, fontsize=12.5, loc="best")
    ax.grid(axis="y", color="#E4E4E4", lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(PLOTDIR / outfile, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    lg.log(f"wrote {PLOTDIR/outfile}")


def report_did(did: pd.DataFrame, what: str) -> None:
    lg.log(f"\ndifference of differences, {what} (percentage points):")
    lg.log(did.to_string(index=False, float_format=lambda v: f"{v:,.2f}"))
    sep = did.loc[did["separates"]]
    if not len(sep):
        lg.log("  -> no pair separates from zero: this sample cannot "
               "establish that the gap differs by age. Report the gaps with "
               "their intervals and say the comparison is inconclusive.")
    else:
        for r in sep.itertuples():
            lg.log(f"  -> {r.group_a} vs {r.group_b}: gap differs by "
                   f"{r.did_pp:.1f} pp (95% CI {r.lo_pp:.1f} to {r.hi_pp:.1f})")
        lg.log("  Uncorrected across pairs; treat a single clearing pair as "
               "suggestive.")


# ---------------------------------------------------------------------- main
def main() -> None:
    lg.OUT.mkdir(parents=True, exist_ok=True)

    dfA = add_age_fields(lg.prep())
    dfB = add_age_fields(lg.prep_survey(lg.prep()))

    elig_tab = eligibility_diagnostic(dfA)

    dfA_e = dfA.loc[dfA["eligible_next_year"].eq(1)].copy()
    dfB_e = dfB.loc[dfB["eligible_next_year"].eq(1)].copy()
    lg.log(f"\nA: {len(dfA):,} -> {len(dfA_e):,} eligible "
           f"({100*len(dfA_e)/len(dfA):.1f}%)")
    lg.log(f"B: {len(dfB):,} -> {len(dfB_e):,} eligible "
           f"({100*len(dfB_e)/len(dfB):.1f}%)")

    refit, _ = refit_primary(dfA, dfA_e)

    # ---- interaction 1: hours x age, on the full administrative frame ----
    lg.log("\n=== 3. HOURS x AGE (frame A) ===")
    res_h, used_h = interaction(dfA_e, "A_hours_x_age", HOURS)
    ph = adjusted_probs(res_h, used_h, "hours_band", lg.HOURS_LABELS,
                        by="age_group")
    ch, boots_h = prob_contrast(res_h, used_h, "hours_band",
                                lg.HOURS_LABELS[0], "126-149")
    did_h = contrast_difference(ch, boots_h)

    # ---- interaction 2: first choice x age, on the survey frame ----------
    lg.log("\n=== 4. FIRST CHOICE x AGE (frame B) ===")
    gates = lg.check_gates(dfB_e)
    res_f, used_f = interaction(
        dfB_e, "B_firstchoice_x_age", HEADLINE,
        extra_terms=[i for i in lg.SURVEY_ITEMS if i != HEADLINE] + gates)
    pf = adjusted_probs(res_f, used_f, HEADLINE, [0, 1], by="age_group")
    cf_, boots_f = prob_contrast(res_f, used_f, HEADLINE, 0, 1)
    did_f = contrast_difference(cf_, boots_f)

    # ---- adjusted probabilities -------------------------------------------
    ph.to_csv(lg.OUT / "adjusted_probs_hours.csv", index=False)
    pf.to_csv(lg.OUT / "adjusted_probs_firstchoice.csv", index=False)
    lg.log("\nadjusted probabilities, hours x age:")
    lg.log(ph.to_string(index=False, float_format=lambda v: f"{v:,.4f}"))
    lg.log("\nadjusted probabilities, first choice x age:")
    lg.log(pf.to_string(index=False, float_format=lambda v: f"{v:,.4f}"))

    # ---- contrasts -------------------------------------------------------
    ch.to_csv(lg.OUT / "contrast_hours.csv", index=False)
    cf_.to_csv(lg.OUT / "contrast_firstchoice.csv", index=False)
    lg.log(f"\nprobability contrasts, pp ({lg.HOURS_LABELS[0]} -> 126-149 hours):")
    lg.log(ch.to_string(index=False, float_format=lambda v: f"{v:,.2f}"))
    lg.log("\nprobability contrasts, pp (first choice no -> yes):")
    lg.log(cf_.to_string(index=False, float_format=lambda v: f"{v:,.2f}"))

    # ---- heterogeneity test ------------------------------------------------
    did_h.to_csv(lg.OUT / "did_hours.csv", index=False)
    did_f.to_csv(lg.OUT / "did_firstchoice.csv", index=False)
    report_did(did_h, "hours gap by age group")
    report_did(did_f, "first-choice gap by age group")

    # ---- appendix ---------------------------------------------------------
    ors = []
    for res, lab in [(res_h, "A_hours_x_age"), (res_f, "B_firstchoice_x_age")]:
        ci = res.conf_int()
        ors.append(pd.DataFrame({
            "fit": lab, "term": res.params.index,
            "or": np.exp(res.params.values),
            "or_lo": np.exp(ci[0].values), "or_hi": np.exp(ci[1].values),
            "p": res.pvalues.values,
        }))
    pd.concat(ors).to_csv(lg.OUT / "interaction_or.csv", index=False)

    elig_tab.to_csv(lg.OUT / "age_eligibility.csv", index=False)
    refit.to_csv(lg.OUT / "primary_refit.csv", index=False)

    # ---- plots ------------------------------------------------------------
    plot_probs(ph, "hours_band", lg.HOURS_LABELS,
               "Full hours close the age gap",
               "Adjusted probability of reapplying · eligible participants "
               "only · 95% CI",
               "hours_by_age.png", "Hours worked in index summer")
    plot_probs(pf.assign(**{HEADLINE: pf[HEADLINE].map({0: "No", 1: "Yes"})}),
               HEADLINE, ["No", "Yes"],
               "Does placement match matter more with age?",
               "Adjusted probability of reapplying · survey respondents "
               "only · wide intervals are the small n",
               "firstchoice_by_age.png", "Got first-choice placement")

    (lg.OUT / "age_analysis_log.txt").write_text("\n".join(lg._LOG),
                                                 encoding="utf-8")
    print(f"\nwrote {lg.OUT}")


if __name__ == "__main__":
    main()