from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests

import sys

sys.path.insert(0, "/Users/sonia/Documents/SYEP")
from models import lg

ROPE = 1.20
COMPOSITE_BLOCKS = ["benefit", "expect"]
HEADLINE = "expect_first_choice"


# ------------------------------------------------------------------ helpers
def match_params(res, variables: list[str]) -> list[str]:
    """
    Parameter names belonging to a set of variables.

    Matched by name rather than hand-typed into a restriction string, because
    the patsy names carry brackets, quotes and commas and a typo in a string
    restriction either fails loudly or, worse, tests the wrong contrast.
    """
    names = list(res.params.index)
    out = []
    for v in variables:
        hits = [n for n in names if n == v or n.startswith(f"{v}[")]
        if not hits:
            raise KeyError(f"no fitted parameter matches {v!r}; have {names}")
        out.extend(hits)
    return out


def wald_block(res, variables: list[str]) -> dict:
    """Joint Wald test that every coefficient in `variables` is zero.

    Reads res.cov_params(), so it inherits whatever covariance estimator the
    fit used -- provider-clustered, under lg's default.
    """
    params = match_params(res, variables)
    names = list(res.params.index)
    R = np.zeros((len(params), len(names)))
    for i, p in enumerate(params):
        R[i, names.index(p)] = 1.0
    test = res.wald_test(R, use_f=False, scalar=True)
    return {"df": len(params),
            "chi2": float(np.squeeze(test.statistic)),
            "p": float(np.squeeze(test.pvalue))}


def kr20(X: pd.DataFrame) -> float:
    """Cronbach's alpha; on binary items this is KR-20."""
    X = X.dropna()
    k = X.shape[1]
    if k < 2 or len(X) < 3:
        return float("nan")
    total_var = X.sum(axis=1).var(ddof=1)
    if total_var <= 0:
        return float("nan")
    return float(k / (k - 1) * (1 - X.var(ddof=1).sum() / total_var))


def equivalence_bound(beta: float, se: float, alpha: float = 0.05) -> float:
    """TOST bound: 'we can rule out effects larger than this'."""
    z = norm.ppf(1 - alpha)
    return float(np.exp(max(abs(beta - z * se), abs(beta + z * se))))


def summarise(res, term: str) -> dict:
    b, se = float(res.params[term]), float(res.bse[term])
    return {"or": float(np.exp(b)),
            "or_lo": float(np.exp(b - 1.959964 * se)),
            "or_hi": float(np.exp(b + 1.959964 * se)),
            "p": float(res.pvalues[term]),
            "rule_out_above": equivalence_bound(b, se)}


def bh(p: list[float]) -> list[float]:
    return list(multipletests(p, alpha=0.05, method="fdr_bh")[1])


def verdict(row) -> str:
    if row["q"] < 0.05:
        return "effect detected"
    if row["rule_out_above"] <= ROPE:
        return f"ruled out above {ROPE:.2f}x"
    return "inconclusive"


# ------------------------------------------------------------------ analyses
def gate_sensitivity(dfB: pd.DataFrame, kept_gates: list[str]) -> pd.DataFrame:
    """Refit with every gate forced in, and see whether the headline moves."""
    base = lg.CORE + lg.QUALITY + lg.SURVEY_ITEMS
    forced = [g for g in lg.SURVEY_GATES if dfB[g].nunique() > 1]

    lg.log("\n=== 1. GATE SENSITIVITY ===")
    lg.log(f"gates kept by check_gates: {kept_gates or 'none'}")
    lg.log(f"gates forced in for this check: {forced}")

    res_spec, _ = lg.fit(dfB, base + kept_gates, "B_as_specified")
    res_gates, _ = lg.fit(dfB, base + forced, "B_all_gates")

    rows = []
    for item in lg.SURVEY_ITEMS:
        a, b = summarise(res_spec, item), summarise(res_gates, item)
        la, lb = np.log(a["or"]), np.log(b["or"])
        rows.append({"term": item,
                     "or_as_specified": a["or"], "p_as_specified": a["p"],
                     "or_all_gates": b["or"], "p_all_gates": b["p"],
                     "pct_change_log_or": 100 * (lb - la) / abs(la)
                     if la != 0 else np.nan})
    out = pd.DataFrame(rows)

    h = out.loc[out["term"] == HEADLINE].iloc[0]
    lg.log(f"\n{HEADLINE}: OR {h['or_as_specified']:.3f} "
           f"(p={h['p_as_specified']:.3g})  ->  {h['or_all_gates']:.3f} "
           f"(p={h['p_all_gates']:.3g})  with all gates forced in")

    was_sig = h["p_as_specified"] < 0.05
    lost_sig = was_sig and h["p_all_gates"] >= 0.05
    moved = abs(np.log(h["or_all_gates"]) - np.log(h["or_as_specified"])) > 0.10

    if not was_sig:
        lg.log("  headline was not significant as specified; gate sensitivity "
               "is not the binding issue here.")
    elif lost_sig or moved:
        lg.log("  !! the headline is sensitive to how block non-response is "
               "modelled, so it cannot be presented as a clean "
               "placement-match effect yet. Refit Model B restricted to "
               "expect-block answerers and see which version survives.")
    else:
        lg.log("  headline is stable to gate specification.")
    return out


def block_wald(res) -> pd.DataFrame:
    lg.log("\n=== 2. BLOCK WALD TESTS ===")
    rows = [{"block": b, **wald_block(res, items)}
            for b, items in lg.BLOCKS.items()]
    residual = [i for i in lg.BLOCKS["expect"] if i != HEADLINE]
    rows.append({"block": f"expect minus {HEADLINE}",
                 **wald_block(res, residual)})

    rows.append({"block": "ALL survey items",
                 **wald_block(res, lg.SURVEY_ITEMS)})
    out = pd.DataFrame(rows)
    out["q"] = bh(out["p"].tolist())
    lg.log(out.to_string(index=False, float_format=lambda v: f"{v:,.4g}"))
    return out


def composites(dfB: pd.DataFrame, kept_gates: list[str]) -> pd.DataFrame:
    """Count of items selected per block, replacing the individual items."""
    lg.log("\n=== 3. COMPOSITE COUNTS ===")
    df = dfB.copy()
    names, alphas = [], {}
    for block in COMPOSITE_BLOCKS:
        items = lg.BLOCKS[block]
        name = f"n_{block}_selected"
        df[name] = df[items].sum(axis=1)
        answered = df[f"{block}_answered"].eq(1)
        alphas[block] = kr20(df.loc[answered, items])
        names.append(name)

    other = [i for b, items in lg.BLOCKS.items()
             for i in items if b not in COMPOSITE_BLOCKS]
    res, _ = lg.fit(df, lg.CORE + lg.QUALITY + names + other + kept_gates,
                    "B_composite")

    rows = [{"block": b, "term": n, "n_items": len(lg.BLOCKS[b]),
             "alpha_kr20": alphas[b], **summarise(res, n)}
            for b, n in zip(COMPOSITE_BLOCKS, names)]
    out = pd.DataFrame(rows)
    out["q"] = bh(out["p"].tolist())
    out["verdict"] = out.apply(verdict, axis=1)
    lg.log(out.to_string(index=False, float_format=lambda v: f"{v:,.3f}"))

    for block, a in alphas.items():
        if a == a and a < 0.6:
            lg.log(f"  ! KR-20 = {a:.2f} for '{block}': the items do not "
                   "scale together, so this count is a tally and not an "
                   "index. Lead with the block Wald test instead.")
    return out


def main() -> None:
    lg.OUT.mkdir(parents=True, exist_ok=True)

    dfA = lg.prep()
    dfB = lg.prep_survey(dfA)
    kept_gates = lg.check_gates(dfB)

    lg.log(f"\nB: {len(dfB):,} participants, "
           f"return rate {dfB['returned'].mean():.3f}")

    gs = gate_sensitivity(dfB, kept_gates)

    res_items, _ = lg.fit(
        dfB, lg.CORE + lg.QUALITY + lg.SURVEY_ITEMS + kept_gates,
        "B_survey_replicate")
    bw = block_wald(res_items)
    cp = composites(dfB, kept_gates)

    gs.to_csv(lg.OUT / "gate_sensitivity.csv", index=False)
    bw.to_csv(lg.OUT / "block_wald.csv", index=False)
    cp.to_csv(lg.OUT / "composites.csv", index=False)

    lg.log("\nThese are SECONDARY analyses. They are BH-corrected within each "
           "table above and must not be merged into the pre-specified family "
           "in coefs.csv. Label them exploratory wherever they appear.")
    (lg.OUT / "block_tests_log.txt").write_text("\n".join(lg._LOG),
                                                encoding="utf-8")
    print(f"\nwrote {lg.OUT}")


if __name__ == "__main__":
    main()