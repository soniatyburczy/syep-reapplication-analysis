"""
lg.py — logistic regression, reapplication analysis.

Unit of observation
-------------------
Frame A (primary):   analysis_frame_worksite.csv, ~130,417 person-years.
Frame B (secondary): built in-house from survey_features_wide.csv.

A is the full administrative risk set. B contains at most one row per person:
the survey record attached to that person's earliest A person-year in the
2022-2025 window. B is selected here rather than consumed pre-selected,
because "earliest" must be defined against A's post-filter risk set -- the
upstream *_earliest_wide.csv extract defined it against its own row set and
therefore disagreed. Because survey response is self-selected, B is not
assumed to estimate the same population parameters as A.

Four fits
---------
1. A_full: all A person-years, with provider-clustered standard errors.
2. A_person_clustered: the same specification on A, but with person-clustered
   standard errors as a dependence sensitivity check. This changes inference,
   not the fitted mean structure; it is not a person fixed-effect model.
3. A_survey_subsample: the administrative model restricted to the exact
   (person, Year) rows represented in B. This is the bridge model that shows
   how the A specification behaves in B's selected analysis population.
4. B_survey: those same person-years plus the survey feature blocks.

Key design decisions
--------------------
* `worked_zero_hours` is not a regression term. It is identical to the "0"
  level of `hours_band`, so including both would make the design rank deficient.
* The `hours_band` reference is 126-149 hours, so each hours coefficient is a
  comparison against a full-time working band rather than non-participation.
* Demographics (`race`, `gender`) enter every fit as categorical
  terms. Their reference level is resolved at runtime to the modal level of the
  provider-complete A frame rather than hardcoded, and the same reference is
  reused across fits so coefficients are directly comparable; if a reference
  level is absent from a smaller frame the fallback is logged rather than
  silently substituted. Missing demographic values are materialised as an
  explicit level (see DEMOGRAPHIC_MISSING_LABEL) so that unrecorded
  demographics do not listwise-delete otherwise complete administrative rows —
  the same non-response treatment already used for survey items. That level is
  a data-availability indicator, not a demographic group, and should not be
  interpreted as one.
* Survey item non-response is materialised as zero so one unanswered block does
  not listwise-delete the row across every survey block. Block-answer gates are
  retained only when they contain information not already encoded by the item
  pattern.
* Benjamini-Hochberg correction is applied separately within each fit. For A
  fits, all non-intercept coefficients form the family; for B_survey, only the
  pre-specified survey terms form the family, while administrative adjustment
  covariates (including demographics) are reported without entering that
  correction. Whether demographics join the A-fit family is controlled by
  DEMOGRAPHICS_IN_BH_FAMILY, since that choice changes the q values of every
  other A-fit term.
* `selection.csv` compares responders and non-responders on each person's
  earliest administrative person-year in the 2022-2025 survey window. It does
  not label every later year of a responder as a response year.

Pre-fit structural checks
-------------------------
* A must contain at most one row per (person, Year).
* The survey file must contain at most one row per (person, Year); a violation
  is a hard error, since it means the extract is not a person-year file.
* Survey participants with no A person-year in 2022-2025 are outside the
  administrative risk set and are dropped, with a logged accounting of how
  many and whether they appear in A at all. A 100% non-match rate is still a
  hard error (join/key problem).
* B rows off the person's earliest A year are dropped under the default
  REQUIRE_SURVEY_AT_EARLIEST_A_YEAR policy, with the recoverable count logged.
  diagnostics.txt carries a survey-year x earliest-A-year cross-tab so the
  shape of that loss is visible rather than assumed.
* Survey gates must be binary; survey items must be numeric after preprocessing.
* Demographic level support (cell counts and return rates per level) is written
  to diagnostics.txt, since sparse levels are the likely separation source once
  demographics are in the design.
* Realised design rank/conditioning and simple binary separation warnings are
  written to diagnostics.txt.

Outputs to data/to_use/models/:
    coefs.csv         tidy coefficients, OR, CI, p, BH q within family
    fitstats.csv      n, llf, pseudo-R2, AIC, convergence per fit
    selection.csv     responders vs non-responders on earliest A person-years
    diagnostics.txt   structure checks, gate redundancy, rank, separation
    residuals_A.csv   A_full person-year fitted values / response residuals
"""

from __future__ import annotations

from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd
import patsy
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

_HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_HERE))

import stats.cohort_table as ct

ROOT = ct.ROOT
OUT = ROOT / "data/to_use/models"
SURVEY_PATH = ROOT / "data/to_use/survey_features_wide.csv"
ID = ct.ID

SURVEY_START_YEAR = 2022
SURVEY_END_YEAR = 2025

# How to pick the single B row per person from the full survey person-year file.
#
# True  — keep the survey record only if it sits on the person's earliest A
#         person-year in the window. Smaller B, but B stays a strict subset of
#         the earliest-person-year frame, so `returned` is never conditioned on
#         prior SYEP participation and the A_survey_subsample bridge model
#         compares like with like.
# False — keep each person's earliest *surveyed* window year even when an
#         earlier unsurveyed A year exists. Larger B, but for those people the
#         outcome is measured after at least one prior participation year,
#         which is a different estimand from the first-observation return.
#
# True is the setting consistent with the pre-specified first-observed-year-only
# design; flip it only as a documented sensitivity analysis.
REQUIRE_SURVEY_AT_EARLIEST_A_YEAR = True

# Whether demographic coefficients join the BH family in the A fits.
#
# True  — demographics are treated as pre-specified predictors of interest, on
#         the reading that "which participant characteristics predict return"
#         includes them. They enter the same family as the other A terms.
#         Consequence: the family grows by one term per non-reference level, so
#         every previously-reported A q value gets larger. Any q you have
#         already quoted from an earlier run is superseded.
# False — demographics are adjustment covariates only. They are estimated and
#         reported with p and CI, but excluded from the BH family, leaving the
#         q values of the other A terms exactly as they were before
#         demographics were added.
DEMOGRAPHICS_IN_BH_FAMILY = True

CORE = [
    "bs(age_on_start, df=3)",
    'C(hours_band, Treatment(reference="126-149"))',
    "C(Year)",
    "C(borough)",
]

QUALITY = [
    "is_no_show",
    "worksite_matched",
]

# Demographic terms are built at runtime by resolve_demographic_terms() so the
# Treatment() reference is an observed level rather than a hardcoded guess.
DEMOGRAPHIC_COLS = ["race", "gender"]
DEMOGRAPHIC_MISSING_LABEL = "Missing/unrecorded"

SURVEY_ITEMS = [
    "cf_paid_alternative",
    "cf_unpaid_activity",
    "cf_leisure_only",
    "cf_no_plans",
    "expect_app_clarity",
    "expect_first_choice",
    "expect_knew_what_to_expect",
    "benefit_career_clarity",
    "benefit_job_readiness",
    "benefit_self_efficacy",
    "benefit_mentor_relationship",
    "benefit_money_management",
]

SURVEY_GATES = ["cf_answered", "expect_answered", "benefit_answered"]

BLOCKS = {
    "cf": ["cf_paid_alternative", "cf_unpaid_activity",
           "cf_leisure_only", "cf_no_plans"],
    "expect": ["expect_app_clarity", "expect_first_choice",
               "expect_knew_what_to_expect"],
    "benefit": ["benefit_career_clarity", "benefit_job_readiness",
                "benefit_self_efficacy", "benefit_mentor_relationship",
                "benefit_money_management"],
}

CLUSTER = "provider"
SEPARATION_MIN_CELL = 20
THIN_CELL = 50
HOURS_BINS = [-0.001, 0, 25, 75, 125, 149.99, np.inf]
HOURS_LABELS = ["0", "1-25", "26-75", "76-125", "126-149", "150 (cap)"]

# Strings that mean "no value recorded" once a demographic column has been read
# as text. Deliberately narrow: a literal administrative category such as
# "Unknown" or "Prefer not to say" is a real answer and is left as its own
# level.
_BLANK_TOKENS = {"", "na", "n/a", "nan", "none", "null"}

_C_TERM = re.compile(r"^C\(\s*([A-Za-z_]\w*)\s*[,)]")

_LOG: list[str] = []


def log(msg: str = "") -> None:
    print(msg)
    _LOG.append(msg)


def _raise_with_rows(message: str, rows: pd.DataFrame,
                     max_rows: int = 20) -> None:
    """Raise a readable validation error with a small sample of bad rows."""
    sample = rows.head(max_rows).to_string(index=False)
    extra = len(rows) - min(len(rows), max_rows)
    suffix = f"\n... plus {extra:,} more row(s)" if extra > 0 else ""
    raise ValueError(f"{message}\n{sample}{suffix}")


def _term_column(term: str) -> str | None:
    """The underlying dataframe column for a bare or C(...) term, if any.

    Returns None for terms whose variation cannot be checked by looking at a
    single column (e.g. spline bases, interactions), which are left alone.
    """
    if term.isidentifier():
        return term
    m = _C_TERM.match(term)
    return m.group(1) if m else None


def prep_demographics(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise demographic columns to strings with an explicit missing level.

    Left as NaN, these columns would be dropped row-wise by the formula-level
    missing handling, quietly shrinking every fit and changing which population
    the administrative model describes. Materialising the gap keeps the row and
    makes the gap an estimable, visible term instead.
    """
    log("\n--- demographics ---")
    for col in DEMOGRAPHIC_COLS:
        if col not in df.columns:
            raise ValueError(
                f"Demographic column '{col}' is not in frame A. Available "
                f"columns include: {sorted(df.columns)[:40]}"
            )

        raw = df[col]
        text = raw.astype("string").str.strip()
        blank = text.isna() | text.str.lower().isin(_BLANK_TOKENS)
        n_blank = int(blank.sum())

        df[col] = text.mask(blank, DEMOGRAPHIC_MISSING_LABEL).astype(str)
        log(
            f"{col}: {df[col].nunique():,} level(s); "
            f"{n_blank:,} rows ({100 * n_blank / len(df):.1f}%) coded as "
            f"'{DEMOGRAPHIC_MISSING_LABEL}'"
        )

    return df


def resolve_demographic_terms(
    df: pd.DataFrame,
    label: str,
    refs: dict[str, str] | None = None,
) -> tuple[list[str], dict[str, str]]:
    """Build C(col, Treatment(reference=...)) terms against observed levels.

    `refs` carries the references chosen on the primary frame so that every fit
    contrasts against the same baseline. A reference missing from a smaller
    frame falls back to that frame's modal level and says so, because a silent
    fallback would make two fits' coefficients look comparable when they are
    measured from different baselines.
    """
    terms: list[str] = []
    chosen: dict[str, str] = {}

    for col in DEMOGRAPHIC_COLS:
        if col not in df.columns:
            raise ValueError(f"Demographic column '{col}' missing in {label}.")

        counts = df[col].value_counts()
        if len(counts) < 2:
            log(f"[{label}] dropped C({col}): only "
                f"{len(counts)} level present in this frame")
            continue

        ref = refs.get(col) if refs else None
        if ref is None:
            ref = str(counts.idxmax())
        elif ref not in counts.index:
            fallback = str(counts.idxmax())
            log(f"[{label}] reference level '{ref}' for {col} does not occur "
                f"in this frame; falling back to '{fallback}' -- {col} "
                f"coefficients are NOT comparable to the other fits")
            ref = fallback

        if '"' in ref:
            raise ValueError(
                f"Reference level for {col} contains a double quote and "
                f"cannot be embedded in a patsy formula: {ref!r}"
            )

        chosen[col] = ref
        terms.append(f'C({col}, Treatment(reference="{ref}"))')

    if terms:
        log(f"[{label}] demographic references: " + ", ".join(
            f"{c}='{r}'" for c, r in chosen.items()))

    return terms, chosen


def prep(path: Path | None = None) -> pd.DataFrame:
    """Load frame A, attach the outcome, and build derived columns."""
    panel = pd.read_csv(path or ct.PANEL)
    lookup = pd.read_csv(ct.LOOKUP, usecols=[ID, "Year", "service_option"])
    df = ct.build_outcome(panel, lookup)
    df["Year"] = pd.to_numeric(df["Year"], errors="raise").astype(int)

    dup = df.duplicated([ID, "Year"], keep=False)
    if dup.any():
        _raise_with_rows(
            "Frame A contains duplicate person-years; expected one row per "
            f"({ID}, Year).",
            df.loc[dup, [ID, "Year"]].sort_values([ID, "Year"]),
        )

    hrs = pd.to_numeric(df["total_hours_paid"], errors="coerce")
    df["hours_band"] = pd.cut(hrs, bins=HOURS_BINS, labels=HOURS_LABELS)
    df["worked_zero_hours"] = hrs.eq(0).astype(int)
    df["worksite_matched"] = df[ct.WS_STATUS].notna().astype(int)
    df["is_no_show"] = df[ct.WS_NO_SHOW].fillna(False).astype(int)

    df = prep_demographics(df)

    return df


def earliest_person_years(dfA: pd.DataFrame) -> pd.DataFrame:
    """One row per person: their earliest A person-year in 2022-2025."""
    window = dfA.loc[
        dfA["Year"].between(SURVEY_START_YEAR, SURVEY_END_YEAR)
    ].copy()

    return (
        window.sort_values([ID, "Year"])
        .drop_duplicates(ID, keep="first")
        .copy()
    )


def survey_coverage_report(survey: pd.DataFrame,
                           earliest: pd.DataFrame) -> None:
    """Cross-tab each person's earliest A year against their surveyed years."""
    merged = survey[[ID, "Year"]].merge(earliest, on=ID, how="inner")
    tab = pd.crosstab(merged["earliest_A_year"], merged["Year"])
    log("\n--- survey year vs earliest A year (person-years) ---")
    log(tab.to_string())


def build_survey_frame(dfA: pd.DataFrame,
                       path: Path | None = None) -> pd.DataFrame:
    """Select B in-house from the full survey person-year file."""
    survey = pd.read_csv(path or SURVEY_PATH)

    required = {ID, "Year", *SURVEY_GATES, *SURVEY_ITEMS}
    missing = sorted(required - set(survey.columns))
    if missing:
        raise ValueError(f"Survey frame is missing required columns: {missing}")

    survey = survey.copy()
    survey["Year"] = pd.to_numeric(survey["Year"], errors="raise").astype(int)

    dup = survey.duplicated([ID, "Year"], keep=False)
    if dup.any():
        _raise_with_rows(
            "Survey file contains more than one row per (person, Year); "
            "expected one survey record per person-year.",
            survey.loc[dup, [ID, "Year"]].sort_values([ID, "Year"]),
        )

    n_raw_rows = len(survey)
    n_raw_people = survey[ID].nunique()

    survey = survey.loc[
        survey["Year"].between(SURVEY_START_YEAR, SURVEY_END_YEAR)
    ].copy()
    log(
        f"\n--- B construction ---\n"
        f"survey file: {n_raw_rows:,} person-years / {n_raw_people:,} people; "
        f"{len(survey):,} person-years inside "
        f"{SURVEY_START_YEAR}-{SURVEY_END_YEAR}"
    )

    earliest = earliest_person_years(dfA)[[ID, "Year"]].rename(
        columns={"Year": "earliest_A_year"}
    )

    survey = survey.merge(earliest, on=ID, how="left", validate="m:1")

    no_admin = survey["earliest_A_year"].isna()
    if no_admin.all():
        raise ValueError(
            "No survey person-year matched any administrative person-year in "
            f"{SURVEY_START_YEAR}-{SURVEY_END_YEAR}. This indicates a key "
            f"problem (check {ID} dtypes/values in both frames), not attrition."
        )

    if no_admin.any():
        dropped = survey.loc[no_admin, [ID, "Year"]]
        in_A_any_year = int(dropped[ID].drop_duplicates().isin(dfA[ID]).sum())
        n_people = dropped[ID].nunique()
        log(
            f"dropped {len(dropped):,} survey person-years "
            f"({n_people:,} people) with no A person-year in the window; "
            f"{in_A_any_year:,} of those people appear in A in some other "
            f"year, {n_people - in_A_any_year:,} appear nowhere in A"
        )
        survey = survey.loc[~no_admin].copy()

    survey["earliest_A_year"] = survey["earliest_A_year"].astype("Int64")

    survey_coverage_report(survey, earliest)

    at_earliest = survey["Year"].eq(survey["earliest_A_year"])
    n_people_linked = survey[ID].nunique()

    if REQUIRE_SURVEY_AT_EARLIEST_A_YEAR:
        keep = survey.loc[at_earliest].copy()
        lost = n_people_linked - keep[ID].nunique()
        recoverable = (
            survey.loc[~survey[ID].isin(keep[ID]), ID].nunique()
        )
        log(
            f"policy=earliest-A-year-only: kept {len(keep):,} people; "
            f"dropped {lost:,} people whose survey year is later than their "
            f"earliest A year in the window "
            f"({recoverable:,} would be recovered by setting "
            f"REQUIRE_SURVEY_AT_EARLIEST_A_YEAR = False, at the cost of "
            f"measuring `returned` after prior participation)"
        )
    else:
        keep = (
            survey.sort_values([ID, "Year"])
            .drop_duplicates(ID, keep="first")
            .copy()
        )
        off_diagonal = int((~keep["Year"].eq(keep["earliest_A_year"])).sum())
        log(
            f"policy=earliest-surveyed-year (SENSITIVITY): kept "
            f"{len(keep):,} people; {off_diagonal:,} of them are observed at "
            f"a year later than their earliest A year, so their outcome is "
            f"conditional on prior participation"
        )

    if keep[ID].duplicated().any():
        raise RuntimeError("B construction produced duplicate persons.")
    if REQUIRE_SURVEY_AT_EARLIEST_A_YEAR and \
            not keep["Year"].eq(keep["earliest_A_year"]).all():
        raise RuntimeError(
            "B construction kept a row off the earliest A year despite the "
            "earliest-A-year-only policy."
        )

    keep = keep.drop(columns="earliest_A_year")
    log(f"B: {len(keep):,} unique participants")
    return keep


def _coerce_numeric(series: pd.Series, name: str,
                    fill_missing: int | None = None) -> pd.Series:
    """Coerce a survey feature to numeric without silently zeroing bad text."""
    numeric = pd.to_numeric(series, errors="coerce")
    bad_text = series.notna() & numeric.isna()
    if bad_text.any():
        values = series.loc[bad_text].astype(str).value_counts().head(10)
        raise ValueError(
            f"{name} contains non-numeric values that would otherwise be "
            f"silently coerced to missing:\n{values.to_string()}"
        )

    if fill_missing is not None:
        numeric = numeric.fillna(fill_missing)

    return numeric.astype(int) if not numeric.isna().any() else numeric


def _coerce_gate(series: pd.Series, name: str) -> pd.Series:
    """Coerce a block-response gate to binary 0/1."""
    numeric = _coerce_numeric(series, name, fill_missing=0)
    observed = set(numeric.dropna().unique())
    if not observed.issubset({0, 1}):
        raise ValueError(
            f"{name} must be binary 0/1 after preprocessing; found "
            f"{sorted(observed)}"
        )
    return numeric.astype(int)


def prep_survey(dfA: pd.DataFrame, path: Path | None = None) -> pd.DataFrame:
    """Build B from the survey file, merge onto A, materialise non-response."""
    survey = build_survey_frame(dfA, path)

    df = dfA.merge(survey, on=[ID, "Year"], how="inner", validate="1:1")
    if len(df) != len(survey):
        raise RuntimeError(
            "Survey merge lost rows despite in-house B construction: "
            f"survey={len(survey):,}, merged={len(df):,}."
        )

    for gate in SURVEY_GATES:
        df[gate] = _coerce_gate(df[gate], gate)

    for col in SURVEY_ITEMS:
        df[col] = _coerce_numeric(df[col], col, fill_missing=0)

    df["n_blocks_answered"] = df[SURVEY_GATES].sum(axis=1)
    return df


def survey_admin_subsample(dfA: pd.DataFrame,
                           dfB: pd.DataFrame) -> pd.DataFrame:
    """Administrative A rows for the exact (person, Year) keys present in B."""
    keys = dfB[[ID, "Year"]].drop_duplicates()
    out = dfA.merge(keys, on=[ID, "Year"], how="inner", validate="1:1")

    if len(out) != len(keys):
        raise RuntimeError(
            "A survey subsample does not match B's person-year keys: "
            f"B keys={len(keys):,}, matched A rows={len(out):,}."
        )

    return out


def check_gates(df: pd.DataFrame) -> list[str]:
    """Return gates that carry information the item pattern does not."""
    keep = []
    log("\n--- gate redundancy ---")
    for block, items in BLOCKS.items():
        gate = f"{block}_answered"
        ans = df[gate].astype(bool)
        lo = int(df.loc[ans, items].sum(axis=1).min()) if ans.any() else 0
        hi = int(df.loc[~ans, items].sum(axis=1).max()) if (~ans).any() else 0
        redundant = lo >= 1 and hi == 0
        log(f"{gate}: min selected among answerers={lo}, "
            f"max among non-answerers={hi} -> "
            f"{'redundant, dropped' if redundant else 'retained'}")
        if not redundant:
            keep.append(gate)
    return keep


def check_design(df: pd.DataFrame, terms: list[str], label: str,
                 cluster: str | None = CLUSTER, quiet: bool = False) -> None:
    """Rank and conditioning of the design the model will actually fit."""
    frame, kept_terms = prepare_fit_frame(df, terms, label, cluster,
                                          quiet=True)
    n_dropped = len(terms) - len(kept_terms)
    _, X = patsy.dmatrices(
        "returned ~ " + " + ".join(kept_terms),
        frame,
        return_type="dataframe",
    )
    Xv = X.to_numpy(dtype=float)
    rank = np.linalg.matrix_rank(Xv)

    norms = np.linalg.norm(Xv, axis=0)
    norms[norms == 0] = 1.0
    cond_scaled = np.linalg.cond(Xv / norms)

    log(f"\n--- design [{label}] ---")
    log(f"shape={X.shape} rank={rank} "
        f"{'RANK DEFICIENT' if rank < X.shape[1] else 'full rank'}"
        + (f" (after dropping {n_dropped} constant term(s); see fit log)"
           if n_dropped else ""))
    log(f"scaled condition index={cond_scaled:,.1f}"
        f"{'  (collinearity worth checking)' if cond_scaled > 30 else ''}")


def check_separation(df: pd.DataFrame, cols: list[str],
                     outcome: str = "returned") -> None:
    """Flag simple binary cells with empty/near-empty outcome support.

    This is a univariate screening diagnostic, not a proof that multivariable
    quasi-separation is absent.
    """
    log("\n--- separation and thin cells ---")
    separated, thin = 0, 0
    for c in cols:
        if c not in df.columns or df[c].nunique(dropna=True) > 2:
            continue
        g = df.groupby(c, observed=True)[outcome].agg(["size", "sum"])
        g["rate"] = g["sum"] / g["size"]

        is_sep = bool(g["rate"].isin([0.0, 1.0]).any()
                      or (g["size"] < SEPARATION_MIN_CELL).any())
        is_thin = bool((g["size"] < THIN_CELL).any())
        if not (is_sep or is_thin):
            continue

        separated += int(is_sep)
        thin += int(is_thin and not is_sep)
        tag = "SEPARATION" if is_sep else "thin"
        log(f"[{tag}] {c}: " + " | ".join(
            f"{i}: n={r['size']:,.0f} rate={r['rate']:.3f}"
            for i, r in g.iterrows()))

    if not (separated or thin):
        log("none")
    else:
        log(f"{separated} term(s) separated/near-empty; "
            f"{thin} additional term(s) with a cell under {THIN_CELL} rows. "
            f"Thin-cell coefficients are estimable but carry wide intervals; "
            f"report them with the CI or not at all.")


def check_category_support(df: pd.DataFrame, cols: list[str], label: str,
                           outcome: str = "returned") -> None:
    """Per-level cell counts and return rates for multi-level categoricals.

    check_separation only inspects binary columns, so without this a sparse
    race or gender level would reach the fit unannounced and surface as an
    implausible odds ratio with an interval spanning orders of magnitude.
    """
    log(f"\n--- categorical level support [{label}] ---")
    for col in cols:
        if col not in df.columns:
            continue
        g = (
            df.groupby(col, observed=True)[outcome]
            .agg(["size", "sum"])
            .sort_values("size", ascending=False)
        )
        g["rate"] = g["sum"] / g["size"]
        for level, row in g.iterrows():
            n = float(row["size"])
            rate = float(row["rate"])
            if rate in (0.0, 1.0) or n < SEPARATION_MIN_CELL:
                tag = "  [SEPARATION]"
            elif n < THIN_CELL:
                tag = "  [thin]"
            else:
                tag = ""
            log(f"{col}={level!r}: n={n:,.0f} return_rate={rate:.3f}{tag}")


def provider_missingness(df: pd.DataFrame) -> None:
    """Where the cluster variable is missing, by year.

    Rows with no `provider` cannot enter a provider-clustered fit. If that
    missingness concentrates in particular years, the provider-clustered fits
    are estimated on a differently-composed sample than the raw frame, which
    is a scope caveat rather than a bug -- but it has to be stated.
    """
    log("\n--- cluster (provider) coverage by year ---")
    tab = (
        df.assign(_missing=df[CLUSTER].isna())
        .groupby("Year")["_missing"]
        .agg(person_years="size", missing_provider="sum")
    )
    tab["pct_missing"] = (
        100 * tab["missing_provider"] / tab["person_years"]
    ).round(1)
    log(tab.to_string())

    overall = 100 * df[CLUSTER].isna().mean()
    log(f"overall: {overall:.1f}% of A person-years have no provider and are "
        f"excluded from every provider-clustered fit")


def drop_degenerate(df: pd.DataFrame, terms: list[str], label: str,
                    quiet: bool = False) -> list[str]:
    """Remove terms with no variation in this subset.

    A constant term is the intercept column times a scalar, so it is exactly
    collinear with the intercept: it costs a rank, sends the condition number to
    machine-precision scale, and can make the Hessian singular. This happens
    naturally for survey gates whose block every respondent answered, and for a
    demographic column that collapses to one level in a small subsample.
    """
    keep = []
    for t in terms:
        col = _term_column(t)
        if col is not None and col in df.columns \
                and df[col].nunique(dropna=True) < 2:
            if not quiet:
                log(f"[{label}] dropped {t}: constant in this subset")
            continue
        keep.append(t)
    return keep


def prepare_fit_frame(df: pd.DataFrame, terms: list[str], label: str,
                      cluster: str | None = CLUSTER,
                      quiet: bool = False) -> tuple[pd.DataFrame, list[str]]:
    """Rows and terms shared by `check_design` and `fit`.

    Single source of truth for what a fit actually sees, so the diagnostic
    cannot describe a different design from the one estimated.
    """
    frame = df.dropna(subset=[cluster]).copy() if cluster else df.copy()
    kept_terms = drop_degenerate(frame, terms, label, quiet=quiet)
    return frame, kept_terms


def fit(df: pd.DataFrame, terms: list[str], label: str,
        cluster: str | None = "provider"):
    """Fit one logit, optionally with cluster-robust standard errors.

    Cluster groups are aligned to the exact rows Patsy/statsmodels retain after
    formula-level missing-value deletion. This avoids a group-vector mismatch
    when adjustment covariates contain missing values.
    """
    cluster_ready, kept_terms = prepare_fit_frame(df, terms, label, cluster)
    formula = "returned ~ " + " + ".join(kept_terms)

    dropped_cluster = len(df) - len(cluster_ready)
    log(f"\n[{label}] starting n={len(df):,}; "
        f"{dropped_cluster:,} dropped on {cluster}")

    model = smf.logit(formula, data=cluster_ready, missing="drop")
    model_idx = model.data.row_labels
    used = cluster_ready.loc[model_idx].copy()
    dropped_formula = len(cluster_ready) - len(used)

    if cluster:
        res = model.fit(
            cov_type="cluster",
            cov_kwds={"groups": used[cluster]},
            maxiter=200,
            disp=False,
        )
    else:
        res = model.fit(maxiter=200, disp=False)

    log(f"formula_missing_drop={dropped_formula:,} "
        f"converged={res.mle_retvals['converged']} "
        f"n_used={int(res.nobs):,} pseudo_r2={res.prsquared:.4f}")
    return res, used


def tidy(res, label: str, family: list[str] | None = None,
         exclude: list[str] | None = None,
         alpha: float = 0.05) -> pd.DataFrame:
    """Odds ratios with BH q values computed within a pre-specified family.

    `exclude` removes prefixes from the family after it is formed, which is how
    a term can be estimated and reported without inflating everyone else's q.
    """
    ci = res.conf_int()
    t = pd.DataFrame({
        "fit": label,
        "term": res.params.index,
        "coef": res.params.values,
        "or": np.exp(res.params.values),
        "or_lo": np.exp(ci[0].values),
        "or_hi": np.exp(ci[1].values),
        "p": res.pvalues.values,
    })

    if family is None:
        mask = t["term"].ne("Intercept")
    else:
        mask = t["term"].str.startswith(tuple(family))

    if exclude:
        mask &= ~t["term"].str.startswith(tuple(exclude))

    t["in_family"] = mask
    t["q"] = np.nan
    if mask.any():
        t.loc[mask, "q"] = multipletests(
            t.loc[mask, "p"],
            alpha=alpha,
            method="fdr_bh",
        )[1]
    return t


def selection_table(dfA: pd.DataFrame, dfB: pd.DataFrame) -> pd.DataFrame:
    """Responders vs non-responders on earliest 2022-2025 A person-years."""
    a = earliest_person_years(dfA)
    a["responded"] = a[ID].isin(set(dfB[ID])).astype(int)

    cols = [
        "returned",
        "age_on_start",
        "total_hours_paid",
        "worked_zero_hours",
        "is_no_show",
        "worksite_matched",
    ]

    rows = []
    for col in cols:
        if col not in a.columns:
            continue
        g = a.groupby("responded")[col].mean()
        rows.append({
            "variable": col,
            "non_responders": round(float(g.get(0, np.nan)), 4),
            "responders": round(float(g.get(1, np.nan)), 4),
            "difference": round(float(g.get(1, np.nan) - g.get(0, np.nan)), 4),
        })

    # Demographics are categorical, so a mean is undefined; compare response
    # rates by level instead. Without this, differential survey response across
    # demographic groups -- the most likely form of selection here -- would be
    # invisible in the one table meant to expose selection.
    for col in DEMOGRAPHIC_COLS:
        if col not in a.columns:
            continue
        share = pd.crosstab(a[col], a["responded"], normalize="columns")
        for level, r in share.iterrows():
            rows.append({
                "variable": f"{col}={level}",
                "non_responders": round(float(r.get(0, np.nan)), 4),
                "responders": round(float(r.get(1, np.nan)), 4),
                "difference": round(float(r.get(1, np.nan) - r.get(0, np.nan)),
                                    4),
            })

    out = pd.DataFrame(rows)
    n = a["responded"].value_counts()
    out.attrs["n_non_responders"] = int(n.get(0, 0))
    out.attrs["n_responders"] = int(n.get(1, 0))
    out.attrs["n_earliest_person_years"] = len(a)
    return out


def residual_frame(res, used: pd.DataFrame) -> pd.DataFrame:
    """Response residuals on the person-years the model actually used."""
    idx = res.fittedvalues.index
    sub = used.loc[idx]
    p_hat = res.predict(sub)
    return pd.DataFrame({
        ID: sub[ID].to_numpy(),
        "Year": sub["Year"].to_numpy(),
        ct.WS_CLUSTER: sub[ct.WS_CLUSTER].to_numpy(),
        ct.WS_STATUS: sub[ct.WS_STATUS].to_numpy(),
        "returned": sub["returned"].to_numpy(),
        "p_hat": p_hat.to_numpy(),
        "resid": sub["returned"].to_numpy() - p_hat.to_numpy(),
    })


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    dfA = prep()
    dfB = prep_survey(dfA)
    dfA_survey = survey_admin_subsample(dfA, dfB)

    log(f"A full person-years: {len(dfA):,}")
    log(f"B survey person-years: {len(dfB):,}")
    log(f"A exact survey-key subsample: {len(dfA_survey):,}")

    gates = check_gates(dfB)
    survey_terms = SURVEY_ITEMS + gates + \
        ([] if gates else ["n_blocks_answered"])

    provider_missingness(dfA)
    dfA_prov = dfA.dropna(subset=[CLUSTER]).copy()
    log(f"\nA provider-complete person-years: {len(dfA_prov):,} "
        f"of {len(dfA):,} "
        f"({100 * len(dfA_prov) / len(dfA):.1f}%); both A fits use this frame "
        f"so that clustering is the only difference between them")

    # References are fixed on the provider-complete A frame -- the largest
    # frame any fit uses -- and reused everywhere so the fits share a baseline.
    demo_A, demo_refs = resolve_demographic_terms(dfA_prov, "A_full")
    demo_sub, _ = resolve_demographic_terms(
        dfA_survey, "A_survey_subsample", refs=demo_refs)
    demo_B, _ = resolve_demographic_terms(dfB, "B_survey", refs=demo_refs)

    check_category_support(dfA_prov, DEMOGRAPHIC_COLS, "A_full")
    check_category_support(dfB, DEMOGRAPHIC_COLS, "B_survey")

    terms_A = CORE + QUALITY + demo_A
    terms_A_sub = CORE + QUALITY + demo_sub
    terms_B = CORE + QUALITY + demo_B + survey_terms

    check_design(dfA_prov, terms_A, "A_full")
    check_design(dfA_survey, terms_A_sub, "A_survey_subsample")
    check_design(dfB, terms_B, "B_survey")

    check_separation(dfB, SURVEY_ITEMS + SURVEY_GATES + QUALITY)

    fits = {
        "A_full": fit(dfA_prov, terms_A, "A_full"),
        "A_person_clustered": fit(
            dfA_prov,
            terms_A,
            "A_person_clustered",
            cluster=ID,
        ),
        "A_survey_subsample": fit(
            dfA_survey,
            terms_A_sub,
            "A_survey_subsample",
        ),
        "B_survey": fit(
            dfB,
            terms_B,
            "B_survey",
        ),
    }

    # The two A fits must now agree on point estimates to numerical tolerance;
    # if they do not, something other than the covariance estimator changed.
    a_full_res = fits["A_full"][0]
    a_pers_res = fits["A_person_clustered"][0]
    if int(a_full_res.nobs) != int(a_pers_res.nobs):
        raise RuntimeError(
            "A_full and A_person_clustered were fitted on different samples "
            f"({int(a_full_res.nobs):,} vs {int(a_pers_res.nobs):,}); the "
            "clustering sensitivity check is not interpretable."
        )
    max_coef_gap = float(
        (a_full_res.params - a_pers_res.params).abs().max()
    )
    se_ratio = (a_pers_res.bse / a_full_res.bse).replace(
        [np.inf, -np.inf], np.nan
    ).dropna()
    log(f"\nclustering check: identical n={int(a_full_res.nobs):,}; "
        f"max |coef difference|={max_coef_gap:.2e} (expected ~0 -- clustering "
        f"changes standard errors only)")
    log(f"SE ratio person/provider: median={se_ratio.median():.3f} "
        f"min={se_ratio.min():.3f} max={se_ratio.max():.3f}")
    log("ratios near 1 mean repeated-person dependence is not widening "
        "intervals beyond what provider clustering already absorbs; ratios "
        "well above 1 mean person-level dependence is the binding constraint")

    families = {
        "A_full": None,
        "A_person_clustered": None,
        "A_survey_subsample": None,
        "B_survey": ["cf_", "expect_", "benefit_", "n_blocks_answered"],
    }

    # Demographic coefficient names begin with the literal term text, so a
    # C(col prefix matches every level of that column.
    demo_prefixes = [f"C({col}" for col in DEMOGRAPHIC_COLS]
    demo_exclude = None if DEMOGRAPHICS_IN_BH_FAMILY else demo_prefixes
    log(f"\nBH family: demographics "
        f"{'included in' if DEMOGRAPHICS_IN_BH_FAMILY else 'excluded from'} "
        f"the A-fit family (DEMOGRAPHICS_IN_BH_FAMILY="
        f"{DEMOGRAPHICS_IN_BH_FAMILY}); they are never in the B_survey family")

    pd.concat([
        tidy(res, k, families[k],
             exclude=demo_exclude if families[k] is None else None)
        for k, (res, _) in fits.items()
    ]).to_csv(OUT / "coefs.csv", index=False)

    pd.DataFrame([
        {
            "fit": k,
            "n": int(res.nobs),
            "llf": res.llf,
            "pseudo_r2": res.prsquared,
            "aic": res.aic,
            "converged": res.mle_retvals["converged"],
        }
        for k, (res, _) in fits.items()
    ]).to_csv(OUT / "fitstats.csv", index=False)

    sel = selection_table(dfA, dfB)
    log(
        f"\nselection frame={sel.attrs['n_earliest_person_years']:,} earliest "
        f"{SURVEY_START_YEAR}-{SURVEY_END_YEAR} person-years; "
        f"responders={sel.attrs['n_responders']:,} "
        f"non_responders={sel.attrs['n_non_responders']:,}"
    )
    log(sel.to_string(index=False))
    sel.to_csv(OUT / "selection.csv", index=False)

    res, used = fits["A_full"]
    residual_frame(res, used).to_csv(OUT / "residuals_A.csv", index=False)

    (OUT / "diagnostics.txt").write_text("\n".join(_LOG), encoding="utf-8")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()