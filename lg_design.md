# `lg.py` — Reapplication Models: Design and Method

Covers the modelling stage only: frame construction, specification, inference, and checks each step.

---

## 1. What the model answers

**Question.** Among Older Youth Community-Based SYEP participants, which participant,
placement, and provider characteristics predict returning to the program the following year?

**Outcome.** `returned` — binary, whether the person has a subsequent SYEP application.

**Unit of observation.** The person-year. One row per participant per program year.

**Estimator.** Logistic regression with cluster-robust standard errors.

---

## 2. Two frames

The pipeline maintains two analysis frames that are deliberately *not* pooled.

| | **Frame A** | **Frame B** |
|---|---|---|
| Source | `analysis_frame_worksite.csv` | `survey_features_wide.csv` |
| Rows | ~130k person-years | ≤1 row per person |
| Contents | Administrative records | Administrative + survey responses |
| Selection | Full risk set | Self-selected (survey responders) |

**Why they stay separate.** Survey response is voluntary. B's respondents are not a random
sample of A, so B is not assumed to estimate the same population parameters. Pooling would
let a selected subgroup drive estimates for the whole program. Instead, A carries the
headline estimates and B is reported as a distinct, explicitly selected secondary analysis.

**Why B is one row per person.** People appear in A repeatedly, and repeated observations of
the same person are not independent. Restricting B to each person's *first* observed year
removes that dependence at the frame level rather than trying to model around it. It also
keeps the outcome interpretable: `returned` always means "returned after a first
participation," never "returned after an unknown number of prior years."

---

## 3. Frame B construction

> **Design decision (amendment).** B is selected inside `lg.py` from the full survey
> person-year file, not consumed pre-selected from `survey_features_earliest_wide.csv`.

The upstream extract defined each person's "earliest" year among *rows that carry a survey*.
`lg.py` defines it among *A's post-filter person-years*. These disagree whenever someone
participated in a year with no survey and was surveyed in a later year — earliest 2023
administratively, earliest 2024 in the survey extract. On the real data this affected
1,018+ rows.

Neither definition is wrong on its own terms, but only one is right for this analysis: the
risk set is A, so "first observed" must be defined against A. `build_survey_frame()` now
performs the selection locally, which makes the two definitions agree **by construction**
rather than asserting after the fact that they already do.

### Selection steps

1. Read the full survey person-year file; require one row per `(person_id, Year)`.
2. Restrict to the 2022–2025 survey window.
3. Join each person's earliest A person-year in that window.
4. Keep the survey row sitting on that earliest A year.

### The policy switch

Step 4 forces a choice for people surveyed *after* their first administrative year.
`REQUIRE_SURVEY_AT_EARLIEST_A_YEAR` controls it:

- **`True` (default).** Drop them. B stays a strict subset of the earliest-person-year
  frame, so `returned` is never conditioned on prior participation and the bridge model
  (§5) compares identical person-years across frames. Consistent with the pre-specified
  first-observed-year-only design.
- **`False` (sensitivity).** Keep their earliest *surveyed* year. Larger B, but for those
  people the outcome is measured after at least one prior participation year — a different
  estimand.

The log reports the sample size under both settings, so the trade-off is a documented number
rather than an implicit default.

### Attrition accounting

Survey participants with no A person-year in the window are outside the risk set and are
dropped, not treated as an error — they are typically people excluded upstream by the track,
enrollment, or index-year filters. The log records how many were dropped and how many of
them appear in A in *any* year, which separates legitimate attrition from a broken join.

`diagnostics.txt` also carries a **survey-year × earliest-A-year cross-tab**. A near-diagonal
table means little is lost; mass above the diagonal means survey coverage starts later than
administrative coverage, and the strict policy costs real sample. This is the table to read
before quoting B's n.

---

## 4. Specification

### Administrative terms (all fits)

| Term | Form | Rationale |
|---|---|---|
| `age_on_start` | Natural cubic spline, 3 df | Return propensity is not linear in age; the program's upper age boundary creates curvature a linear term would miss. |
| `hours_band` | Categorical, ref = 126–149 | Hours paid relate to return non-monotonically. Bands: 0 / 1–25 / 26–75 / 76–125 / 126–149 / 150 (cap). |
| `Year` | Categorical | Absorbs cohort-level shocks (program capacity, funding, COVID-era disruption). |
| `borough` | Categorical | Geographic differences in provider mix and labor market. |
| `is_no_show` | Binary | Placement quality. |
| `worksite_matched` | Binary | Whether a worksite record attached at all. |

**Reference category.** The `hours_band` reference is 126–149, not 0. Each hours coefficient
is therefore a contrast against a full-time working band, so the comparison is "worked less
than full-time" rather than "worked at all versus not." The latter would confound dose with
participation.

**`worked_zero_hours` is computed but not a regression term.** It is identical to the `0`
level of `hours_band`; including both makes the design rank-deficient. It is retained for
descriptive tables only.

### Survey terms (frame B only)

Twelve pre-specified items across three blocks:

- **`cf_*`** — counterfactual activity absent SYEP (paid alternative, unpaid activity,
  leisure only, no plans)
- **`expect_*`** — pre-program expectations (application clarity, first-choice placement,
  knew what to expect)
- **`benefit_*`** — self-reported benefits (career clarity, job readiness, self-efficacy,
  mentor relationship, money management)

**Block non-response is materialised as `0`, not left missing.** Under statsmodels'
`missing='drop'`, one unanswered block would listwise-delete the row across *all three*
blocks, discarding answered blocks along with the unanswered one. Zero-filling keeps the
answered information.

**Block gates (`cf_answered`, `expect_answered`, `benefit_answered`) carry the
response/non-response distinction that zero-filling would otherwise erase** — but only when
that information is not already recoverable from the item pattern. `check_gates()` tests this
empirically: if every answerer selected ≥1 item and no non-answerer selected any, the gate is
a deterministic function of the items, so it is dropped as redundant. Retained gates enter as
regressors; when all three are redundant, a single `n_blocks_answered` count is used instead.

---

## 5. The four fits

| Fit | Frame | Clustering | Purpose |
|---|---|---|---|
| `A_full` | A, all person-years | `provider` | Headline estimates |
| `A_person_clustered` | A, all person-years | `person_id` | Dependence sensitivity |
| `A_survey_subsample` | A, restricted to B's keys | `provider` | Bridge model |
| `B_survey` | B | `provider` | Survey-augmented |

**Clustering on `provider`.** Participants at the same provider share recruitment practices,
placement quality, and staff, so their outcomes are correlated; independent standard errors
would be too narrow. `provider` has 50 values at ~100% coverage. (`worksite_id` was rejected
as a cluster variable — it proved effectively a row identifier, which would collapse
clustering back to no clustering.)

**`A_person_clustered` changes inference only, not the fitted mean structure.** It is not a
person fixed-effects model. It exists to show whether repeated-person dependence in A
materially widens the intervals — a direct check on the concern that motivated B's
one-row-per-person restriction.

> **Design decision (amendment).** Both A fits run on the **provider-complete** rows.
> `provider` is missing on ~12% of A person-years, concentrated in 2022. Fitting
> `A_person_clustered` on the raw frame would have changed the estimation sample at the same
> time as the clustering variable, confounding the sensitivity check. The pipeline now
> asserts that the two A fits share an identical n and reports their maximum coefficient
> difference (expected ~0) plus the median SE ratio, which is the actual result of the check:
> ratios near 1 mean provider clustering already absorbs person-level dependence.
>
> **Scope consequence.** `A_full` is the full risk set *among person-years with a provider
> record*. Because that missingness is concentrated in 2022, the A fits are lighter on 2022
> than the raw frame is. `diagnostics.txt` reports provider coverage by year.

**`A_survey_subsample` is the bridge.** It runs the *administrative* specification on the
*exact* person-years present in B. Comparing it to `A_full` isolates how much of any
difference in `B_survey` comes from who responded to the survey, versus from the survey
variables themselves. Without it, selection effects and survey effects are inseparable.

---

## 6. Multiple comparisons

Benjamini–Hochberg FDR correction, applied **separately within each fit**, because the fits
have different power and different declared inferential families.

- **A fits:** all non-intercept coefficients form the family.
- **`B_survey`:** only the pre-specified survey terms form the family. Administrative
  covariates are reported with p-values but excluded from the correction — they are
  adjustment terms, not hypotheses under test.

Predictors and family membership were fixed before any outcome cross-tabs were examined.

> **Known behaviour.** Family membership is matched by prefix (`cf_`, `expect_`, `benefit_`),
> which also sweeps in any retained block gates. Treat gates as part of the survey family, or
> switch to exact item-name matching if they are adjustment terms.

---

## 7. Validation gates

The pipeline distinguishes conditions that should **drop rows with an accounting** from
conditions that should **halt**.

**Drop and log** — expected data conditions:
- Survey participants with no A person-year in the window.
- Survey rows off the person's earliest A year (under the strict policy).
- Terms constant within a subset (`drop_degenerate`) — a constant column is collinear with
  the intercept and can make the Hessian singular, which happens naturally in the smaller
  survey subsample.

**Halt** — conditions where dropping would mask a bug:
- Duplicate `(person_id, Year)` in A or in the survey file. Violates the unit of observation.
- **100% non-match between the survey file and A.** Total failure to join is a key problem —
  typically an ID dtype mismatch — not attrition. Silently producing an empty B here would be
  the worst possible failure mode.
- A retained B row off the earliest A year under the strict policy (post-condition).
- Non-numeric text in a survey item, or a gate that is not binary after preprocessing.
  Coercing these silently would turn a data problem into a plausible-looking zero.

**Reported, not enforced** — written to `diagnostics.txt` for review:
- Provider coverage by year, since cluster missingness changes which sample is estimated.
- Design matrix rank and scaled condition index per fit.
- Univariate separation screening at two thresholds: cells with 0%/100% outcome rates or
  n < 20 are flagged as `SEPARATION`; cells under 50 rows are flagged as `thin`. Thin-cell
  coefficients are estimable but carry wide intervals. This is a screen, not proof that
  multivariable quasi-separation is absent.

**Two subtleties in the design diagnostic.**

*It runs the same preparation as the fit.* `check_design` calls the shared
`prepare_fit_frame()` — cluster-complete rows, then constant-term removal — so the reported
shape is the design actually estimated. Reporting the pre-preparation design instead makes
constant columns that `fit()` removes surface as a phantom rank deficiency: a constant column
is the intercept times a scalar, so it is exactly collinear, costing one rank and sending the
raw condition number to machine-precision scale (~1e18). In practice this fires on
`cf_answered` and `expect_answered`, whose blocks every respondent answered.

*Conditioning is reported as a Belsley scaled condition index*, with columns normalised to
unit length before `cond()`. The conventional >30 threshold applies to this scaled quantity;
applying it to the raw matrix flags harmless differences in column units.

**Fitting note.** Newton iterations are capped at 200 rather than statsmodels' default of 35,
which is tight for a design combining splines with several categorical expansions. It is a
no-op where the fit already converges.

---

## 8. Selection analysis

`selection.csv` compares responders and non-responders on their **earliest 2022–2025 A
person-year**, across `returned`, `age_on_start`, `total_hours_paid`, `worked_zero_hours`,
`is_no_show`, and `worksite_matched`.

Responder status is assigned **per person, not per person-year**. Keying on `(person_id,
Year)` would misclassify a responder as a non-responder whenever their survey year differs
from their earliest A year — which is possible under the sensitivity policy. Responding is a
person-level attribute and is treated as one.

This table is the basis for judging how far `B_survey` generalises. Large responder /
non-responder gaps are a caution on external validity, not a defect in the fit.

---

## 9. Outputs

All written to `data/to_use/models/`.

| File | Contents |
|---|---|
| `coefs.csv` | Tidy coefficients: term, coefficient, odds ratio, 95% CI, p, BH q, family flag |
| `fitstats.csv` | Per fit: n, log-likelihood, pseudo-R², AIC, convergence flag |
| `selection.csv` | Responder vs non-responder means on earliest person-years |
| `diagnostics.txt` | Full run log — B construction, cross-tab, provider coverage, gate redundancy, rank, separation, clustering check |
| `residuals_A.csv` | `A_full` fitted values and response residuals, keyed to person-year |

**Read `diagnostics.txt` first.** Sample sizes, dropped-row accounting, and the coverage
cross-tab all live there. A coefficient table read without it can silently be a table about a
much smaller and differently-selected population than intended.
