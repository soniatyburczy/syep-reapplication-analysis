# SYEP Reapplication: Findings
 
Older Youth Community-Based track, 2022–2024 index years. Outcome: applied again the
following year. 130,417 person-years; 114,625 in the estimated model. Logistic regression,
provider-clustered standard errors, Benjamini–Hochberg correction within each fit.
 
---
 
## Headline
 
> **How much of the program a young person completes is by far the strongest predictor of
> whether they come back. What they say about the program afterward predicts almost nothing.**
 
---
 
## Finding 1 — Completion intensity, monotonic dose-response
 
Hours worked, adjusted for age, year, borough, and placement quality. Reference band is
126–149 hours.
 
| Hours worked | Odds ratio | 95% CI | q | Illustrative return rate |
|---|---|---|---|---|
| 0 | **0.24** | 0.22 – 0.25 | <0.0001 | 23.8% |
| 1–25 | **0.38** | 0.34 – 0.42 | <0.0001 | 33.3% |
| 26–75 | **0.64** | 0.53 – 0.78 | <0.0001 | 45.8% |
| 76–125 | **0.85** | 0.76 – 0.94 | 0.0026 | 52.8% |
| 126–149 | *reference* | — | — | 57.0% |
| 150 (cap) | **1.33** | 1.24 – 1.42 | <0.0001 | 63.8% |
 
- Six ordered categories, **no inversions** — every step up in hours raises return odds.
- Odds spread from lowest to highest band is **5.6×**.
- Roughly a **40-percentage-point** gradient across the range.
- All six contrasts survive BH correction at q < 0.003.
- Program-wide baseline reapplication rate: **57.0%** (n = 97,761 first-observed participants).
**Illustrative rates anchor the reference band at the 57.0% program average.** They convey
magnitude, not observed cell means. Replace with actual band rates before publishing:
`df.groupby("hours_band")["returned"].mean()`.
 
### It replicates in the survey subsample
 
Same specification, different population (n = 4,785 completers). The low bands hold:
 
| Hours | A_full OR | Subsample OR | Subsample CI | q |
|---|---|---|---|---|
| 0 | 0.24 | 0.26 | 0.12 – 0.54 | 0.002 |
| 1–25 | 0.38 | 0.42 | 0.21 – 0.84 | 0.044 |
| 26–75 | 0.64 | 0.55 | 0.36 – 0.83 | 0.018 |
 
The upper bands flatten (150-cap OR falls to 1.04, q = 0.83) — expected, because survey
responders are 98.8% completers, so there is almost no low-hours variation left to detect.
 
---
 
## Finding 2 — No-shows are not a separate risk factor
 
`is_no_show`: OR 1.05, CI 0.93 – 1.17, q = 0.53.
 
Once hours worked is in the model, no-show status adds nothing — because
no-shows *are* the zero-hours band. Hours fully absorbs it. The risk is not "who no-showed,"
it is "who accumulated few hours," which is a broader group.
 
---
 
## Finding 3 — Self-reported program benefits do not predict return
 
Five pre-specified benefit items, n = 4,785 survey-linked completers. **None survives BH.**
 
| Item | OR | 95% CI | q |
|---|---|---|---|
| Career clarity | 1.04 | 0.92 – 1.19 | 0.70 |
| Job readiness | 0.85 | 0.73 – 1.00 | 0.20 |
| Self-efficacy | 1.04 | 0.87 – 1.24 | 0.75 |
| Mentor relationship | 0.99 | 0.89 – 1.11 | 0.91 |
| Money management | 1.04 | 0.89 – 1.22 | 0.70 |
 
- Odds ratios span **0.85 to 1.04**; the widest interval runs 0.73 – 1.24.
- These are **precise nulls**, not underpowered ones — we can rule out effects larger than
  roughly ±20% in odds.
- If the program's logic model assumes youth who report benefiting are the ones who come
  back, this tests that assumption and **does not support it.**
---
 
## Finding 4 — Placement fit is the one survey variable that works
 
`expect_first_choice`: **OR 1.31, CI 1.13 – 1.51, q = 0.003.** Roughly +5.4 percentage points
at the responder baseline of 69.6%.
 
The only survey item that predicts reapplication is not about how the program
felt — it is about whether the young person got the placement they asked for. This is
consistent with the hours result: **allocation and completion predict return; attitudes do
not.**
 
---
 
## Finding 5 — Return rates rose over the period
 
2023 vs 2022: OR 1.20 (1.11 – 1.30, q < 0.0001). 2024 vs 2022: OR 1.31 (1.20 – 1.42,
q < 0.0001). Secondary; report as context, not as a program effect.
 
---
 
## What this does *not* establish
 
**Association, not causation.** Hours worked is endogenous. The same underlying engagement
that keeps a young person on site to 150 hours plausibly also brings them back. This does not
license "assign more hours and retention will rise."
 
**Mechanism is unidentified.** We can locate *where* retention is won or lost — early in the
placement — but not *why* any individual returns. The survey was designed for a pre/post
attitude-change study, and its items are retrospective satisfaction measures, which are
ceiling-loaded and weakly behavior-predictive.
 
**Provider characteristics are unexamined.** `provider` enters as a clustering variable only;
no provider-level predictors are in the model. This is one third of the stated research
question and it remains open.
 
**Two coefficients are not interpretable and are excluded from the table.** In the survey
model, the intercept (CI 0.12 – 10.77) and `worksite_matched` (CI 0.16 – 9.32) are
near-constant in the completer population. `is_no_show` in that fit rests on 29 rows.
 
**`worksite_matched` runs counter-intuitively in the full model** (OR 0.89, q = 0.007):
having a matched worksite record predicts *lower* return. This is most plausibly an
administrative-completeness artifact rather than a participant-experience effect, and it
reverses direction in the subsample. Flagged for follow-up, not reported as a finding.
 
---
 
## Limitations
 
**Survey respondents are a completer population.** Responders return at 69.6% vs 56.1% for
non-responders — a 13.5-point gap. 20.0% of non-responders worked zero hours; 1.2% of
responders did. This is by design: matched pre/post responses require program completion.
Consequence: **the survey findings describe completers, not SYEP participants**, and the
administrative predictors lose 68% of their explanatory power in that population
(pseudo-R² 0.074 → 0.024) because their variance has been truncated.
 
**Provider is unrecorded for 37.8% of 2022 person-years** and 0.0% of 2023 and 2024 — a
data-collection regime change, not attrition. 2022 is 32.0% of the raw frame but 22.7% of the
estimated sample. A 2022-excluded refit is the sensitivity check.
 
**Borough is null under provider clustering** (Brooklyn q = 0.053) but strongly significant
under person clustering (q < 0.0001). Providers nest within boroughs, so provider clustering
is the appropriate and conservative choice. Worth noting as a case where the inference
decision changed the conclusion.
 
**Clustering rests on 46 providers**, near the conventional minimum for cluster-robust
asymptotics.
 
**2025 is excluded as an index year** — reapplication would require 2026 data (full data unavailable: 2026 cycle not finished)
 
**1,038 survey respondents were excluded** because their survey year postdates their first
administrative year; including them would measure return conditional on prior participation.
 
---
 
## TLDR
 
Hours worked show the largest and clearest association with reapplication. Relative to participants working 126–149 hours, those with substantially fewer hours have markedly lower odds of reapplying, while participants reaching 150+ hours have approximately 33% higher odds. Year-to-year differences are also nontrivial, while the worksite-matching indicator has a comparatively modest association. Age appears associated with reapplication as well, but because age is modeled with a spline, its effect should be interpreted from predicted probabilities rather than individual spline coefficients. Survey-reported benefits — career clarity, job readiness,
self-efficacy, mentorship, money management — show no association with returning. The one
survey item that does predict return is whether the participant received their first-choice
placement (OR 1.31). Together these point to allocation and completion, not post-hoc
satisfaction, as where retention is determined. Because hours worked is not randomly assigned,
these are associations rather than causal effects, and they identify where attrition happens
without explaining why.
 