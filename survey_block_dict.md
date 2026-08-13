# Survey Block Feature Dictionary

**Frame:** `survey_features_wide.csv`
**Unit of observation:** one row per (`Participant.Unique.ID`, `Year`)
**Population:** SYEP Older Youth, Community-Based, matched pre/post survey respondents, 2022–2025
**Rows:** 8,882 person-years / 8,209 unique people (673 people appear in >1 year, see `survey_features_earliest_wide.csv`)
**Built by:** `build_survey_features.py` on top of `survey_blocks.py`

---

## 1. What a "block" is

The survey export names checkbox items as `<question stem> - <selection label>`, so one
conceptual question is spread across 5–15 columns. A **block** is one such battery.
Each block is declared once as a `BlockSpec` (stem + groupings) and collapsed into a small
number of modeling features. Item columns are *discovered* from the stem rather than
hand-listed, so a mistyped label raises rather than silently dropping an item.

Every block produces:

| Suffix | Type | Meaning |
|---|---|---|
| `_answered` | bool | Respondent engaged with the block (selected ≥1 item, or explicitly selected "None of the above") |
| `_n_selected` | float | Count of substantive items selected |
| `_<group>` | float | 1 if **any** item in that theoretical group was selected |
| `_<single>` | float | 1 if that one specific item was selected |

**Missing-data rule:** where `_answered` is `False`, every derived feature for that block is
set to `NaN`. This prevents block non-response from masquerading as "selected nothing" —
a distinction that matters because these batteries have no forced response.

**Value coercion:** cells are mapped to 0/1 through an explicit allowlist of tokens
(`1/true/yes/x/checked/selected` → 1; `0/false/no/unchecked/""` → 0; the selection label
itself → 1). Anything unrecognized becomes `NaN` and is surfaced by `audit_block()` with a
count and a specimen value, rather than being guessed at. This was added because a prior
pipeline bug involved `'0'` string tokens in boolean columns.

---

## 2. Blocks retained

### `cf` — Counterfactual / outside options *(PRE wave, 2022–2025)*

> *"What would you have done this summer if you had not been in SYEP? [Check all that apply]"*

**Construct:** what the respondent forwent to participate. This is the measure of
*outside options*, and it is the block that separates the two competing explanations for
repeat participation: preference (they liked it) versus necessity (they had no alternative).

| Feature | Items | Rate |
|---|---|---|
| `cf_paid_alternative` | A different part-time job; A different full-time job; Paid summer internship | 0.606 |
| `cf_unpaid_activity` | Volunteered; Unpaid summer internship; Summer camp; Summer school | 0.284 |
| `cf_leisure_only` | Spending time with friends; Spending time with family | 0.560 |
| `cf_no_plans` | I had no other summer plans | 0.158 |
| `cf_n_selected` | count, excluding "no plans" | — |

**Notes.** Paid work forgone is a genuine opportunity cost; "spending time with family" is
not a competing option in the same sense, so these are deliberately kept separate rather
than summed. `cf_no_plans` is *not* the complement of `cf_paid_alternative` — respondents
can select both, and many select neither. Check the cross-tab; if the both-cell is large,
recode as a 3-level categorical (paid alternative / soft only / no plans) instead.

**Theoretical use.** The interaction `cf_paid_alternative × <experience quality>` tests
whether experience predicts return *more* among youth with real alternatives. That is the
central hypothesis of the project.

---

### `expect` — Entry expectations and application access *(PRE wave, 2022–2025)*

> *"Please indicate which of the following statements you agree with. [Check all that apply]"*

**Construct:** two things, deliberately not merged. `expect_first_choice` is a *preference*
measure and belongs conceptually with `cf`; the others are about *process*.

| Feature | Items | Rate |
|---|---|---|
| `expect_first_choice` | SYEP was my first choice for a summer program this year | 0.797 |
| `expect_knew_what_to_expect` | When I started this program, I had a good idea of what it would be like | 0.493 |
| `expect_app_clarity` | Understood all the steps to apply and enroll; Was able to get information quickly and easily | 0.799 |
| `expect_n_selected` | count, 0–4 | — |

**Notes.** A ceiling effect was anticipated here and did not materialize —
`expect_knew_what_to_expect` in particular splits almost evenly and is the more informative
of the two process items.

---

### `benefit` — Perceived benefits received *(POST wave, 2022–2025)*

> *"Besides earning money, what were the five biggest benefits you received from SYEP this past summer? [Select up to 5]"*

**Construct:** perceived returns to participation — the closest available proxy for
experience quality in the pooled frame.

| Feature | Items | Rate |
|---|---|---|
| `benefit_career_clarity` | Identified careers I am interested in; Developed a career plan with clear next steps | 0.430 |
| `benefit_job_readiness` | Understood how to look for and get a job; Understood what employers are looking for; Developed a resume or other application materials; Understood how to interact with professionals | 0.831 |
| `benefit_self_efficacy` | Developed new skills; Understood my own strengths; Felt motivated to seek a job | 0.802 |
| `benefit_mentor_relationship` | Developed relationships with mentors/supervisors | 0.347 |
| `benefit_money_management` | Understood money management | 0.313 |
| `benefit_n_selected` | count | — |

**Item harmonization.** "Earned course credit" (2022) was renamed "Earned academic credit"
(2023+). These were grouped as `benefit_academic_credit` via `max()`, which is safe because
only one is non-null in any given year — but pooled endorsement is ~2.5%, so the feature was
dropped for low variance.

**Excluded from `_n_selected`:** "Other (please specify)" (free text, not a checkbox — 191
uncoercible values) and "Decided to go to college" (2022 only, n=128).

**Cap caution.** The cap is "up to 5," not exactly 5, so the indicators do not sum to a
constant and can be modeled alongside an intercept. Because selection counts vary, include
`benefit_n_selected` as a covariate whenever using individual `benefit_*` dummies —
otherwise a dummy partly measures response style (how many boxes a person ticks) rather
than the benefit itself.

---

### `motiv` — Reasons for participating *(PRE wave, **2023–2025 only**)* — SECONDARY

> *"Besides earning money, what were other top reasons you wanted to participate in SYEP? [Check up to THREE responses]"*

| Feature | Items |
|---|---|
| `motiv_career` | Learn more about career options; Find a summer job or internship; Learn new skills; Receive work readiness training |
| `motiv_social` | Meet new people while participating in the programming |
| `motiv_structure` | Have safe and productive activity this summer |
| `motiv_credit` | To earn course credit |

**Held out of the primary model.** Not asked in 2022, so including it removes 2022 as an
index year and roughly a third of rows. Use in a secondary 2023–2024 specification instead.

**Instrument note.** The "Earn money" option is populated for all 6,350 in-scope rows and
selected by **zero** respondents — a suppressed option, presumably because the stem reads
"besides earning money." It contributes no variance and is excluded.
Built via `include_secondary=True`.

---

## 3. Blocks dropped

| Block | Wave | Reason |
|---|---|---|
| `ADULTS` — statements about adults met in SYEP (10 items) | POST | appears to be 2022 only |
| `PRIOR_SUMMERS` — activities in prior summers (10 items) | PRE | appears to be 2022 only |
| `SUPERVISOR` — 3 Likert items on supervisor relationship | POST | appears to be 2022 only (α = 0.945, but on ~16% of rows) |
| `EXTENSION` — offered option to extend internship | POST | **Substantive**, not coverage |
| `expect_nota`, `cf_nota`, `benefit_academic_credit` | — | Near-constant (≤0.03) |

**On `EXTENSION`.** Whether a worksite offers continuation is driven by that worksite's
budget and headcount, not by the participant. Modeling it as a participant characteristic
would attribute a funding condition to the individual. Dropped on reasoning, not on data
quality.

**On the low-variance drops.** These were removed after inspecting endorsement rates and
*before* inspecting any association with the outcome. The ordering matters: dropping
predictors on the basis of their outcome association is a fishing expedition; dropping them
for having no variance is not.

---

## 4. Instrument change over time — a limitation to state

---

## 5. Known issues carried forward


**No attrition comparison from this frame.** The input was already pruned to matched pairs
upstream, so `wave_coverage` is 100% `both` and pre-only/post-only respondents are not
observable here. Building the attrition comparison requires returning to the unpruned frame.

**Post-wave timing.** `benefit_*` is measured at program end, before the t+1 application
decision, so it is temporally prior to the outcome. But perceived benefits and return are
both plausibly downstream of the same latent satisfaction, so these coefficients are
**descriptive associations, not effects**.

**Self-report.** All features in this frame are self-reported. Where an administrative
equivalent exists (notably prior participation, which we calculate ourselves based on
application data), prefer the administrative measure for
modeling and reserve the self-report for validation.