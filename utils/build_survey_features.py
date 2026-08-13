"""
build_survey_features.py
========================

Assemble the final wide survey feature frame: one row per
(Participant.Unique.ID, Year), pre-wave and post-wave features side by side.

Blocks retained after the coverage audit
----------------------------------------
    CF            pre    2022-2025    counterfactual / outside options
    EXPECTATIONS  pre    2022-2025    first choice, entry expectations
    MOTIVATIONS   pre    2023-2025    reasons for participating   [SECONDARY]
    BENEFITS      post   2022-2025    perceived benefits received

Blocks dropped, and why
-----------------------
    ADULTS         post   2022 only  -- discontinued after 2022
    PRIOR_SUMMERS  pre    2022 only  -- discontinued; retained for one-off
                                        validation against admin history
    SUPERVISOR     post   2022 only  -- discontinued. Likert triple; alpha
                                        0.945 but computed on ~16% of rows
    EXTENSION      post   --         -- "offered option to extend" reflects
                                        worksite budget and headcount, not a
                                        participant characteristic. Dropped
                                        on substantive grounds, not coverage.

The 2022 instrument was substantially richer than 2023-2025. Everything
measuring adult/supervisor relationship quality lives there and nowhere
else, so in the pooled model that construct is represented only by
`benefit_mentor_relationship`. State this as a limitation. A 2022-only
model (n ~ 2,532) using the full relational battery is viable future work.

Run order
---------
    1. audit_block() on each spec, confirm n_uncoercible == 0
    2. this script
    3. merge onto the admin panel, then model
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from utils.survey_blocks import (
    KEYS,
    BlockSpec,
    build_block_frame,
    block_year_matrix,
    _norm,
)

# --------------------------------------------------------------------------
# 1. Specs -- edit labels to match your export exactly
# --------------------------------------------------------------------------

CF = BlockSpec(
    prefix="cf",
    stem="What would you have done this summer if you had not been in SYEP? [Check all that apply]",
    groups={
        "paid_alternative": [
            "A different part-time job",
            "A different full-time job",
            "Paid summer internship",
        ],
        "unpaid_activity": [
            "Volunteered",
            "Unpaid summer internship",
            "Summer camp",
            "Summer school",
        ],
        "leisure_only": [
            "Spending time with friends",
            "Spending time with family",
        ],
    },
    singles={"no_plans": "I had no other summer plans"},
    exclude_from_count=["I had no other summer plans"],
)

EXPECTATIONS = BlockSpec(
    prefix="expect",
    stem="Please indicate which of the following statements you agree with. [Check all that apply]",
    groups={
        "app_clarity": [
            "I understood all the steps I needed to complete to apply and enroll to the program",
            "I was able to get the information I needed to apply quickly and easily",
        ],
    },
    singles={
        "first_choice": "SYEP was my first choice for a summer program this year",
        "knew_what_to_expect": "When I started this program, I had a good idea of what it would be like",
    },
)

# 2023+ only. Held out of the primary model -- including it costs 2022.
MOTIVATIONS = BlockSpec(
    prefix="motiv",
    stem="Besides earning money, what were other top reasons you wanted to participate in SYEP? [Check up to THREE responses]",
    groups={
        "career": [
            "Learn more about career options",
            "Find a summer job or internship",
            "Learn new skills",
            "Receive work readiness training",
        ],
        "social": ["Meet new people while participating in the programming"],
        "structure": ["Have safe and productive activity this summer"],
    },
    singles={"credit": "To earn course credit"},
    # "Earn money" is a suppressed option: populated but never selected (n=0).
    # "Other (please specify)" is free text, not a checkbox.
    exclude_from_count=["Earn money", "Other (please specify)"],
)

BENEFITS = BlockSpec(
    prefix="benefit",
    stem="Besides earning money, what were the five biggest benefits you received from SYEP this past summer? [Select up to 5]",
    groups={
        "career_clarity": [
            "Identified careers I am interested in",
            "Developed a career plan with clear next steps",
        ],
        "job_readiness": [
            "Understood how to look for and get a job",
            "Understood what employers are looking for",
            "Developed a resume or other job application materials",
            "Understood how to interact with professionals",
        ],
        "self_efficacy": [
            "Developed new skills",
            "Understood my own strengths",
            "Felt motivated to seek a job",
        ],
        # Renamed 2023+: "Earned course credit" -> "Earned academic credit".
        # Only one is non-null in any given year, so max() harmonizes them.
        # Endorsement is only ~2.5% pooled -- likely drop from the model.
        "academic_credit": [
            "Earned course credit",
            "Earned academic credit",
        ],
    },
    singles={
        "mentor_relationship": "Developed relationships with mentors/supervisors",
        "money_management": "Understood money management",
    },
    exclude_from_count=[
        "Other (please specify)",      # free text
        "Decided to go to college",    # 2022 only, n=128
    ],
)

PRE_SPECS = [CF, EXPECTATIONS]
POST_SPECS = [BENEFITS]
SECONDARY_SPECS = [MOTIVATIONS]        # pre-wave, 2023+ only

PRE_LABEL, POST_LABEL = "PRE", "POST"

# Near-constant indicators: no variance to model. Dropped after inspecting
# endorsement rates, not after inspecting their association with the outcome.
LOW_VARIANCE = ["expect_nota", "cf_nota", "benefit_academic_credit"]


# --------------------------------------------------------------------------
# 2. Build
# --------------------------------------------------------------------------

def build_wide(df: pd.DataFrame,
               include_secondary: bool = False,
               how: str = "inner",
               drop_low_variance: bool = True) -> pd.DataFrame:
    """Build the wide (one row per person-year) survey feature frame.

    Parameters
    ----------
    include_secondary
        Add MOTIVATIONS. Restricts usable index years to 2023+.
    how
        'inner' = matched pairs only (primary analysis).
        'outer' = keep pre-only and post-only respondents, for attrition
        comparison. Only meaningful if `df` has NOT already been pruned
        to matched pairs upstream.
    drop_low_variance
        Remove the near-constant indicators listed in LOW_VARIANCE.
    """
    pre_specs = list(PRE_SPECS) + (list(SECONDARY_SPECS) if include_secondary else [])

    print("Block coverage before split:")
    print(block_year_matrix(df, pre_specs + POST_SPECS).to_string(), "\n")

    survey_type = df["SurveyType"].map(_norm)
    pre_rows = df[survey_type == _norm(PRE_LABEL)]
    post_rows = df[survey_type == _norm(POST_LABEL)]
    if pre_rows.empty or post_rows.empty:
        raise ValueError(
            f"SurveyType did not split on {PRE_LABEL!r}/{POST_LABEL!r}. "
            f"Found: {df['SurveyType'].unique().tolist()}"
        )

    pre = build_block_frame(pre_rows, pre_specs).drop(columns="SurveyType")
    post = build_block_frame(post_rows, POST_SPECS).drop(columns="SurveyType")

    id_keys = ["Participant.Unique.ID", "Year"]
    wide = pre.merge(post, on=id_keys, how=how, validate="1:1",
                     indicator=True, suffixes=("", "_dup"))
    wide = wide.drop(columns=[c for c in wide.columns if c.endswith("_dup")])
    wide = wide.rename(columns={"_merge": "wave_coverage"})

    if drop_low_variance:
        hits = [c for c in LOW_VARIANCE if c in wide.columns]
        if hits:
            wide = wide.drop(columns=hits)
            print(f"dropped low-variance indicators: {hits}\n")

    return wide


def report(wide: pd.DataFrame) -> None:
    """Sanity checks. Run these before modeling."""
    n_rows = len(wide)
    n_people = wide["Participant.Unique.ID"].nunique()
    print(f"rows: {n_rows:,}   unique people: {n_people:,}")
    if n_rows > n_people:
        print(f"  ! {n_rows - n_people:,} repeated people. Rows are NOT independent --\n"
              f"    cluster SEs on Participant.Unique.ID, or keep first year only.")

    print("\nrows per year:")
    print(wide["Year"].value_counts().sort_index().to_string())

    if "wave_coverage" in wide.columns:
        vc = wide["wave_coverage"].value_counts()
        print("\nwave coverage:")
        print(vc.to_string())
        if vc.get("left_only", 0) == 0 and vc.get("right_only", 0) == 0:
            print("  note: all rows matched -- input was already pruned to pairs,\n"
                  "        so no attrition comparison is possible from this frame.")

    feat = [c for c in wide.columns
            if c.startswith(("cf_", "expect_", "motiv_", "benefit_"))]

    cov = wide[feat].notna().mean().sort_values()
    print("\nfeature coverage (want ~1.0 for in-scope features):")
    print(cov.round(3).to_string())
    thin = cov[cov < 0.90]
    if len(thin):
        print(f"  ! below 0.90: {list(thin.index)}")

    binary = [c for c in feat
              if wide[c].notna().any() and wide[c].dropna().isin([0, 1]).all()]
    rates = wide[binary].mean().sort_values()
    print("\nbinary endorsement rates:")
    print(rates.round(3).to_string())
    flagged = rates[(rates < 0.05) | (rates > 0.95)]
    flagged = [c for c in flagged.index if not c.endswith("_answered")]
    if flagged:
        print(f"  ! near-constant, consider dropping: {flagged}")


if __name__ == "__main__":
    # df = pd.read_csv("...")   # your long survey frame
    #
    # wide = build_wide(df, include_secondary=False, how="inner")
    # report(wide)
    # wide.to_csv("survey_features_wide.csv", index=False)
    print(__doc__)