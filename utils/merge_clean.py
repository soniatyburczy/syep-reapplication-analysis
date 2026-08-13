"""
merge_clean.py — build the 2015–2025 SYEP Community-Based application frame.

Writes:
    data/processed/merged.csv           all Community-Based SYEP applications
    data/processed/merged_enrolled.csv  enrolled rows only

Enrollment comes from application status, which is named ApplicationStatusCode
in the pre-2021 extract and Application.Status from 2021; the two coalesce into
`app_status` and `enrolled` is membership in ENROLLED_STATUSES. build_outcome
asserts the column is populated in every cycle year. `paid` is retained as a
diagnostic column but no longer drives `outcome`.

Scope note — service_option is deliberately NOT filtered here.
    The downstream analysis frame is Older Youth, but the reapplication
    outcome must be measured across every service option: a participant who
    was Older Youth in year t and applied to a different option in t+1 did
    reapply, and filtering to Older Youth at this stage would score them a
    non-returner and bias the outcome downward. So this frame stays
    all-options and the Older Youth restriction happens after the outcome
    lookup is built. checks() crosstabs service_option by year so the
    composition is visible without being imposed.

Site fields — worksite_id, worksite_name, provider, organization, contract,
borough — are resolved for the between-site variance work downstream, where
worksite_id is the cluster variable for robust standard errors. These only
need to be populated from 2022; earlier cycles are present in this frame to
supply prior-participation history, which uses nothing beyond
(person, year, applied, enrolled).
"""

import warnings
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "/Users/sonia/Documents/SYEP")

from utils.id_join import add_unique_id

ROOT = "/Users/sonia/Documents/SYEP"
APPS_POST = f"{ROOT}/data/raw/applications.csv"
APPS_PRE = f"{ROOT}/data/raw/applications_pre_2020.csv"
IDS = f"{ROOT}/data/raw/unique_ids.csv"
OUT = f"{ROOT}/data/processed/merged.csv"
OUT_ENROLLED = f"{ROOT}/data/processed/merged_enrolled.csv"

PROGRAM_TYPE = "Community-Based"

# Matched case-insensitively. A status meaning enrolled but absent from this
# set reads as not-enrolled rather than as an error, so check it against the
# status-by-year crosstab build_outcome prints — the vocabularies of the two
# source columns have to agree across the 2020/2021 boundary for a single
# membership test to be valid for every cycle.

ENROLLED_STATUSES = {"enrolled", "deenrolled"}

# History floor. `Program.Type` / `ProgramType` are unpopulated for the bulk
# of 2015–2018, and the populated remainder is not a random sample: Non
# Lottery is 57–59% of labelled rows in 2016–2017 against 4.5% in 2019, the
# first fully covered cycle. Those cycles are excluded rather than used.

MIN_YEAR = 2019
EXPECTED_YEARS = set(range(MIN_YEAR, 2026))

# Canonical name -> candidate source columns. Where more than one candidate is
# present the values are coalesced in order, because the pre-2021 extract
# stores several fields as complementary halves of a join rather than as
# duplicates: Initiative.Name covers 2015 and 2020, InitiativeName the rest.
# Taking the first match alone drops whole cycles.

SPEC = {
    "application_id":          ("Application.ID", "ApplicationOnlineID"),
    "age_on_start":            ("Age.on.Start.Date",),
    "service_option":          (
        "Service.Option",
        "Service.Option.Name",
        "ServiceOptionName",
    ),
    "program_type":            ("Program.Type", "ProgramType"),
    "program_name":            ("Program.Name", "ProgramName"),
    "subgroup":                ("SYEP.Programs.Subgroup",),
    "initiative":              (
        "Initiative",
        "Initiative.Name",
        "InitiativeName",
    ),
    "cycle":                   ("Cycle", "CycleName"),
    "total_hours_paid":        ("Total.Hours.Paid", "TotalHoursPaid"),
    "NumberofTimesSelected":   ("NumberofTimesSelected",),
    "participated_other_dycd": (
        "Participated.in.any.other.DYCD.funded.Workforce.programs.",
        "ParticipatedDYCDFundedProgram",
    ),
    "dycd_program": (
        "DYCD.funded.Workforce.programs",
        "DYCDFundedProgram",
    ),
    # Same field, renamed across the extract boundary: ApplicationStatusCode
    # pre-2021, Application.Status from 2021. One is present per frame, so
    # these coalesce — take() prints the per-candidate null rates, which is
    # what confirms they are complementary halves rather than two columns
    # carrying different things.
    "app_status": ("Application.Status", "ApplicationStatusCode"),
    "date_selected": ("FirstDateSelected", "DateSelected"),
    "date_declined": ("Date.Declined", "DateDeclined"),
    "date_noshow": ("Date.NoShow", "DateNoShow"),
    "date_deenrolled": ("Date.DeEnrolled", "DateDeEnrolled"),
    # Placement site. Needed downstream as the cluster variable for robust
    # standard errors and as the unit of the between-site variance check.
    # Only has to be populated for the 2022+ index cycles; the pre-2021 rows
    # exist in this frame to supply prior-participation history, and history
    # needs nothing but (person, year, applied, enrolled).
    "worksite_id": ("Worksite.ID", "WorksiteID", "WorkSiteID"),
    "worksite_name": ("Worksite.Name", "WorksiteName", "WorkSiteName"),
    # Pinned, not coalesced. Provider.Short.Name and Organization.Name are
    # plausibly different granularities of the same hierarchy rather than one
    # field renamed across the boundary; coalescing would silently mix levels
    # and make a provider-level grouping meaningless. Resolved separately so
    # the per-candidate null rates printed by take() can settle which to use.
    "provider": ("Provider.Short.Name", "ProviderShortName"),
    "organization": ("Organization.Name", "OrganizationName"),
    "contract": ("Contract.Short.Name", "ContractShortName"),
    "borough": ("Borough", "Contract.Borough", "ContractBorough"),
}

# Pinned per frame. `Cycle` in the pre-2021 file is a contract period, not the
# program cycle — resolving it by candidate order stamps every pre-2021 row
# 2015. The two ID columns are different spaces. Neither may be coalesced.

PRE_OVERRIDE = {
    "application_id": ("ApplicationOnlineID",),
    "cycle": ("CycleName",),
}

PRE_SPEC = {**SPEC, **PRE_OVERRIDE}
NO_COALESCE = {"application_id", "cycle", "provider", "organization"}


def read(path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return pd.read_csv(path, encoding="utf-8-sig", low_memory=False)


def take(df, spec, label):
    """Resolve canonical columns, coalescing multi-candidate matches."""

    out = pd.DataFrame(index=df.index)
    missing = []

    print(f"\n{label}: {len(df):,} rows")

    for canon, candidates in spec.items():
        found = [c for c in candidates if c in df.columns]

        if not found:
            out[canon] = pd.NA
            missing.append(canon)
            continue

        if canon in NO_COALESCE or len(found) == 1:
            out[canon] = df[found[0]]

            if len(found) > 1:
                print(
                    f"  {canon}: pinned to {found[0]}, "
                    f"ignoring {found[1:]}"
                )

            continue

        rates = ", ".join(
            f"{c} {df[c].isna().mean():.0%} null"
            for c in found
        )

        s = df[found[0]].astype("string").str.strip()

        for c in found[1:]:
            s = s.fillna(df[c].astype("string").str.strip())

        out[canon] = s

        print(
            f"  {canon}: coalesced {found} "
            f"-> {s.isna().mean():.0%} null "
            f"[{rates}]"
        )

    if missing:
        print(f"  absent, filled NA: {missing}")

    return out


def scope(df):
    """SYEP cycles, Community-Based only.

    The cycle filter alone separates SYEP from Summer Bridge 2020, so no
    initiative filter is needed as a redundancy guard.

    SYEP 2020 is kept. It ran online and is not comparable to other cycles,
    but it is a real cycle with real records; build_history.py excludes it
    from participation counts rather than the frame excluding it here.
    """

    cyc = df["cycle"].astype("string").str.strip()
    ptype = df["program_type"].astype("string").str.strip()
    y = cyc.str.extract(r"(\d{4})")[0]

    print("\nprogram_type null rate by cycle year:")
    print(pd.crosstab(y, ptype.isna()))

    m_cycle = cyc.str.contains("SYEP", na=False)
    m_type = ptype.eq(PROGRAM_TYPE).fillna(False)
    m_year = pd.to_numeric(y, errors="coerce").ge(MIN_YEAR).fillna(False)

    print(
        f"\nrows: {len(df):,}"
        f" -> SYEP cycle {m_cycle.sum():,}"
        f" -> {PROGRAM_TYPE} {(m_cycle & m_type).sum():,}"
        f" -> {MIN_YEAR}+ {(m_cycle & m_type & m_year).sum():,}"
    )

    print("\nprogram_type values dropped:")
    print(ptype[m_cycle & ~m_type].value_counts(dropna=False).head(10))

    print(f"\nrows dropped by the {MIN_YEAR} floor, by year:")
    print(y[m_cycle & m_type & ~m_year].value_counts().sort_index())

    return df[m_cycle & m_type & m_year].copy()


def build_outcome(df):
    hours = pd.to_numeric(df["total_hours_paid"], errors="coerce")
    n_sel = pd.to_numeric(df["NumberofTimesSelected"], errors="coerce")

    st = df["app_status"].astype("string").str.strip()

    # Diagnostic: the vocabulary has to be stable across 2020/2021, where the
    # source column changes name. A value that appears on only one side of
    # that boundary means the two columns are not interchangeable.
    print("\napplication status by year:")
    print(pd.crosstab(st, df["Year"], dropna=False))

    enrolled = (
        st.str.casefold().isin(ENROLLED_STATUSES).fillna(False).astype(bool)
    )
    
    # A year with no status at all would report a 0% enrollment rate rather
    # than an error — the same failure mode as the scope-filter null
    # propagation, so it is a hard stop.
    cov = df["app_status"].notna().groupby(df["Year"]).mean()

    print("\napp_status coverage by year:")
    print(cov.round(3))

    dark = cov.index[cov.eq(0)].tolist()

    if dark:
        raise AssertionError(
            f"app_status is entirely null in {dark}, so enrolled would read "
            f"False for every row in those cycles. the source column is "
            f"named differently again, or is absent from that extract."
        )

    selected = df["date_selected"].notna()
    selected |= n_sel.gt(0).fillna(False)
    selected |= enrolled
    selected |= st.str.casefold().eq("deenrolled").fillna(False).astype(bool)

    has_signal = (
        df["date_selected"].notna()
        | n_sel.notna()
        | df["app_status"].notna()
    )

    df["total_hours_paid"] = hours
    df["paid"] = hours.gt(0).fillna(False)
    df["enrolled"] = enrolled
    df["selected"] = selected

    # Retained from the hours-based definition: the two should agree closely,
    # and where they do not the disagreement is the finding, not noise.
    print("\npaid x enrolled:")
    print(pd.crosstab(df["paid"], df["enrolled"]))

    df["outcome"] = np.select(
        [
            df["enrolled"],
            df["date_declined"].notna(),
            df["selected"],
            has_signal,
        ],
        [
            "enrolled",
            "declined",
            "selected_not_enrolled",
            "not_selected",
        ],
        default=None,
    )

    return df

def checks(df):
    print("\nenrolled rate by year:")
    print(df.groupby("Year")["enrolled"].mean().round(3))

    print("\noutcome by year:")
    print(pd.crosstab(df["Year"], df["outcome"], dropna=False))

    print("\noutcome null (no selection signal) by year:")
    print(df.groupby("Year")["outcome"].apply(lambda s: s.isna().mean()))

    print("\nservice_option by year:")
    print(pd.crosstab(df["service_option"], df["Year"]))

    print("\nprogram_name by year (Lottery share is the coverage check):")
    print(pd.crosstab(df["Year"], df["program_name"]))

    print("\nnull rate by year:")
    for c in (
        "service_option",
        "age_on_start",
        "total_hours_paid",
        "date_selected",
    ):
        r = df.groupby("Year")[c].apply(lambda s: s.isna().mean())
        print(
            f"  {c}: "
            + ", ".join(f"{y}:{v:.0%}" for y, v in r.items())
        )

    # Site fields only need to be populated for the index cycles (2022+).
    # Pre-2021 rows are here to supply prior-participation history, which uses
    # nothing but (person, year, applied, enrolled), so nulls there are fine.
    print("\nsite field coverage by year (2022+ is what matters):")
    for c in ("worksite_id", "provider", "organization", "borough"):
        if c not in df.columns:
            print(f"  {c}: ABSENT from frame")
            continue
        r = df.groupby("Year")[c].apply(lambda s: s.notna().mean())
        print(f"  {c}: " + ", ".join(f"{y}:{v:.0%}" for y, v in r.items()))

    # Cluster-robust SEs need enough clusters for the asymptotics to behave.
    idx = df["Year"].ge(2022)
    for c in ("worksite_id", "provider", "organization"):
        if c not in df.columns:
            continue
        n = df.loc[idx, c].nunique()
        print(f"\n{c}: {n:,} distinct values in 2022+")
        if 0 < n < 40:
            print(f"  ! under 40 clusters — robust SE asymptotics unreliable")

    # provider vs organization: is one nested in the other, or are they the
    # same field under two names? A near-1:1 mapping means duplicate columns;
    # many organizations per provider means a real hierarchy, pick a level.
    if {"provider", "organization"} <= set(df.columns):
        pair = df.loc[idx, ["provider", "organization"]].dropna()
        if len(pair):
            per_prov = pair.groupby("provider")["organization"].nunique()
            print(
                f"\nprovider -> organization: {len(per_prov):,} providers, "
                f"median {per_prov.median():.0f} orgs each, max {per_prov.max():,}"
            )

    print(
        f"\nduplicate (person, year): "
        f"{df.duplicated(subset=['Participant.Unique.ID', 'Year']).sum():,}"
    )

    print(
        f"null Participant.Unique.ID: "
        f"{df['Participant.Unique.ID'].isna().sum():,}"
    )


def main():
    apps_post = read(APPS_POST)
    apps_pre = read(APPS_PRE)
    id_df = read(IDS)

    merged = pd.concat(
        [
            take(apps_post, SPEC, "applications"),
            take(apps_pre, PRE_SPEC, "applications_pre_2021"),
        ],
        ignore_index=True,
    )

    merged = scope(merged)

    # Defensive check before downstream joins.
    dup = merged.duplicated(subset="application_id", keep=False)

    if dup.any():
        print("\nDuplicate application_id rows:")
        print(
            merged.loc[dup]
            .sort_values("application_id")
            [["application_id", "cycle", "program_name"]]
        )

        before = len(merged)
        merged = merged.drop_duplicates(
            subset="application_id",
            keep="first"
        )
        print(f"Dropped {before - len(merged):,} duplicate application_id rows.")

    merged["Year"] = pd.to_numeric(
        merged["cycle"].str.extract(r"(\d{4})")[0],
        errors="raise",
    ).astype(int)

    assert merged["Year"].between(MIN_YEAR, 2025).all()

    print("\nrows by year:")
    print(merged["Year"].value_counts().sort_index())

    seen = set(merged["Year"].unique())

    if seen != EXPECTED_YEARS:
        raise AssertionError(
            f"year coverage wrong. missing {sorted(EXPECTED_YEARS - seen)}, "
            f"unexpected {sorted(seen - EXPECTED_YEARS)}. "
            f"a filter is running on a column that is null "
            f"in the missing cycles."
        )

    merged = build_outcome(merged)

    # Run on the full frame, not just paid rows.
    merged, report = add_unique_id(
        merged,
        id_df,
        df_key="application_id",
        map_key="ApplicationOnlineID",
        map_value="SSN_Encoded",
    )

    print(f"\nid join: {report}")

    checks(merged)

    merged.to_csv(
        OUT,
        encoding="utf-8-sig",
        index=False,
    )

    merged[merged["enrolled"]].to_csv(
        OUT_ENROLLED,
        encoding="utf-8-sig",
        index=False,
    )

    print(
        f"\nwrote {OUT} ({len(merged):,}) and "
        f"{OUT_ENROLLED} ({merged['enrolled'].sum():,})"
    )

    return merged


if __name__ == "__main__":
    main()