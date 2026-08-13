"""
Column pruning + ordering for the SYEP applications dataset.

Usage:
    from utils.prune import profile_columns, find_constants, find_dupe_pairs, KEEP, prune

    # inspect first
    prof = profile_columns(df_all)
    prof.head(40)

    find_constants(df_all)

    # then prune
    df_all = prune(df_all, KEEP)
"""

import pandas as pd

# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def profile_columns(df):
    """Per-column null rate, cardinality, dtype, and a sample value.

    Sort by pct_null to find dead columns; filter nunique <= 1 to find
    constants left over from the scope filter.
    """
    rows = []
    for c in df.columns:
        s = df[c]
        nn = s.notna()
        rows.append({
            'column': c,
            'dtype': str(s.dtype),
            'pct_null': round(s.isna().mean() * 100, 1),
            'nunique': s.nunique(dropna=True),
            'sample': s[nn].iloc[0] if nn.any() else None,
        })
    return pd.DataFrame(rows).sort_values('pct_null', ascending=False)


def find_constants(df):
    """Columns with 0 or 1 distinct non-null values — safe drops."""
    return [c for c in df.columns if df[c].nunique(dropna=True) <= 1]


def find_dupe_pairs(df, cols=None):
    """Columns whose non-null values are identical — redundant pairs.

    Catches things like Provider.Full.Name vs Provider.Short.Name and the
    Orientation.Completed / Orientation.Completeded typo pair.
    """
    cols = cols or df.columns
    out = []
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            if df[a].dtype != df[b].dtype:
                continue
            if df[a].equals(df[b]):
                out.append((a, b))
    return out


def prune(df, keep, strict=False):
    """Subset to `keep` and reorder. Warns about names not present."""
    missing = [c for c in keep if c not in df.columns]
    if missing:
        msg = f"{len(missing)} keep-column(s) not in df: {missing}"
        if strict:
            raise KeyError(msg)
        print(f"WARNING: {msg}")
    present = [c for c in keep if c in df.columns]
    dropped = [c for c in df.columns if c not in set(present)]
    print(f"Keeping {len(present)} of {len(df.columns)} columns "
          f"({len(dropped)} dropped)")
    return df[present].copy()


# ---------------------------------------------------------------------------
# Tiered keep-lists
# ---------------------------------------------------------------------------

KEYS = [
    'Participant.Unique.ID',     # person-level, cross-cycle
    'Application.ID',            # row-level, survey join key
    'Worksite.ID',               # placement-level
]

TIME = [
    'Year',
    'Cohort',                    # A / B start timing
]

STATUS = [
    'Enrolled.Flag',
    'Application.Status',
    'NumberofTimesSelected',
    'Date.Selected',
    'Date.Enrolled',
    'Date.DeEnrolled',
    'Date.No.Show',
    'Date.Declined',
]

PLACEMENT = [
    'Organization.Name',
    'Provider.Short.Name',
    'Contract.Short.Name',
    'Contract.Borough',
    'Worksite.Name',
    'Program',
    'Service.Option',
    'Funding.Slot.Type',
    'Special.Project',
]

PARTICIPATION = [
    'Hours.Worked',
    'Total.Hours.Paid',
    'Pay.Rate',
    'Training.Hours',
    'Orientation.Completed',
    'WorkSite.Assignment.Date',
    'Participated.in.any.other.DYCD.funded.Workforce.programs.',
    'DYCD.funded.Workforce.programs',
]

DEMOGRAPHICS = [
    'Age.on.Start.Date',
    'Gender',
    'Preferred.Gender.Identity',
    'Race.Ethnicity',
    'Race', 'YAIP.SYEP.Ethnicity',
    'Borough',
    'Citizenship.Status',
    'Primary.Language',
    'English.Proficiency',
]

EDUCATION = [
    'Educational.Status',
    'Educational.Student.Type',
    'Current.or.Last.Grade',
    'What.type.of.school.did.do.you.attend',
    'School.Name.f',
    'Major',
    'Work.Status',
    'Previous.Work.Experience',
    'GPA',
]

NEED = [
    'NYCHA.Housing',
    'Public.Assistance',
    'Individual.with.Disability',
    'In.Foster.Care.System',
    'Homeless',
    'Runaway',
    'Offender.Court.Involved',
    'Parent',
    'Have.IEP',
    'Number.of.Family.Members',
    'Household.Headed.by',
]

OPTIONAL_CAREER = [
    'Career.Goal.1',
    'Career.Goal.2',
    'Career.Goal.3',
]

OPTIONAL_GEO = [
    'GIS.Community.District',
    'GIS.PUMA',
    'GIS.Council.District',
]

OPTIONAL_FINANCIAL = [
    'Set.a.Savings.Goal.',
    'Savings.Goal.Amount',
    'Do.you.have.a.Bank.Account',
]

# configure this
KEEP = (KEYS + TIME + STATUS + PLACEMENT + PARTICIPATION
        + DEMOGRAPHICS + EDUCATION + NEED + OPTIONAL_CAREER 
        + OPTIONAL_GEO + OPTIONAL_FINANCIAL)


# ---------------------------------------------------------------------------
# Explicit drops, grouped by reason (documentation, not executed)
# ---------------------------------------------------------------------------

DROP_REASONS = {
    'pii': [
        'Email', 'Second.Email', 
    ],
    'constant_after_scope_filter': [
        'Initiative', 'Program.Type', 'SYEP.Programs',
        'SYEP.Programs.Subgroup', 'OY.YY',
    ],
    'redundant_with_kept_column': [
        'Provider.Full.Name', 'Contract.Full.Name',   # short names kept
        'City',                                       # Borough kept
        'GIS.Is.NYCHA',                               # NYCHA.Housing kept
        'GIS.Police.Precinct', 'Police.Precinct.Flag',
        'Contract.City', 'Contract.Zip.Code',
        'School.Name',                                # School.Name.f kept
        'FirstDateSelected', 'LastDateSelected',      # NumberofTimesSelected kept
        'Cycle.Start.Date', 'Cycle.End.Date', 'Cycle',# implied by Year
    ],
    'suspected_import_artifact': [
        'Orientation.Completeded',
        'Member.of.the.Business.LINK..HRA.Cash.Assistance.Program..1',
        'Gender.Based.Domestic.Violence.Victim..1',
    ],
    'payroll_admin': [
        'Payroll.ID', 'Payment.Method', 'Paid.In.System',
        'Federal.Filing.Status', 'Federal.Exemptions',
        'Interested.in.Direct.Deposit.', 'Interested.in.opening.a.savings.account',
        'SYEP.Savings.Disclaimer', 'SYEP.Study.Disclaimer',
    ],
    'free_text_low_value': [
        'Miscellaneous', 'Other', 'Preferred.Gender.Other', 'Gender.Pronoun.Other',
        'Sexual.Orientation.Other', 'Other...Race', 'Other...Asian.Origin',
        'Other...Native.Hawaiian.or.Pacific.Islander.origin',
        'Other...Hispanic.or.Latinx.e.a.o.origin',
        'YAIP.Area.of.Career.Interest.Other', 'Barrier..Other.Description',
        'Other.School', 'Other.Languages',
    ],
    'income_detail_collapsed_to_public_assistance': [
        'Family.Income.Last.Year', 'Total.income.for.the.last.12.months',
        'Family.Assistance..Formerly.known.as.AFDC.', 'Food.Stamps', 'S.S.I.',
        'Safety.Net..Formerly.known.as.HR.', 'Unemployment',
        'Workman.s.Compensation',
    ],
    'survey_operational': [
        'How.did.you.hear.about.us.', 'Interested.in.SYEP.Pride.',
        'Please.select..Yes..if.you.would.like.to.receive.text.updates',
        'Do.you.have.access.to.an.electronic.device.with.internet.accessibility.',
        'Are.you.familiar.with.any.of.these.skills.',
        'Referring.Agency', 'Online.Application',
    ],
    'sparse_or_unused': [
        'Contract..', 'Career.Goal..pre.2016.', 'Selected.Languages', 'Served.in.Military',
        'Asian.Origin', 'Native.Hawaiian.or.Pacific.Islander.origin',
        'Hispanic.or.Latinx.e.a.o.origin', 'Covered.By.HealthCare',
        'Sexual.Orientation', 'Gender.Pronoun', 'NYCHA.MAP',
        'ACS.Preventative.Services', 'Current.DOE...D79.Student.',
        'Member.of.the.Business.LINK..HRA.Cash.Assistance.Program.',
        'CUNY.School', 'DOE.School', 'SUNY.School', 'Charter.School',
        'Are.you.enrolled.in.a.DOE..Alternative.school.',
        'Are.you.enrolled.in.a.DOE..Transitional.Housing.program.',
        'Are.you.enrolled.in.a.DOE..Access.Program',
        'GIS.Is.Valid.Address', 'GIS.Data.Quality', 'Geo.Code.Date',
        'GIS.Assembly.District', 'GIS.NYCHA.Development',
        'GIS.Fed.Congressional.District',
        'Date.Submitted', 'Date.Completed', 'Date.Preemployment',
        'Date.Pending', 'Date.Rejected',
        'Orientation.Completed.By',
        'Capricorn.Intake.ID', 'Ethnicity', 'Resume.Referrals',
    ],
    'barriers': [
        'Barrier..Have.not.graduated.from.HS.GED.HSE',
        'Barrier..Homeless',
        'Barrier..Experience.in.Foster.Care',
        'Barrier..Physical.Medical.Disability',
        'Barrier..Limited.Literacy.or.Math.Skills',
        'Barrier..Runaway',
        'Barrier..Pregnancy.Parenting',
        'Barrier..Criminal.Record',
        'Barrier..Limited.Work.History..less.than.3.months.',
        'Barrier..RHY..Runaway.and.Homeless.Youth..LGBTQ',
        'Barrier..Mental.Health.challenge',
        'Barrier..Alcohol.Substance.Abuse',
        'Barrier..Family.Responsibilities',
        'Barrier..Unstable.Housing',
        'Barrier..Other',
        'Barier..None',
    ]
}