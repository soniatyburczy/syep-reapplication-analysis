"""
attrition.py -- exact counts at every filter, index years 2022-2024.
Each row is a filter; `n` is what survives it, `dropped` is what it removed.
"""
import numpy as np
import pandas as pd

MERGED   = "data/to_use/merged.csv"
AGE_CAP  = 24
INDEX_YEARS = (2022, 2023, 2024)


def waterfall(df, steps, by_year=True):
    """steps: list of (label, boolean mask aligned to df.index)."""
    rows, keep = [], pd.Series(True, index=df.index)
    rows.append({"step": "all rows in merged.csv", "n": int(keep.sum()),
                 "dropped": 0, "people": df.loc[keep, "Participant.Unique.ID"].nunique()})
    for label, mask in steps:
        before = int(keep.sum())
        keep = keep & mask.reindex(df.index).fillna(False)
        rows.append({"step": label, "n": int(keep.sum()),
                     "dropped": before - int(keep.sum()),
                     "people": df.loc[keep, "Participant.Unique.ID"].nunique()})
    tbl = pd.DataFrame(rows)
    tbl["pct_of_start"] = (tbl["n"] / tbl.loc[0, "n"]).round(4)

    per_year = None
    if by_year:
        per_year = pd.DataFrame(index=sorted(INDEX_YEARS))
        k = pd.Series(True, index=df.index)
        for label, mask in steps:
            k = k & mask.reindex(df.index).fillna(False)
            per_year[label] = df.loc[k].groupby("Year").size().reindex(per_year.index).fillna(0).astype(int)
    return tbl, per_year


def main():
    df = pd.read_csv(MERGED, encoding="utf-8-sig", low_memory=False)
    df["age_on_start"] = pd.to_numeric(df["age_on_start"], errors="coerce")

    # ---- t+1 application lookup, built on the ALL-OPTIONS SYEP frame ----
    # This must be built before any Older Youth / enrolled filtering,
    # otherwise a returner who switched option scores as a non-returner.
    applied_next = set(
        zip(df["Participant.Unique.ID"], df["Year"] - 1)
    )  # (person, t) pairs where person applied in t+1

    pid, yr = df["Participant.Unique.ID"], df["Year"]
    returned = pd.Series(
        [(p, y) in applied_next for p, y in zip(pid, yr)], index=df.index
    )

    steps = [
        ("index years 2022-2024",        yr.isin(INDEX_YEARS)),
        ("enrolled in year t",           df["enrolled"].astype(bool)),
        ("Older Youth",                  df["service_option"].eq("Older Youth")),
        ("non-null Participant.Unique.ID", pid.notna()),
        ("deduped (person, year)",       ~df.duplicated(["Participant.Unique.ID", "Year"], keep="first")),
        ("age known",                    df["age_on_start"].notna()),
        (f"age-eligible at t+1 (<{AGE_CAP})", df["age_on_start"] < AGE_CAP),
    ]

    tbl, per_year = waterfall(df, steps)

    print("\nATTRITION WATERFALL\n" + "=" * 72)
    print(tbl.to_string(index=False))
    print("\nSurviving rows by index year, cumulative through each filter:")
    print(per_year.T.to_string())

    analysis = df.loc[
        np.logical_and.reduce([m.reindex(df.index).fillna(False) for _, m in steps])
    ].copy()
    analysis["returned"] = returned.loc[analysis.index]

    print(f"\nANALYSIS FRAME: {len(analysis):,} person-years, "
          f"{analysis['Participant.Unique.ID'].nunique():,} people")
    print("\nreturn rate by index year:")
    print(analysis.groupby("Year")["returned"].agg(["size", "sum", "mean"]).round(3).to_string())

    analysis.to_csv("data/to_use/analysis_frame.csv", encoding="utf-8-sig", index=False)
    tbl.to_csv("data/stats/attrition_waterfall.csv", index=False)
    return tbl, analysis


if __name__ == "__main__":
    main()