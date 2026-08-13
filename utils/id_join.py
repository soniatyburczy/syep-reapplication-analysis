import pandas as pd

def clean_id(s: pd.Series) -> pd.Series:
    """Normalize an ID column to stripped strings, dropping float artifacts."""
    return (
        s.astype(str)
         .str.strip()
         .str.replace(r'\.0$', '', regex=True)
    )


def add_unique_id(
    df,
    id_map,
    df_key,
    map_key,
    map_value,
    out_col='Participant.Unique.ID',
    verbose=True,
):
    """
    Map a person-level key onto `df` via a lookup table.

    df        : frame to add the ID to (applications, surveys, etc.)
    id_map    : lookup frame containing key -> unique-id pairs
    df_key    : name of the join column in `df`        e.g. 'Application.ID'
                                                       or 'Recipient - custom_value4'
    map_key   : name of the join column in `id_map`    e.g. 'ApplicationOnlineID'
    map_value : name of the unique-id column in `id_map'  e.g. 'SSN_Encoded'
    out_col   : name of the column to create

    Returns (df_with_id, report_dict). Does not mutate the inputs.
    """
    df = df.copy()
    id_map = id_map.copy()

    df[df_key] = clean_id(df[df_key])
    id_map[map_key] = clean_id(id_map[map_key])

    # restrict the lookup to keys that actually appear in df, so out-of-scope
    # duplicate keys can't trigger InvalidIndexError or a silent bad pick
    in_scope = id_map[map_key].isin(df[df_key])
    id_map_matched = id_map[in_scope]

    # any remaining key with >1 distinct value is a genuine conflict
    conflicts = (
        id_map_matched.groupby(map_key)[map_value]
        .nunique()
        .loc[lambda s: s > 1]
        .index
        .tolist()
    )

    lookup = (
        id_map_matched[~id_map_matched[map_key].isin(conflicts)]
        .drop_duplicates(subset=map_key)
        .set_index(map_key)[map_value]
    )

    df[out_col] = df[df_key].map(lookup)

    unmapped_by_year = (
        df.groupby('Year')[out_col]
        .agg(
            total='size',
            unmapped=lambda s: s.isna().sum(),
        )
    )

    unmapped_by_year['unmapped_pct'] = (
        100 * unmapped_by_year['unmapped'] / unmapped_by_year['total']
    ).round(2)
        
    report = {
        'rows': len(df),
        'mapped': int(df[out_col].notna().sum()),
        'unmapped': int(df[out_col].isna().sum()),
        'unmapped_pct': round(df[out_col].isna().mean() * 100, 3),
        'unmapped_ids': int(df.loc[df[out_col].isna(), df_key].nunique()),
        'unmapped_by_year': unmapped_by_year,
        'lookup_size': len(lookup),
        'map_rows_dropped_out_of_scope': int((~in_scope).sum()),
        'conflicting_keys': conflicts,
    }

    if verbose:
        print(f"{report['mapped']:,} mapped / {report['rows']:,} rows "
              f"({report['unmapped']:,} unmapped, {report['unmapped_pct']}%, "
              f"{report['unmapped_ids']:,} distinct IDs)")
        if conflicts:
            print(f"⚠ {len(conflicts)} in-scope key(s) map to multiple "
                  f"{map_value} values — excluded from lookup: {conflicts[:10]}")

    return df, report