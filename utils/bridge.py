import pandas as pd
import re
from rapidfuzz import process, fuzz

df = pd.read_csv('data/to_use/analysis_frame.csv')

# =========================================================
# 1. Normalize worksite names
# =========================================================

def normalize_worksite_name(x):
    if pd.isna(x):
        return ""

    x = str(x).upper().strip()

    # Standardize punctuation
    x = re.sub(r"[&]", " AND ", x)
    x = re.sub(r"[-_/]", " ", x)
    x = re.sub(r"[^\w\s]", "", x)
    x = re.sub(r"\s+", " ", x).strip()

    # Common abbreviations
    replacements = {
        r"\bCTR\b": "CENTER",
        r"\bCT\b": "CENTER",
        r"\bCOMM\b": "COMMUNITY",
        r"\bSCH\b": "SCHOOL",
        r"\bHS\b": "HIGH SCHOOL",
        r"\bST\b": "SAINT",
        r"\bMT\b": "MOUNT",
    }

    for pattern, replacement in replacements.items():
        x = re.sub(pattern, replacement, x)

    return x


df["worksite_name_norm"] = (
    df["worksite_name"]
    .map(normalize_worksite_name)
)


# =========================================================
# 2. Reduce dataset to one row per Year + worksite_id
# =========================================================

sites = (
    df[
        [
            "Year",
            "worksite_id",
            "worksite_name",
            "worksite_name_norm",
        ]
    ]
    .drop_duplicates(["Year", "worksite_id"])
    .copy()
)

sites = sites[
    sites["worksite_name_norm"].ne("")
].copy()


# =========================================================
# 3. Create a blocking key
#
# This prevents us from comparing every worksite against
# every other worksite.
# =========================================================

GENERIC_WORDS = {
    "THE",
    "CENTER",
    "SCHOOL",
    "COMMUNITY",
    "PROGRAM",
    "INC",
    "LLC",
}


def blocking_key(name):
    words = [
        word
        for word in name.split()
        if word not in GENERIC_WORDS
    ]

    # First few meaningful characters
    return "".join(words)[:8]


sites["block"] = (
    sites["worksite_name_norm"]
    .map(blocking_key)
)


# =========================================================
# 4. Build candidate names by block
# =========================================================

block_to_names = (
    sites
    .groupby("block")["worksite_name_norm"]
    .unique()
    .to_dict()
)


# =========================================================
# 5. Fuzzy-match names across years
# =========================================================

# We match NAME -> NAME rather than row -> row.
# This makes the operation much cheaper.

name_years = (
    sites
    .groupby("worksite_name_norm")["Year"]
    .agg(set)
    .to_dict()
)

canonical_map = {}
match_scores = {}

unique_names = sites["worksite_name_norm"].unique()

for name in unique_names:

    block = blocking_key(name)

    candidates = block_to_names.get(block, [])

    # Only consider names that occur in another year.
    years = name_years[name]

    candidates = [
        candidate
        for candidate in candidates
        if candidate != name
        and not years.isdisjoint(name_years[candidate])
    ]

    if not candidates:
        canonical_map[name] = name
        match_scores[name] = 100
        continue

    result = process.extractOne(
        name,
        candidates,
        scorer=fuzz.token_set_ratio,
        score_cutoff=92,
    )

    if result is None:
        canonical_map[name] = name
        match_scores[name] = 0
        continue

    matched_name, score, _ = result

    # Use the longer spelling as the readable canonical name
    canonical_map[name] = max(
        name,
        matched_name,
        key=len
    )

    match_scores[name] = score


# =========================================================
# 6. Assign canonical name to each yearly worksite
# =========================================================

sites["worksite_name_canonical"] = (
    sites["worksite_name_norm"]
    .map(canonical_map)
)


sites["fuzzy_match_score"] = (
    sites["worksite_name_norm"]
    .map(match_scores)
)


# =========================================================
# 7. Create a stable cross-year worksite ID
# =========================================================

# Every canonical name gets a stable ID.

canonical_names = (
    sites["worksite_name_canonical"]
    .dropna()
    .unique()
)

canonical_to_id = {
    name: f"WS_{i:05d}"
    for i, name in enumerate(
        sorted(canonical_names),
        start=1
    )
}

sites["worksite_cluster_id"] = (
    sites["worksite_name_canonical"]
    .map(canonical_to_id)
)


# =========================================================
# 8. Create the bridge table
# =========================================================

worksite_bridge = sites[
    [
        "Year",
        "worksite_id",
        "worksite_name",
        "worksite_name_canonical",
        "worksite_cluster_id",
        "fuzzy_match_score",
    ]
].copy()


# =========================================================
# 9. Merge the bridge back onto the full dataset
# =========================================================

df = df.merge(
    worksite_bridge[
        [
            "Year",
            "worksite_id",
            "worksite_name_canonical",
            "worksite_cluster_id",
            "fuzzy_match_score",
        ]
    ],
    on=["Year", "worksite_id"],
    how="left",
    validate="many_to_one",
)


# =========================================================
# 10. Save a review file
# =========================================================

worksite_bridge.sort_values(
    ["worksite_name_canonical", "Year"]
).to_csv(
    "worksite_cross_year_bridge_review.csv",
    index=False
)