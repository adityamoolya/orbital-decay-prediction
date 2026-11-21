
from pathlib import Path
import pandas as pd

BASE_DIR = Path("/run/media/adi/New Volume/El/satelite/data/preprocessed/")
CSV_FILE = BASE_DIR / "tle_features.csv"
PARQUET_FILE = BASE_DIR / "tle_features.parquet"

TLE_COLUMNS = [
    "CCSDS_OMM_VERS", "COMMENT", "CREATION_DATE", "ORIGINATOR",
    "OBJECT_NAME", "OBJECT_ID", "CENTER_NAME", "REF_FRAME",
    "TIME_SYSTEM", "MEAN_ELEMENT_THEORY", "EPOCH", "MEAN_MOTION",
    "ECCENTRICITY", "INCLINATION", "RA_OF_ASC_NODE",
    "ARG_OF_PERICENTER", "MEAN_ANOMALY", "EPHEMERIS_TYPE",
    "CLASSIFICATION_TYPE", "NORAD_CAT_ID", "ELEMENT_SET_NO",
    "REV_AT_EPOCH", "BSTAR", "MEAN_MOTION_DOT",
    "MEAN_MOTION_DDOT", "SEMIMAJOR_AXIS", "PERIOD", "APOAPSIS",
    "PERIAPSIS", "OBJECT_TYPE", "RCS_SIZE", "COUNTRY_CODE",
    "LAUNCH_DATE", "SITE", "DECAY_DATE", "FILE", "GP_ID",
    "TLE_LINE0", "TLE_LINE1", "TLE_LINE2",
]

NUMERIC_COLUMNS = [
    "MEAN_MOTION", "ECCENTRICITY", "INCLINATION",
    "RA_OF_ASC_NODE", "ARG_OF_PERICENTER", "MEAN_ANOMALY",
    "NORAD_CAT_ID", "ELEMENT_SET_NO", "REV_AT_EPOCH",
    "BSTAR", "MEAN_MOTION_DOT", "MEAN_MOTION_DDOT",
    "SEMIMAJOR_AXIS", "PERIOD", "APOAPSIS", "PERIAPSIS",
]

DATE_COLUMNS = [
    "CREATION_DATE", "EPOCH", "LAUNCH_DATE", "DECAY_DATE",
    "CATALOG_DECAY_DATE",
]


def check_file(path):
    print(f"\n{'=' * 65}")
    print(f"CHECKING: {path}")

    if not path.exists():
        print("ERROR: File does not exist.")
        return None

    try:
        if path.suffix == ".csv":
            df = pd.read_csv(path)
        else:
            df = pd.read_parquet(path)
    except Exception as exc:
        print(f"CORRUPT / UNREADABLE FILE: {exc}")
        return None

    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")

    # 1. Check expected columns
    missing_columns = set(TLE_COLUMNS) - set(df.columns)
    extra_columns = set(df.columns) - set(TLE_COLUMNS)

    print("\n--- COLUMN CHECK ---")
    print("Missing columns:", sorted(missing_columns) or "None")
    print("Extra columns:", sorted(extra_columns) or "None")

    # 2. Check missing values
    print("\n--- MISSING VALUES ---")
    missing = df.isna().sum()
    blanks = (
        df.select_dtypes(include="object")
        .apply(lambda col: col.astype(str).str.strip().eq(""))
        .sum()
    )

    for col in df.columns:
        count = int(missing[col] + blanks.get(col, 0))
        if count:
            print(f"{col}: {count:,} missing/blank")

    # 3. Check numeric fields
    print("\n--- INVALID NUMERIC VALUES ---")
    for col in NUMERIC_COLUMNS:
        if col not in df.columns:
            continue

        raw = df[col]
        converted = pd.to_numeric(raw, errors="coerce")
        invalid = raw.notna() & raw.astype(str).str.strip().ne("") & converted.isna()

        if invalid.any():
            print(f"{col}: {invalid.sum()} invalid values")
            print("Examples:", raw[invalid].head(5).tolist())

    # 4. Check date fields
    print("\n--- INVALID DATE VALUES ---")
    for col in DATE_COLUMNS:
        if col not in df.columns:
            continue

        raw = df[col]
        converted = pd.to_datetime(raw, errors="coerce", utc=True)
        invalid = raw.notna() & raw.astype(str).str.strip().ne("") & converted.isna()

        if invalid.any():
            print(f"{col}: {invalid.sum()} invalid dates")
            print("Examples:", raw[invalid].head(5).tolist())

    # 5. Check orbital value ranges
    print("\n--- ORBITAL RANGE CHECK ---")

    range_checks = {
        "ECCENTRICITY": (0, 1),
        "INCLINATION": (0, 180),
        "MEAN_MOTION": (0, float("inf")),
        "PERIOD": (0, float("inf")),
        "SEMIMAJOR_AXIS": (0, float("inf")),
    }

    for col, (lower, upper) in range_checks.items():
        if col not in df.columns:
            continue

        values = pd.to_numeric(df[col], errors="coerce")
        invalid = values.notna() & (
            (values < lower) | (values > upper)
        )

        if invalid.any():
            print(f"{col}: {invalid.sum()} out-of-range values")
            print("Examples:", df.loc[invalid, col].head(5).tolist())

    # 6. Check duplicate records without removing them
    print("\n--- DUPLICATE CHECK ---")
    if {"NORAD_CAT_ID", "EPOCH"}.issubset(df.columns):
        duplicates = df.duplicated(
            subset=["NORAD_CAT_ID", "EPOCH"], keep=False
        )
        print(f"Duplicate NORAD_CAT_ID + EPOCH rows: {duplicates.sum():,}")
    else:
        print("Cannot check: NORAD_CAT_ID or EPOCH missing.")

    # 7. Check TLE line checksums, when present
    print("\n--- TLE CHECKSUM CHECK ---")

    def valid_checksum(line):
        if not isinstance(line, str) or len(line) < 2:
            return None

        checksum_char = line[-1]
        if not checksum_char.isdigit():
            return False

        total = sum(int(c) for c in line[:-1] if c.isdigit())
        total += line[:-1].count("-")
        return total % 10 == int(checksum_char)

    for col in ["TLE_LINE1", "TLE_LINE2"]:
        if col not in df.columns:
            continue

        populated = df[col].dropna().astype(str)
        populated = populated[populated.str.strip().ne("")]
        results = populated.map(valid_checksum)

        checked = results.dropna()
        invalid_count = int((checked == False).sum())

        print(
            f"{col}: {len(checked)} checked, "
            f"{invalid_count} invalid checksums"
        )

    return df


csv_df = check_file(CSV_FILE)
parquet_df = check_file(PARQUET_FILE)

# 8. Compare exported files
print(f"\n{'=' * 65}")
print("CSV vs PARQUET COMPARISON")

if csv_df is not None and parquet_df is not None:
    print("Same row count:", len(csv_df) == len(parquet_df))
    print("Same columns:", set(csv_df.columns) == set(parquet_df.columns))

    if len(csv_df) == len(parquet_df):
        print(
            "Same column order:",
            list(csv_df.columns) == list(parquet_df.columns)
        )

    print("\nValidation finished. No data has been modified.")



# ============================================================
# MAP DECAY DATES FROM CELESTRAK CATALOG
# ============================================================

CATALOG_FILE = Path("../../data/raw/decayed_satellites.csv")

catalog = pd.read_csv(CATALOG_FILE)

# Prepare catalog mapping
catalog["NORAD_CAT_ID"] = pd.to_numeric(
    catalog["NORAD Catalog Number"],
    errors="coerce"
)

catalog["CATALOG_DECAY_DATE"] = catalog["Decay Date"]

catalog = catalog[
    ["NORAD_CAT_ID", "CATALOG_DECAY_DATE"]
].dropna(subset=["NORAD_CAT_ID"])

catalog = catalog.drop_duplicates(
    subset=["NORAD_CAT_ID"]
)

for name, df in [
    ("CSV", csv_df),
    ("PARQUET", parquet_df),
]:
    if df is None:
        continue

    df["NORAD_CAT_ID"] = pd.to_numeric(
        df["NORAD_CAT_ID"],
        errors="coerce"
    )

    # Remove previous mapping column if script is rerun
    df.drop(columns=["CATALOG_DECAY_DATE"], errors="ignore", inplace=True)

    df = df.merge(
        catalog,
        on="NORAD_CAT_ID",
        how="left",
        validate="many_to_one"
    )

    matched = df["CATALOG_DECAY_DATE"].notna().sum()

    if name == "CSV":
        df.to_csv(CSV_FILE, index=False)
        csv_df = df
    else:
        df.to_parquet(PARQUET_FILE, index=False)
        parquet_df = df

    print(
        f"{name}: {matched:,}/{len(df):,} records "
        f"mapped to CelesTrak decay dates"
    )

print("Decay date mapping completed.")