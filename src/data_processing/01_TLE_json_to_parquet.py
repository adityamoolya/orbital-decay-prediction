"""
Extract and consolidate raw TLE JSON files into Parquet and CSV datasets.

- Reads every JSON file in data/raw/tle_history.
- Handles standard JSON lists as well as string-encoded JSON payloads.
- Preserves all 40 original API fields and exact raw values without any conversions or alterations.
- Outputs datasets directly to:
    - data/preprocessed/tle_features.parquet
    - data/preprocessed/tle_data_205.csv
    - data/preprocessed/tle_features.csv
"""

import json
import logging
from pathlib import Path
import pandas as pd

# --------------------------------------------------
# PATH CONFIGURATION
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent.parent
RAW_DIR = BASE_DIR / "data" / "raw" / "tle_history"
OUTPUT_DIR = BASE_DIR / "data" / "preprocessed"

PARQUET_OUTPUT = OUTPUT_DIR / "tle_features.parquet"
CSV_OUTPUT_205 = OUTPUT_DIR / "tle_data_205.csv"
CSV_OUTPUT_FEATURES = OUTPUT_DIR / "tle_features.csv"

# --------------------------------------------------
# LOGGING
# --------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# --------------------------------------------------
# SCHEMA DEFINITION (40 Space-Track TLE Fields)
# --------------------------------------------------
TLE_COLUMNS = [
    "CCSDS_OMM_VERS",
    "COMMENT",
    "CREATION_DATE",
    "ORIGINATOR",
    "OBJECT_NAME",
    "OBJECT_ID",
    "CENTER_NAME",
    "REF_FRAME",
    "TIME_SYSTEM",
    "MEAN_ELEMENT_THEORY",
    "EPOCH",
    "MEAN_MOTION",
    "ECCENTRICITY",
    "INCLINATION",
    "RA_OF_ASC_NODE",
    "ARG_OF_PERICENTER",
    "MEAN_ANOMALY",
    "EPHEMERIS_TYPE",
    "CLASSIFICATION_TYPE",
    "NORAD_CAT_ID",
    "ELEMENT_SET_NO",
    "REV_AT_EPOCH",
    "BSTAR",
    "MEAN_MOTION_DOT",
    "MEAN_MOTION_DDOT",
    "SEMIMAJOR_AXIS",
    "PERIOD",
    "APOAPSIS",
    "PERIAPSIS",
    "OBJECT_TYPE",
    "RCS_SIZE",
    "COUNTRY_CODE",
    "LAUNCH_DATE",
    "SITE",
    "DECAY_DATE",
    "FILE",
    "GP_ID",
    "TLE_LINE0",
    "TLE_LINE1",
    "TLE_LINE2",
]


def load_raw_tle_records(raw_dir: Path):
    """
    Iterate through all JSON files in the raw directory,
    handling direct lists/dicts as well as string-encoded JSON payloads.
    Returns a list of raw record dictionaries.
    """
    json_files = sorted(raw_dir.glob("*.json"))
    logger.info("Found %d JSON files in %s", len(json_files), raw_dir)

    all_records = []
    failed_files = []

    for file_path in json_files:
        try:
            with file_path.open("r", encoding="utf-8") as f:
                content = json.load(f)

            # Handle case where file contains a string-encoded JSON
            if isinstance(content, str):
                content = json.loads(content)

            # Normalize single object into list
            if isinstance(content, dict):
                records = [content]
            elif isinstance(content, list):
                records = content
            else:
                logger.warning("%s: Unrecognized data format %s", file_path.name, type(content))
                failed_files.append(file_path.name)
                continue

            all_records.extend(records)

        except Exception as exc:
            logger.error("Failed to parse %s: %s", file_path.name, exc)
            failed_files.append(file_path.name)

    logger.info("Total raw records accumulated: %d", len(all_records))
    if failed_files:
        logger.warning("%d files failed to parse: %s", len(failed_files), failed_files)

    return all_records


def build_and_export_dataset():
    """
    Loads raw records, builds the DataFrame with exact values and columns,
    and writes Parquet and CSV files.
    """
    logger.info("Starting raw TLE extraction and consolidation...")

    if not RAW_DIR.exists():
        raise FileNotFoundError(f"Raw directory does not exist: {RAW_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    records = load_raw_tle_records(RAW_DIR)
    if not records:
        raise ValueError("No records found to process.")

    # Create DataFrame with exact column ordering; preserving raw values completely
    df = pd.DataFrame(records, columns=TLE_COLUMNS)

    logger.info("DataFrame created with shape: %s", df.shape)

    # 1. Export Parquet
    logger.info("Saving to Parquet: %s", PARQUET_OUTPUT)
    df.to_parquet(PARQUET_OUTPUT, index=False, engine="pyarrow")

    # 2. Export CSV (tle_data_205.csv)
    logger.info("Saving to CSV: %s", CSV_OUTPUT_205)
    df.to_csv(CSV_OUTPUT_205, index=False)

    # 3. Export CSV (tle_features.csv for visual inspection)
    logger.info("Saving to CSV: %s", CSV_OUTPUT_FEATURES)
    df.to_csv(CSV_OUTPUT_FEATURES, index=False)

    logger.info("=" * 60)
    logger.info("COMPLETED SUCCESSFULLY")
    logger.info("Total rows: %d", len(df))
    logger.info("Total columns: %d", len(df.columns))
    logger.info("Parquet: %s", PARQUET_OUTPUT)
    logger.info("CSV 1:   %s", CSV_OUTPUT_205)
    logger.info("CSV 2:   %s", CSV_OUTPUT_FEATURES)
    logger.info("=" * 60)


if __name__ == "__main__":
    build_and_export_dataset()
