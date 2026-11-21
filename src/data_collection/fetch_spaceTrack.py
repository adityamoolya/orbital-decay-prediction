"""
Pulls full lifetime TLE history (gp_history) for every NORAD ID in the
decayed-satellites catalog, once each, caching raw responses locally so
no object is ever re-queried.
"""

import os
import sys
import json
import time
import logging
from pathlib import Path

import pandas as pd
from spacetrack import SpaceTrackClient
import spacetrack.operators as op
from dotenv import load_dotenv
load_dotenv()
# ---- config ----
CATALOG_CSV = Path("data/raw/decayed_satellites.csv")
NORAD_COLUMN = "NORAD Catalog Number"         
OUT_DIR = Path("data/raw/tle_history")
MAX_RETRIES = 3
RETRY_BACKOFF_SEC = 10

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


def load_norad_ids(csv_path: Path) -> list[int]:
    df = pd.read_csv(csv_path)
    if NORAD_COLUMN not in df.columns:
        log.error(f"Column '{NORAD_COLUMN}' not found. Available columns: {list(df.columns)}")
        sys.exit(1)
    return df[NORAD_COLUMN].dropna().astype(int).unique().tolist()


def already_fetched(norad_id: int) -> bool:
    return (OUT_DIR / f"{norad_id}.json").exists()


def fetch_one(st: SpaceTrackClient, norad_id: int) -> int | None:
    """Returns number of TLE records fetched, or None on failure."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            data = st.gp_history(
                norad_cat_id=norad_id,
                orderby="epoch asc",
                format="json",
            )

            # --- validate before writing anything to disk ---
            if isinstance(data, str):
                data = json.loads(data)  # handle double-encoded responses

            if isinstance(data, dict) and "error" in data:
                raise RuntimeError(f"API error: {data['error']}")
            if isinstance(data, list) and len(data) == 1 and isinstance(data[0], dict) and "error" in data[0]:
                raise RuntimeError(f"API error: {data[0]['error']}")
            if not isinstance(data, list):
                raise RuntimeError(f"Unexpected response shape: {type(data)}")
            # --- end validation ---

            if not data:
                log.warning(f"{norad_id}: 0 records returned — check ID validity")
                return 0

            out_path = OUT_DIR / f"{norad_id}.json"
            out_path.write_text(json.dumps(data, indent=2))
            log.info(f"{norad_id}: {len(data)} records saved")
            return len(data)

        except Exception as e:
            log.warning(f"{norad_id}: attempt {attempt}/{MAX_RETRIES} failed — {e}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF_SEC * attempt)
            else:
                log.error(f"{norad_id}: giving up after {MAX_RETRIES} attempts")
                return None


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    identity = os.environ.get("SPACETRACK_USERNAME")
    password = os.environ.get("SPACETRACK_PASSWORD")
    if not identity or not password:
        log.error("Set SPACETRACK_USERNAME and SPACETRACK_PASSWORD env vars")
        sys.exit(1)

    norad_ids = load_norad_ids(CATALOG_CSV)
    log.info(f"Loaded {len(norad_ids)} NORAD IDs from catalog")

    pending = [nid for nid in norad_ids if not already_fetched(nid)]
    skipped = len(norad_ids) - len(pending)
    log.info(f"{skipped} already cached locally, {len(pending)} to fetch")

    results = {"ok": 0, "empty": 0, "failed": 0}

    with SpaceTrackClient(identity=identity, password=password) as st:
        for i, norad_id in enumerate(pending, 1):
            log.info(f"[{i}/{len(pending)}] fetching {norad_id}")
            count = fetch_one(st, norad_id)
            if count is None:
                results["failed"] += 1
            elif count == 0:
                results["empty"] += 1
            else:
                results["ok"] += 1

    log.info(f"Done. ok={results['ok']} empty={results['empty']} failed={results['failed']}")
    if results["failed"]:
        log.warning("Some IDs failed — re-run the script; already-cached ones will be skipped automatically")


if __name__ == "__main__":
    main()