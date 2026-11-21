"""
Scans data/raw/tle_history/*.json and reports:
- how many TLE records each satellite actually got
- which files are error responses (rate-limited / failed) instead of real data
- totals across the whole catalog
Optionally refetches failed IDs with --refetch.
"""

import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from spacetrack import SpaceTrackClient

from data_collection.fetch_spaceTrack import fetch_one  # reuse the fixed, validated fetch logic

TLE_DIR = Path("data/raw/tle_history")
REFETCH_DELAY_SEC = 15  # deliberately slow — you already hit the limit once today


def load_records(path: Path):
    """Returns (records_list_or_None, error_message_or_None)."""
    raw = path.read_text()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None, f"invalid JSON: {raw[:80]}"

    if isinstance(data, str):
        try:
            data = json.loads(data)
        except json.JSONDecodeError:
            return None, f"double-encoded non-JSON: {data[:80]}"

    if isinstance(data, dict) and "error" in data:
        return None, data["error"]

    if isinstance(data, list) and len(data) == 1 and isinstance(data[0], dict) and "error" in data[0]:
        return None, data[0]["error"]

    if not isinstance(data, list):
        return None, f"unexpected shape: {type(data)}"

    return data, None


def scan() -> tuple[list, list, list]:
    """Returns (ok, failed, empty)."""
    files = sorted(TLE_DIR.glob("*.json"))
    if not files:
        print(f"No files found in {TLE_DIR}")
        return [], [], []

    ok, failed, empty = [], [], []
    total_records = 0

    for f in files:
        norad_id = f.stem
        records, error = load_records(f)

        if error:
            failed.append((norad_id, error))
            continue

        count = len(records)
        if count == 0:
            empty.append(norad_id)
        else:
            ok.append((norad_id, count))
            total_records += count

    print(f"\n{'='*60}")
    print(f"Scanned {len(files)} files")
    print(f"  OK:     {len(ok)}  ({total_records} total TLE records)")
    print(f"  Empty:  {len(empty)}")
    print(f"  Failed: {len(failed)}")
    print(f"{'='*60}\n")

    if ok:
        counts = sorted(ok, key=lambda x: x[1])
        print(f"Record count range: min={counts[0][1]} ({counts[0][0]}), max={counts[-1][1]} ({counts[-1][0]})")

    if failed:
        print(f"\n--- FAILED (need re-fetch — these are NOT valid cached data) ---")
        for norad_id, err in failed:
            print(f"  {norad_id}: {err[:100]}")

    if empty:
        print(f"\n--- EMPTY (0 records — check if NORAD ID is valid) ---")
        for norad_id in empty:
            print(f"  {norad_id}")

    return ok, failed, empty


def refetch_failed(failed_ids: list[str]):
    if not failed_ids:
        print("Nothing to refetch.")
        return

    load_dotenv()
    identity = os.environ.get("SPACETRACK_USERNAME")
    password = os.environ.get("SPACETRACK_PASSWORD")

    expected_keys = None
    good_files = list(TLE_DIR.glob("*.json"))
    if good_files:
        sample, _ = load_records(good_files[0])
        if sample:
            expected_keys = set(sample[0].keys())

    print(f"\nRefetching {len(failed_ids)} failed IDs, {REFETCH_DELAY_SEC}s apart...\n")

    with SpaceTrackClient(identity=identity, password=password) as st:
        for i, norad_id in enumerate(failed_ids, 1):
            bad_path = TLE_DIR / f"{norad_id}.json"
            if bad_path.exists():
                bad_path.unlink()
                print(f"  deleted stale {norad_id}.json")

            print(f"[{i}/{len(failed_ids)}] fetching {norad_id}...")
            count = fetch_one(st, int(norad_id))

            if count:
                records, _ = load_records(TLE_DIR / f"{norad_id}.json")
                got_keys = set(records[0].keys())
                if expected_keys and got_keys != expected_keys:
                    print(f"  ⚠ {norad_id}: schema mismatch — extra {got_keys - expected_keys}, "
                          f"missing {expected_keys - got_keys}")
                else:
                    print(f"  ✓ {norad_id}: {count} records, schema matches")
            else:
                print(f"  ✗ {norad_id}: still failed")

            if i < len(failed_ids):
                time.sleep(REFETCH_DELAY_SEC)


if __name__ == "__main__":
    ok, failed, empty = scan()

    if "--refetch" in sys.argv:
        refetch_failed([nid for nid, _ in failed])