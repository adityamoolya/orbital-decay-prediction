# this script will fetch all the sattelite catalgoue number of decayed satellites and store in data/raw/decayed_satellites.csv
import requests
import pandas as pd
from io import StringIO
from pathlib import Path

URL = "https://celestrak.org/satcat/decayed-with-last.php"
OUTPUT = Path("data/raw/decayed_satellites.csv")


def download_decayed_satellites():
    print("Fetching CelesTrak decayed satellite data...")

    response = requests.get(
        URL,
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=30
    )
    response.raise_for_status()

    if "text/html" not in response.headers.get("Content-Type", ""):
        raise RuntimeError("Unexpected response format")

    print("Parsing HTML tables...")

    tables = pd.read_html(StringIO(response.text))

    if not tables:
        raise RuntimeError("No tables found on the webpage")

    # Select the largest table
    df = max(tables, key=len)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT, index=False)

    print("\nDownload successful!")
    print(f"Total records: {len(df)}")
    print(f"Total columns: {len(df.columns)}")
    print(f"Saved to: {OUTPUT}")

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nFirst 5 records:")
    print(df.head())

    return df


if __name__ == "__main__":
    download_decayed_satellites()
