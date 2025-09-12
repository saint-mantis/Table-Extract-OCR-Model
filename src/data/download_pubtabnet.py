import os
import requests
import zipfile
from pathlib import Path

PUBTABNET_URL = "https://zenodo.org/record/5096203/files/PubTabNet_2.0.0.zip?download=1"
DATA_DIR = Path("data/pubtabnet")
ZIP_PATH = DATA_DIR / "PubTabNet_2.0.0.zip"


def download_pubtabnet():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not ZIP_PATH.exists():
        print(f"Downloading PubTabNet to {ZIP_PATH}...")
        with requests.get(PUBTABNET_URL, stream=True) as r:
            r.raise_for_status()
            with open(ZIP_PATH, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        print("Download complete.")
    else:
        print("PubTabNet zip already exists.")

    # Extract
    extract_dir = DATA_DIR / "PubTabNet_2.0.0"
    if not extract_dir.exists():
        print(f"Extracting to {extract_dir}...")
        with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall(DATA_DIR)
        print("Extraction complete.")
    else:
        print("PubTabNet already extracted.")

if __name__ == "__main__":
    download_pubtabnet()
    print("Done.")
