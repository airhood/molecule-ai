import argparse
import tarfile
from pathlib import Path

import requests
from tqdm import tqdm

URLS = {
    "summary.csv": (
        "https://libdrive.ethz.ch/index.php/s/X5vOBNSITAG5vzM/download"
        "?path=/&files=summary.csv"
    ),
    "structures.tar.gz": (
        "https://libdrive.ethz.ch/index.php/s/X5vOBNSITAG5vzM/download"
        "?path=/&files=structures.tar.gz"
    ),
}


def download_file(url: str, dest: Path) -> None:
    if dest.exists():
        print(f"[skip] {dest.name} already exists")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    try:
        with requests.get(url, stream=True, timeout=30) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            with open(tmp, "wb") as f, tqdm(
                total=total, unit="B", unit_scale=True, desc=dest.name, leave=True
            ) as bar:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    f.write(chunk)
                    bar.update(len(chunk))
        tmp.rename(dest)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Download QMugs dataset")
    parser.add_argument("--data-dir", default="./data/raw", dest="data_dir")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    download_file(URLS["summary.csv"], data_dir / "summary.csv")

    structures_dir = data_dir / "structures"
    tar_path = data_dir / "structures.tar.gz"

    if structures_dir.exists():
        print("[skip] structures/ already extracted")
    else:
        download_file(URLS["structures.tar.gz"], tar_path)
        with tarfile.open(tar_path, "r:gz") as tar:
            with tqdm(unit=" files", desc="Extracting") as bar:
                for member in tar:
                    tar.extract(member, data_dir, filter="data")
                    bar.update(1)
        print("Extraction complete.")
        tar_path.unlink()
        print(f"Deleted {tar_path.name}")


if __name__ == "__main__":
    main()
