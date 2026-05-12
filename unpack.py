import argparse
import zipfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Unpack processed dataset")
    parser.add_argument("--zip", default="processed.zip")
    parser.add_argument("--out-dir", default="./data/processed", dest="out_dir")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Extracting {args.zip} -> {out_dir} ...")
    with zipfile.ZipFile(args.zip, "r") as z:
        z.extractall(out_dir)
    print("Done.")


if __name__ == "__main__":
    main()
