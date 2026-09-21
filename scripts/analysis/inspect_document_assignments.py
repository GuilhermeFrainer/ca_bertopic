"""Read-only inspection of complete topic membership and raw metadata."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.document_assignments import inspect_run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument(
        "--source", type=Path, help="Relocated unchanged source parquet"
    )
    parser.add_argument("--indices", type=int, nargs="+", default=[3367, 5129])
    args = parser.parse_args()
    print(
        json.dumps(
            inspect_run(args.manifest, args.source, args.indices), indent=2, default=str
        )
    )


if __name__ == "__main__":
    main()
