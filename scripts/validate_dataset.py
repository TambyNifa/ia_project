"""Valide un fichier CSV TENA AI et affiche son rapport JSON."""

import argparse
import json
from pathlib import Path

from app.core.logging import configure_logging
from app.ml.data.validation import validate_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Valider un dataset TENA AI.")
    parser.add_argument("path", type=Path, help="Chemin vers le fichier CSV.")
    args = parser.parse_args()
    configure_logging()
    report = validate_csv(args.path)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["missing_columns"] or report["invalid_rows"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
