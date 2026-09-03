"""Importe X-FACT dans data/external et data/processed sans modifier les sources."""

import argparse
import json
from pathlib import Path

from app.core.logging import configure_logging
from app.ml.data.x_fact import import_sources, split_sources


def main() -> None:
    parser = argparse.ArgumentParser(description="Importer X-FACT V1.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/x_fact_v1"),
        help="Répertoire de sortie des splits harmonisés.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("data/processed/x_fact_v1/quality_report.json"),
        help="Chemin du rapport JSON.",
    )
    parser.add_argument(
        "--external",
        type=Path,
        default=Path("data/external/x_fact"),
        help="Répertoire de conservation des CSV sources.",
    )
    args = parser.parse_args()
    configure_logging()
    report = import_sources(
        split_sources(), args.output, external_directory=args.external
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
