"""Configuration de la journalisation."""

import logging

from app.core.config import settings


def configure_logging() -> None:
    """Configure le format de logs de l'application."""
    logging.basicConfig(
        level=settings.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
