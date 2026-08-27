"""Point d'entrée de l'API FastAPI."""

from fastapi import FastAPI

from app.api.routes.health import router as health_router
from app.core.config import settings

app = FastAPI(
    title=settings.app_name,
    description=(
        "API d'aide à l'analyse de la fiabilité d'informations. "
        "Les résultats sont des estimations, pas des vérifications factuelles."
    ),
    version="0.1.0",
)

app.include_router(health_router)
