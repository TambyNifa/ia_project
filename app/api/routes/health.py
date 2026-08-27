"""Route de vérification de disponibilité."""

from fastapi import APIRouter

router = APIRouter(tags=["system"])


@router.get("/health")
def health_check() -> dict[str, str]:
    """Retourne l'état de disponibilité de l'API."""
    return {"status": "ok", "service": "tena-ai"}
