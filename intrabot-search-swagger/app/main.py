from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi

from app.api.v1.endpoints.search import router as search_router

# ── OpenAPI metadata ──────────────────────────────────────────────────────────

_DESCRIPTION = """
## IntraBot Search Service

Microservice de **recherche sémantique RAG** sur les documents de l'intranet.

### Pipeline

1. La question est **vectorisée** via `intrabot-ingestion POST /embed` (Cohere embed-multilingual-v3.0)
2. Les **chunks les plus proches** sont récupérés depuis ChromaDB (similarité cosinus)
3. Un **prompt augmenté** est construit avec les chunks et envoyé à Cohere `command-r-plus`
4. La **réponse générée** est retournée avec les sources citées

### Ports

| Port | Rôle |
|---|---|
| `8002` | Ce service (intrabot-search) |
| `8000` | intrabot-ingestion (embedding) |
| `8003` | ChromaDB |

### Dépendances

- `intrabot-ingestion` doit être démarré et avoir indexé les documents avant toute requête
- La variable `COHERE_API_KEY` doit être définie dans `.env`
"""

_TAGS_METADATA = [
    {
        "name": "search",
        "description": "Recherche RAG — pose une question, reçois une réponse avec sources.",
    },
    {
        "name": "ops",
        "description": "Opérations de supervision (health check).",
    },
]

# ── Application ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="IntraBot Search Service",
    description=_DESCRIPTION,
    version="1.0.0",
    openapi_tags=_TAGS_METADATA,
    contact={
        "name": "IntraBot Project",
    },
    license_info={
        "name": "Academic — Paris Dauphine 2025-2026",
    },
)

app.include_router(search_router, prefix="/api/v1", tags=["search"])


@app.get(
    "/health",
    tags=["ops"],
    summary="Health check",
    response_description="Service opérationnel",
)
def health_check() -> dict[str, str]:
    return {"status": "ok"}


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    from app.core.config import get_settings

    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=True,
    )
