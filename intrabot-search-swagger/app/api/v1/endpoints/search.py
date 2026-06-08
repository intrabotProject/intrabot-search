from fastapi import APIRouter, Depends
from fastapi.openapi.models import Example

from app.domain.interfaces.primary.search.search_service import ISearchService
from app.domain.models.search_request import SearchRequest
from app.domain.models.search_response import SearchResponse, SourceChunk
from app.factories.provider_factory import get_search_service

router = APIRouter()

_EXAMPLE_RESPONSE = {
    "answer": (
        "Le télétravail est autorisé jusqu'à 3 jours par semaine pour les collaborateurs "
        "ayant plus de 6 mois d'ancienneté (source : rh_politique_teletravail_2025.pdf)."
    ),
    "sources": [
        {
            "chunk_id": "3f2a1c4e-...",
            "filename": "rh_politique_teletravail_2025.pdf",
            "excerpt": (
                "Le télétravail est autorisé jusqu'à 3 jours par semaine pour les "
                "collaborateurs ayant plus de 6 mois d'ancienneté."
            ),
            "similarity_score": 0.91,
        }
    ],
    "latency_ms": 1240,
}

_EXAMPLE_NO_SOURCE = {
    "answer": "I cannot find the answer to this question in the available documents.",
    "sources": [],
    "latency_ms": 380,
}


@router.post(
    "/search",
    response_model=SearchResponse,
    summary="Interroger le RAG",
    description=(
        "Pose une question en langage naturel. "
        "Le service embarque la question, interroge ChromaDB, "
        "construit un prompt augmenté et retourne la réponse de Cohere "
        "avec les chunks sources utilisés.\n\n"
        "Si aucun document pertinent n'est trouvé, la réponse indique explicitement "
        "l'absence de source plutôt que de fabriquer une réponse."
    ),
    responses={
        200: {
            "description": "Réponse générée avec sources",
            "content": {
                "application/json": {
                    "examples": {
                        "avec_source": {
                            "summary": "Question avec document correspondant",
                            "value": _EXAMPLE_RESPONSE,
                        },
                        "sans_source": {
                            "summary": "Question sans document correspondant",
                            "value": _EXAMPLE_NO_SOURCE,
                        },
                    }
                }
            },
        },
        422: {"description": "Question vide, trop longue ou top_k hors bornes"},
    },
)
def search(
    request: SearchRequest,
    search_service: ISearchService = Depends(get_search_service),
) -> SearchResponse:
    return search_service.search(request)
