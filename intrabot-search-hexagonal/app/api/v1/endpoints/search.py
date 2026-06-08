from fastapi import APIRouter, Depends

from app.domain.interfaces.primary.search.search_service import ISearchService
from app.domain.models.search_request import SearchRequest
from app.domain.models.search_response import SearchResponse
from app.factories.provider_factory import get_search_service

router = APIRouter()


@router.post(
    "/search",
    response_model=SearchResponse,
    summary="Query the RAG pipeline",
    description=(
        "Primary adapter (HTTP driving adapter). "
        "Receives the question, delegates to the ISearchService primary port, "
        "and returns the generated answer with source citations."
    ),
)
def search(
    request: SearchRequest,
    search_service: ISearchService = Depends(get_search_service),
) -> SearchResponse:
    """
    The endpoint depends exclusively on ISearchService (primary port),
    never on RAGSearchService directly — this is the hexagonal contract.
    """
    return search_service.search(request)