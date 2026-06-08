from abc import ABC, abstractmethod

from app.domain.models.search_request import SearchRequest
from app.domain.models.search_response import SearchResponse


class ISearchService(ABC):
    """
    PRIMARY PORT (driving port) — left side of the hexagon.

    Defines the contract through which the outside world (HTTP adapter,
    CLI, message broker…) drives the application core.

    Any primary adapter (FastAPI endpoint, CLI command, test stub…) must
    depend on this interface, never on a concrete service class.
    This is what makes the architecture symmetric: both sides of the hexagon
    communicate with the core exclusively through interfaces.
    """

    @abstractmethod
    def search(self, request: SearchRequest) -> SearchResponse:
        """
        Execute a RAG query and return the generated answer with sources.

        Args:
            request: A validated SearchRequest carrying the user question
                     and the number of chunks to retrieve.
        Returns:
            A SearchResponse containing the answer, source citations,
            and end-to-end latency.
        """
