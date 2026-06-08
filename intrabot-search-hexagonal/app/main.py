from fastapi import FastAPI

from app.api.v1.endpoints.search import router as search_router

app = FastAPI(
    title="IntraBot Search Service",
    description="RAG-powered semantic search over intranet documents.",
    version="1.0.0",
)

app.include_router(search_router, prefix="/api/v1", tags=["search"])


@app.get("/health", tags=["ops"])
def health_check() -> dict[str, str]:
    return {"status": "ok"}
