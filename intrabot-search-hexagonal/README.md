# intrabot-search

Microservice de recherche sémantique RAG pour le projet **IntraBot**.  
Il reçoit une question en langage naturel, interroge une base vectorielle ChromaDB, construit un prompt augmenté et retourne la réponse générée par un LLM Mistral avec citation des sources.

---

## Architecture

```
intrabot-search/
├── app/
│   ├── main.py                              ← FastAPI app (entrée du process)
│   ├── core/
│   │   └── config.py                        ← Settings chargés depuis .env
│   ├── api/v1/endpoints/
│   │   └── search.py                        ← POST /search
│   ├── domain/
│   │   ├── models/
│   │   │   ├── retrieved_chunk.py           ← dataclass interne (frozen)
│   │   │   ├── search_request.py            ← Pydantic : input validé
│   │   │   └── search_response.py           ← Pydantic : output structuré
│   │   └── interfaces/
│   │       ├── embedding/                   ← IEmbeddingProvider
│   │       ├── vector_store/                ← IVectorStore
│   │       ├── llm/                         ← ILLMProvider
│   │       └── prompt/                      ← IPromptBuilder
│   ├── application/
│   │   ├── prompt/rag_prompt_builder.py     ← Guardrail anti-hallucination
│   │   └── search/rag_search_service.py     ← Orchestration RAG
│   ├── infrastructure/
│   │   ├── embedding/mistral_embedding_provider.py
│   │   ├── vector_store/chroma_vector_store.py
│   │   └── llm/mistral_llm_provider.py
│   └── factories/provider_factory.py       ← DI / câblage des Strategy
└── tests/
    ├── fixtures/
    │   ├── embeddings.py    ← 5 vecteurs doc + 3 vecteurs requête (1024 dims)
    │   └── documents.py     ← 5 chunks intranet + question sans source
    ├── unit/
    │   ├── domain/          ← SearchRequest, SearchResponse, SourceChunk
    │   ├── application/     ← RAGPromptBuilder, RAGSearchService
    │   └── infrastructure/  ← ChromaVectorStore, Mistral providers
    └── integration/
        └── test_search_endpoint.py
```

### Principes de conception

| Principe | Implémentation |
|---|---|
| **SRP** | Chaque classe a une responsabilité unique : le guardrail est dans `RAGPromptBuilder`, pas dans `MistralLLMProvider` |
| **OCP** | `RAGSearchService` dépend d'abstractions ; ajouter OpenAI = créer `OpenAILLMProvider(ILLMProvider)`, aucune modification du service |
| **LSP** | Tests de cas limites documentés : `IVectorStore` retournant `[]`, prompt vide, scores bas sur topic inconnu |
| **ISP** | Un dossier par interface (`embedding/`, `llm/`, `vector_store/`, `prompt/`) — pas de dossier fourre-tout |
| **DIP** | `RAGSearchService` et l'endpoint FastAPI dépendent uniquement d'interfaces, jamais de classes concrètes |

**Pattern Strategy** : `RAGSearchService` accepte n'importe quelle implémentation de `IEmbeddingProvider`, `IVectorStore`, `ILLMProvider` et `IPromptBuilder`. Changer de fournisseur LLM ne demande qu'une ligne dans `provider_factory.py`.

---

## Pipeline RAG

```
POST /api/v1/search
  { question, top_k? }
        │
        ▼
  embed_text(question)          ← mistral-embed  (1024 dims)
        │
        ▼
  query_similar_chunks(top_k)   ← ChromaDB  score = 1 − cosine_distance
        │
        ▼
  build_rag_prompt(             ← guardrail : si chunks=[] → "DOCUMENTS: (none)"
    question, chunks)
        │
        ▼
  generate_answer(prompt)       ← mistral-large-latest
        │
        ▼
  SearchResponse
  { answer, sources[chunk_id, filename, excerpt, score], latency_ms }
```

---

## Démarrage rapide

### Prérequis

- Python 3.11+
- Docker & Docker Compose
- Clé API Mistral

### Variables d'environnement

```bash
cp .env.example .env
# Renseigner MISTRAL_API_KEY dans .env
```

### Lancement avec Docker Compose

```bash
docker-compose up --build
```

Le service écoute sur `http://localhost:8002`.  
ChromaDB est accessible sur `http://localhost:8003`.

### Lancement local (développement)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# ChromaDB doit tourner séparément (voir docker-compose.yml)
uvicorn app.main:app --reload --port 8002
```

---

## API

### `POST /api/v1/search`

**Corps de la requête**

```json
{
  "question": "Combien de jours de télétravail sont autorisés ?",
  "top_k": 5
}
```

| Champ | Type | Obligatoire | Contraintes |
|---|---|---|---|
| `question` | string | oui | 1–2000 caractères, non blanc |
| `top_k` | int | non | 1–20, défaut 5 |

**Réponse 200**

```json
{
  "answer": "Le télétravail est autorisé jusqu'à 3 jours par semaine.",
  "sources": [
    {
      "chunk_id": "chunk-001",
      "filename": "rh_politique_teletravail_2025.pdf",
      "excerpt": "Le télétravail est autorisé...",
      "similarity_score": 0.91
    }
  ],
  "latency_ms": 320
}
```

**Erreurs**

| Code | Cause |
|---|---|
| 422 | Question vide, blanche, trop longue, ou `top_k` hors bornes |
| 500 | Erreur Mistral API ou ChromaDB indisponible |

### `GET /health`

```json
{ "status": "ok" }
```

---

## Tests

```bash
# Depuis la racine du projet
pytest

# Uniquement les tests unitaires
pytest tests/unit/

# Uniquement les tests d'intégration
pytest tests/integration/

# Avec couverture
pytest --cov=app --cov-report=term-missing
```

### Jeu de données de test

`tests/fixtures/embeddings.py` génère cinq vecteurs document et trois vecteurs requête (1024 dimensions, normalisés L2, seed fixe 42).

| Vecteur | Cible | sim cosinus |
|---|---|---|
| `QUERY_EMBEDDING_TELEWORK` | `CHUNK_EMBEDDING_TELEWORK_POLICY` | ~0.987 |
| `QUERY_EMBEDDING_CI_CD` | `CHUNK_EMBEDDING_CI_CD_PIPELINE` | ~0.987 |
| `QUERY_EMBEDDING_UNKNOWN_TOPIC` | aucune | ~0.0 |

`QUERY_EMBEDDING_UNKNOWN_TOPIC` est utilisé pour tester que le service ne fabrique pas de réponse quand aucun document ne correspond (prévention des hallucinations).

### Cas de tests LSP documentés

| Fichier | Test | Contrat vérifié |
|---|---|---|
| `test_rag_search_service.py` | `test_prompt_builder_receives_empty_list_when_no_chunks_retrieved` | `IVectorStore` peut retourner `[]` sans lever d'exception |
| `test_rag_search_service.py` | `test_llm_is_still_called_even_when_no_chunks_are_retrieved` | `ILLMProvider` ne court-circuite jamais l'appel API |
| `test_rag_prompt_builder.py` | `test_prompt_with_empty_chunks_still_contains_system_instruction` | `IPromptBuilder` maintient le guardrail même sans contexte |
| `test_chroma_vector_store.py` | `test_empty_collection_returns_empty_list` | `IVectorStore` retourne `[]` sur collection vide, jamais d'exception |
| `test_chroma_vector_store.py` | `test_top_k_exceeding_collection_size_returns_all_available` | `top_k > len(collection)` est géré silencieusement |
| `test_mistral_llm_provider.py` | `test_generate_answer_with_no_context_prompt_still_calls_api` | Le provider ne court-circuite pas sur un prompt sans contexte |

---

## Changer de fournisseur LLM

1. Créer `app/infrastructure/llm/openai_llm_provider.py` implémentant `ILLMProvider`
2. Dans `app/factories/provider_factory.py`, remplacer `MistralLLMProvider` par `OpenAILLMProvider`
3. Aucune autre modification

```python
# app/infrastructure/llm/openai_llm_provider.py
from openai import OpenAI
from app.domain.interfaces.llm.llm_provider import ILLMProvider

class OpenAILLMProvider(ILLMProvider):
    def __init__(self, api_key: str) -> None:
        self._client = OpenAI(api_key=api_key)

    def generate_answer(self, augmented_prompt: str) -> str:
        response = self._client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": augmented_prompt}],
        )
        return response.choices[0].message.content or ""
```

---

## Communication avec les autres microservices

Ce service est **consommateur en lecture seule** de ChromaDB.  
Il n'appelle jamais `ingestion-service` directement.  
Il est exposé via la **gateway** (port 8000) sur `POST /api/chat`.

```
[Front]  →  [Gateway :8000]  →  [intrabot-search :8002]  →  [ChromaDB :8000]
                                                          →  [Mistral API]
```

---

*Université Paris Dauphine — 2025-2026 — IntraBot Project*
