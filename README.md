# intrabot-search

Microservice de recherche sémantique RAG pour le projet **IntraBot**.  
Reçoit une question en langage naturel, interroge ChromaDB, construit un prompt augmenté et retourne la réponse générée par Cohere avec citation des sources.

---

## Architecture

```
[Front]
  └── POST /api/v1/search
        └── Gateway (port 8000)
              └── intrabot-search (port 8002)
                    ├── POST /embed → intrabot-ingestion (port 8001)
                    ├── ChromaDB PersistentClient (fichier local partagé)
                    └── Cohere command-r-plus-08-2024
```

---

## Prérequis

- **Python 3.11** (testé sur 3.11.15)
- **pip** mis à jour
- Une clé API **Cohere** — [https://dashboard.cohere.com](https://dashboard.cohere.com)
- **intrabot-ingestion** démarré sur le port 8001 (nécessaire pour l'embedding des questions)
- Les documents indexés dans ChromaDB via `POST /ingest` sur l'ingestion service

### ⚠️ Apple Silicon (M1/M2/M3/M4) — configuration obligatoire

Sur Mac Apple Silicon, PyTorch tente d'utiliser le GPU **MPS** (Metal Performance Shaders) pour accélérer les calculs. Le modèle de layout de Docling (RT-DETRv2) utilise des tenseurs en `float64`, que MPS ne supporte pas. Cela provoque un crash silencieux pendant l'ingestion — Docling retourne un texte vide, aucun chunk n'est indexé.

**Solution : forcer l'exécution sur CPU avant tout démarrage.**

```bash
echo 'export PYTORCH_DEVICE=cpu' >> ~/.zshrc
echo 'export DOCLING_DEVICE=cpu' >> ~/.zshrc
source ~/.zshrc
```

Ces variables doivent être définies dans le shell **avant** de démarrer l'ingestion service. Les ajouter à `~/.zshrc` garantit qu'elles sont toujours présentes dans chaque nouveau terminal.

---

## Installation

### 1. Se positionner dans le projet

```bash
cd intrabot-search-final-clean
```

### 2. Créer et activer un environnement virtuel

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 4. Configurer les variables d'environnement

```bash
cp .env.example .env
nano .env
```

Renseigner les valeurs suivantes :

```env
COHERE_API_KEY=your_actual_cohere_api_key_here
INGESTION_SERVICE_URL=http://localhost:8001
CHROMA_PATH=/chemin/absolu/vers/intrabot-ingestion/data/chroma
APP_HOST=0.0.0.0
APP_PORT=8002
LOG_LEVEL=INFO
```

> **Important**
> - `CHROMA_PATH` doit pointer vers le **dossier** contenant `chroma.sqlite3`, pas vers le fichier lui-même.
> - Utiliser un chemin **absolu** — les chemins relatifs ne sont pas résolus correctement selon le répertoire de lancement.
> - Ne pas utiliser de syntaxe Python (`chroma_path: str = "..."`) — le format `.env` est strictement `KEY=value`.

---

## Démarrage

Les services doivent être démarrés dans cet ordre.

### Terminal 1 — intrabot-ingestion

```bash
cd /chemin/vers/intrabot-ingestion
source .venv/bin/activate
uvicorn app.infrastructure.api:app --host 0.0.0.0 --port 8001 --reload
```

### Terminal 2 — Indexation des documents

À lancer une seule fois, ou après ajout de nouveaux documents dans `data/docs/` :

```bash
curl -X POST http://localhost:8001/ingest
```

Vérifier que l'indexation a fonctionné :

```bash
sqlite3 /chemin/vers/intrabot-ingestion/data/chroma/chroma.sqlite3 \
  "SELECT json_extract(string_value, '$') as source, COUNT(*) as chunks
   FROM embedding_metadata WHERE key = 'source' GROUP BY source;"
```

Le résultat doit lister les fichiers indexés avec leur nombre de chunks. Si `COUNT(*) = 0`, voir la section erreurs.

### Terminal 3 — intrabot-search

```bash
cd /chemin/vers/intrabot-search-final-clean
source /chemin/vers/intrabot-search-final/.venv/bin/activate
uvicorn app.main:app --host 0.0.0.0 --port 8002 --reload
```

---

## Utilisation

### Swagger UI

```
http://localhost:8002/docs
```

### Exemple de requête

```bash
curl -X POST http://localhost:8002/api/v1/search \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Quelle est la politique de télétravail ?",
    "top_k": 5
  }'
```

### Exemple de réponse

```json
{
  "answer": "Le télétravail est autorisé jusqu'à 3 jours par semaine...",
  "sources": [
    {
      "chunk_id": "3f2a1c4e-...",
      "filename": "rh_politique_teletravail_2025.pdf",
      "excerpt": "Le télétravail est autorisé...",
      "similarity_score": 0.91
    }
  ],
  "latency_ms": 1240
}
```

### Health check

```bash
curl http://localhost:8002/health
```

---

## Tests

```bash
# Depuis la racine du projet (là où se trouve pytest.ini)
cd intrabot-search-final-clean
pytest

# Tests unitaires uniquement (sans dépendances externes)
pytest tests/unit/

# Tests d'intégration
pytest tests/integration/

# Avec couverture
pytest --cov=app --cov-report=term-missing
```

> **Important** : toujours lancer `pytest` depuis `intrabot-search-final-clean/`, jamais depuis le dossier parent. Lancer depuis le parent provoque une `ImportPathMismatchError` si plusieurs projets sont présents.

---

## Erreurs fréquentes

### `ModuleNotFoundError: No module named 'app'`

**Cause** : uvicorn lancé depuis le mauvais répertoire.  
**Fix** : se placer à la racine du projet avant de lancer uvicorn.

```bash
cd intrabot-search-final-clean
uvicorn app.main:app --host 0.0.0.0 --port 8002 --reload
```

---

### `Field required: cohere_api_key`

**Cause** : le fichier `.env` est absent ou `COHERE_API_KEY` n'est pas renseignée.  
**Fix** :

```bash
cp .env.example .env
nano .env  # renseigner COHERE_API_KEY
```

---

### `Extra inputs are not permitted: chroma_host / chroma_port`

**Cause** : l'ancien `.env` contient encore `CHROMA_HOST` et `CHROMA_PORT` qui n'existent plus dans `config.py`.  
**Fix** : supprimer ces deux lignes du `.env`. Seul `CHROMA_PATH` est utilisé.

---

### `python-dotenv could not parse statement starting at line N`

**Cause** : syntaxe invalide dans le `.env` (syntaxe Python, commentaire mal formé, espace autour du `=`).  
**Fix** : vérifier que chaque ligne respecte le format `KEY=value`.

```bash
cat -n .env  # identifier la ligne problématique
```

---

### `nodename nor servname provided, or not known`

**Cause** : `INGESTION_SERVICE_URL` contient un hostname Docker (`intrabot-ingestion`) non résolu en local.  
**Fix** :

```env
INGESTION_SERVICE_URL=http://localhost:8001
```

---

### `model 'command-r-plus' was removed`

**Cause** : le modèle Cohere `command-r-plus` a été retiré le 15 septembre 2025.  
**Fix** : mettre à jour `cohere_llm_provider.py` :

```python
COHERE_GENERATION_MODEL: str = "command-r-plus-08-2024"
```

---

### `chunks_indexed: 0` après `POST /ingest` — Apple Silicon

**Cause** : Docling utilise PyTorch pour son modèle de layout (RT-DETRv2). Sur Apple Silicon, PyTorch tente d'utiliser le GPU MPS qui ne supporte pas `float64`. Le modèle crashe silencieusement et retourne un texte vide — aucun chunk n'est produit.

Le symptôme caractéristique dans les logs :
```
TypeError: Cannot convert a MPS Tensor to float64 dtype as the MPS framework
doesn't support float64. Please use float32 instead.
Stage layout failed for run 1: ...
Text length: 0
Chunks produced: 0
```

**Fix** : forcer l'exécution sur CPU. Ces variables doivent être définies **avant** de démarrer le service, pas dans `main.py` (PyTorch les lit à l'import, avant l'exécution du code applicatif).

```bash
# Dans le terminal courant (temporaire)
export PYTORCH_DEVICE=cpu
export DOCLING_DEVICE=cpu

# Permanent — à ajouter dans ~/.zshrc
echo 'export PYTORCH_DEVICE=cpu' >> ~/.zshrc
echo 'export DOCLING_DEVICE=cpu' >> ~/.zshrc
source ~/.zshrc
```

> `PYTORCH_ENABLE_MPS_FALLBACK=1` ne suffit pas — il active le fallback CPU uniquement pour les opérations non supportées, mais le modèle reste partiellement sur MPS et continue à échouer. Le forçage complet sur CPU est requis.

---

### `total number of texts must be at most 96`

**Cause** : l'API Cohere limite les appels d'embedding à 96 textes par batch. Un document volumineux peut produire plus de 96 chunks.  
**Fix** : mettre à jour `embed_documents` dans `app/adapters/embedder/cohere_embedder.py` de l'ingestion service :

```python
def embed_documents(self, texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    COHERE_BATCH_LIMIT = 96
    all_embeddings = []
    for i in range(0, len(texts), COHERE_BATCH_LIMIT):
        batch = texts[i:i + COHERE_BATCH_LIMIT]
        resp = self.client.embed(
            texts=batch,
            model=self.MODEL,
            input_type="search_document",
            embedding_types=["float"],
        )
        all_embeddings.extend(resp.embeddings.float)
    return all_embeddings
```

---

### `Error creating hnsw segment reader: Nothing found on disk`

**Cause** : `@lru_cache` sur `PersistentClient` crée un client partagé non thread-safe entre les requêtes. La première requête réussit, les suivantes échouent.  
**Fix** : supprimer `@lru_cache` de `_get_chroma_client()` dans `provider_factory.py`.

```python
def _get_chroma_client() -> chromadb.ClientAPI:
    settings = get_settings()
    return chromadb.PersistentClient(path=settings.chroma_path)
```

---

### `SELECT COUNT(*) FROM embeddings` retourne `0` après suppression de la base

**Cause** : supprimer uniquement `chroma.sqlite3` laisse les dossiers HNSW orphelins, ce qui corrompt l'état de ChromaDB lors de la prochaine ingestion.  
**Fix** : supprimer le **dossier entier** avant de relancer l'ingestion.

```bash
rm -rf /chemin/vers/intrabot-ingestion/data/chroma
mkdir -p /chemin/vers/intrabot-ingestion/data/chroma
# Puis relancer POST /ingest
```

---

### Réponse systématique `"I cannot find the answer"`

**Causes possibles et diagnostic** :

```bash
# 1. Vérifier que ChromaDB contient des données
sqlite3 /chemin/vers/data/chroma/chroma.sqlite3 \
  "SELECT json_extract(string_value, '$') as source, COUNT(*) as chunks
   FROM embedding_metadata WHERE key = 'source' GROUP BY source;"

# 2. Vérifier que l'ingestion service répond
curl http://localhost:8001/health

# 3. Tester avec une question liée au contenu réel des documents indexés
curl -X POST http://localhost:8002/api/v1/search \
  -H "Content-Type: application/json" \
  -d '{"question": "mot-clé présent dans vos documents", "top_k": 5}'
```

| Résultat du diagnostic | Action |
|---|---|
| `COUNT(*) = 0` | Relancer `POST /ingest` |
| `health` ne répond pas | Démarrer intrabot-ingestion |
| Question liée au contenu fonctionne | La question initiale n'a pas de document correspondant |

---

### `ImportPathMismatchError` lors de l'exécution de pytest

**Cause** : pytest lancé depuis le dossier parent contenant plusieurs projets avec le même nom de module `tests`.  
**Fix** : toujours lancer pytest depuis la racine du projet.

```bash
cd intrabot-search-final-clean
pytest
```

---

## Structure du projet

```
intrabot-search-hexagonal/
├── app/
│   ├── main.py                              ← FastAPI + Swagger
│   ├── core/config.py                       ← Settings (.env)
│   ├── api/v1/endpoints/search.py           ← POST /search
│   ├── domain/
│   │   ├── models/                          ← SearchRequest, SearchResponse, RetrievedChunk
│   │   └── interfaces/
│   │       ├── primary/search/              ← ISearchService (port entrant)
│   │       └── secondary/                   ← IEmbeddingProvider, IVectorStore, ILLMProvider, IPromptBuilder
│   ├── application/
│   │   ├── search/rag_search_service.py     ← Orchestration RAG
│   │   └── prompt/rag_prompt_builder.py     ← Guardrail anti-hallucination
│   ├── infrastructure/
│   │   ├── embedding/                       ← Appel HTTP → intrabot-ingestion /embed
│   │   ├── vector_store/                    ← ChromaDB PersistentClient
│   │   └── llm/                             ← Cohere command-r-plus-08-2024
│   └── factories/provider_factory.py        ← Composition root (câblage des dépendances)
└── tests/
    ├── fixtures/
    │   ├── embeddings.py                    ← Vecteurs synthétiques 1024 dims (seed fixe)
    │   └── documents.py                     ← Chunks intranet alignés sur le modèle Chunk de l'ingestion
    ├── unit/                                ← Tests par classe (mocks, pas de dépendances externes)
    └── integration/                         ← Tests endpoint HTTP (FastAPI TestClient)
```

---

*Université Paris Dauphine — Projet IntraBot — 2025-2026*
