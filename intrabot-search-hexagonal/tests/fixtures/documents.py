"""
Document chunk fixtures representing a realistic intranet knowledge base.
Aligned with the intrabot-ingestion Chunk dataclass:

    @dataclass
    class Chunk:
        text: str
        metadata: dict  # {"source": str, "chunk_index": int, "headings": str}

Each ChunkFixture maps 1-to-1 to an embedding in tests/fixtures/embeddings.py
via its position in ALL_CHUNK_FIXTURES / ALL_CHUNK_EMBEDDINGS.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ChunkFixture:
    chunk_id: str       # synthetic ID used only in tests (ChromaStore generates real UUIDs)
    source: str         # maps to metadata["source"] — filename of the origin document
    text: str           # chunk content as stored by DoclingChunker
    chunk_index: int    # maps to metadata["chunk_index"]
    headings: str = ""  # maps to metadata["headings"] — section path from Docling

    @property
    def metadata(self) -> dict:
        """Returns the metadata dict exactly as ChromaStore stores it."""
        return {
            "source": self.source,
            "chunk_index": self.chunk_index,
            "headings": self.headings,
        }


CHUNK_TELEWORK_POLICY = ChunkFixture(
    chunk_id="chunk-001",
    source="rh_politique_teletravail_2025.pdf",
    text=(
        "Le télétravail est autorisé jusqu'à 3 jours par semaine pour les collaborateurs "
        "ayant plus de 6 mois d'ancienneté. Une demande écrite doit être soumise au manager "
        "au moins 5 jours ouvrés à l'avance. L'équipement informatique est fourni par l'entreprise."
    ),
    chunk_index=0,
)

CHUNK_CI_CD_PIPELINE = ChunkFixture(
    chunk_id="chunk-002",
    source="tech_cicd_pipeline_guide.pdf",
    text=(
        "Le pipeline CI/CD s'appuie sur GitHub Actions. Chaque pull request déclenche "
        "automatiquement les tests unitaires, les tests d'intégration et l'analyse SonarQube. "
        "Un déploiement en staging est effectué après validation de la branche develop."
    ),
    chunk_index=0,
)

CHUNK_EXPENSE_REPORT = ChunkFixture(
    chunk_id="chunk-003",
    source="finance_notes_de_frais_procedure.pdf",
    text=(
        "Les notes de frais doivent être soumises dans les 30 jours suivant la dépense. "
        "Tout frais supérieur à 150 € nécessite une facture originale. "
        "La validation est effectuée par le responsable hiérarchique direct."
    ),
    chunk_index=0,
)

CHUNK_ONBOARDING_GUIDE = ChunkFixture(
    chunk_id="chunk-004",
    source="rh_guide_onboarding_2025.pdf",
    text=(
        "Le processus d'onboarding dure 3 semaines. La première semaine est consacrée "
        "aux formations obligatoires (sécurité, RGPD, outils internes). "
        "Un buddy est assigné à chaque nouveau collaborateur pour la durée de la période d'essai."
    ),
    chunk_index=0,
)

CHUNK_SECURITY_POLICY = ChunkFixture(
    chunk_id="chunk-005",
    source="si_politique_securite_informatique.pdf",
    text=(
        "Les mots de passe doivent comporter au minimum 12 caractères avec majuscules, "
        "chiffres et caractères spéciaux. Le renouvellement est obligatoire tous les 90 jours. "
        "Toute tentative de connexion échouée après 5 essais entraîne un verrouillage du compte."
    ),
    chunk_index=0,
)

ALL_CHUNK_FIXTURES: list[ChunkFixture] = [
    CHUNK_TELEWORK_POLICY,
    CHUNK_CI_CD_PIPELINE,
    CHUNK_EXPENSE_REPORT,
    CHUNK_ONBOARDING_GUIDE,
    CHUNK_SECURITY_POLICY,
]

# A question that has no matching document in the knowledge base.
# Used to verify that the system refuses to hallucinate rather than fabricating
# an answer from LLM parametric knowledge.
QUESTION_WITH_NO_MATCHING_SOURCE: str = (
    "Quelle est la procédure pour commander des fournitures de bureau ?"
)