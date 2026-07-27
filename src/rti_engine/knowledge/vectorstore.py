"""Vector store ingestion and filtered retrieval for the knowledge corpus.

Every chunk is stored with the metadata needed to cite it and to filter
it. Filtering is the important half: the German, French and Spanish status
notes all discuss the same directive, so embedding similarity alone cannot
separate them. A Spanish requester answered with German law would receive
a fluent, confident, wrong answer.

Pinecone stores no null metadata and cannot filter on an absent field, so
documents that apply regardless of country carry an explicit sentinel
rather than nothing. A query for one country then matches that sentinel
plus its own code, and nothing else.

Ingestion is idempotent: the namespace is cleared before writing, so
re-running after a corpus change leaves no orphaned vectors behind.
"""

from functools import lru_cache
from typing import Any

from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from pydantic import BaseModel, ConfigDict

from rti_engine.config.settings import get_settings
from rti_engine.knowledge.chunking import Chunk
from rti_engine.knowledge.corpus import chunk_corpus
from rti_engine.llm.factory import get_embeddings

NAMESPACE = "corpus"
"""All corpus vectors live here, so the namespace can be cleared wholesale."""

GLOBAL_JURISDICTION = "ALL"
"""Marks a document that applies regardless of country."""

UPSERT_BATCH_SIZE = 100
DEFAULT_TOP_K = 5

DEFAULT_QUOTAS: dict[str, int] = {
    "national_status": 3,
    "legislation": 4,
    "company_policy": 3,
}
"""Chunks retrieved per document kind.

Undifferentiated top-k does not work on this corpus. The directive is 106
chunks against 19 for the national notes, and it uses the vocabulary a
legal question is phrased in, so it wins every ranking. The national
notes then never surface — and since those notes are the only source that
knows a country has not transposed, the system would answer as though the
directive were in force everywhere.

Quotas guarantee each layer of the legal picture is represented rather
than hoping cosine similarity balances them.
"""


class ConfigurationError(RuntimeError):
    """Raised when the vector store is used without being configured."""


class RetrievedChunk(BaseModel):
    """One search result, carrying everything needed to cite it."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    text: str
    citation: str
    document_id: str
    document_kind: str
    jurisdiction: str
    score: float


def _require(value: str | None, name: str) -> str:
    """Return a required setting, or fail with a message naming it."""
    if not value:
        raise ConfigurationError(f"{name} is not set; check your .env file")
    return value


def chunk_metadata(chunk: Chunk) -> dict[str, Any]:
    """Flatten a chunk into Pinecone-storable metadata.

    The text is stored alongside the vector so retrieval returns usable
    content without a second lookup against another store.
    """
    return {
        "text": chunk.text,
        "citation": chunk.citation,
        "document_id": chunk.document_id,
        "document_kind": chunk.document_kind,
        "jurisdiction": chunk.jurisdiction or GLOBAL_JURISDICTION,
        "section_type": chunk.section_type,
        "section_number": chunk.section_number,
        "heading": chunk.heading or "",
        "token_count": chunk.token_count,
    }


@lru_cache
def get_vector_store() -> PineconeVectorStore:
    """Return the vector store, constructed once per process.

    The API key is passed explicitly rather than left to the environment:
    configuration is loaded from .env into the settings object and is never
    exported into the process environment, so a library reading os.environ
    would find nothing.
    """
    settings = get_settings()

    return PineconeVectorStore(
        index_name=_require(settings.pinecone_index, "PINECONE_INDEX"),
        pinecone_api_key=_require(settings.pinecone_api_key, "PINECONE_API_KEY"),
        embedding=get_embeddings(),
        namespace=NAMESPACE,
        text_key="text",
    )


def _pinecone_client() -> Pinecone:
    """Return a Pinecone client built from settings.

    The key is passed explicitly for the same reason as in the vector
    store: it is loaded from .env into the settings object and never
    exported into the process environment.
    """
    return Pinecone(api_key=_require(get_settings().pinecone_api_key, "PINECONE_API_KEY"))


def clear_namespace() -> None:
    """Delete every vector in the corpus namespace.

    Tolerates an empty or absent namespace: on a fresh index there is
    nothing to delete, and Pinecone reports that as an error rather than a
    no-op.
    """
    index_name = _require(get_settings().pinecone_index, "PINECONE_INDEX")
    index = _pinecone_client().Index(index_name)
    try:
        index.delete(delete_all=True, namespace=NAMESPACE)
    except Exception:  # noqa: BLE001 - absent namespace is not a failure
        return


def ingest_corpus(chunks: list[Chunk] | None = None, replace: bool = True) -> int:
    """Embed the corpus and write it to Pinecone. Returns the count written."""
    to_write = chunks if chunks is not None else chunk_corpus()
    if not to_write:
        raise ValueError("no chunks to ingest; the corpus may be missing")

    if replace:
        clear_namespace()

    store = get_vector_store()
    for start in range(0, len(to_write), UPSERT_BATCH_SIZE):
        batch = to_write[start : start + UPSERT_BATCH_SIZE]
        store.add_texts(
            texts=[chunk.text for chunk in batch],
            metadatas=[chunk_metadata(chunk) for chunk in batch],
            ids=[chunk.chunk_id for chunk in batch],
        )

    return len(to_write)


def build_filter(
    jurisdiction: str | None = None, document_kinds: list[str] | None = None
) -> dict[str, Any] | None:
    """Build a Pinecone metadata filter for a query.

    A jurisdiction filter always admits the global sentinel, so EU law and
    company policy remain visible to a country-scoped query while other
    countries' law does not.
    """
    conditions: dict[str, Any] = {}

    if jurisdiction:
        conditions["jurisdiction"] = {"$in": [GLOBAL_JURISDICTION, jurisdiction]}
    if document_kinds:
        conditions["document_kind"] = {"$in": document_kinds}

    return conditions or None


def search(
    query: str,
    jurisdiction: str | None = None,
    document_kinds: list[str] | None = None,
    k: int = DEFAULT_TOP_K,
) -> list[RetrievedChunk]:
    """Retrieve the most relevant chunks, scoped to a jurisdiction."""
    results = get_vector_store().similarity_search_with_score(
        query, k=k, filter=build_filter(jurisdiction, document_kinds)
    )

    return [
        RetrievedChunk(
            text=str(document.metadata.get("text", document.page_content)),
            citation=str(document.metadata.get("citation", "")),
            document_id=str(document.metadata.get("document_id", "")),
            document_kind=str(document.metadata.get("document_kind", "")),
            jurisdiction=str(document.metadata.get("jurisdiction", "")),
            score=float(score),
        )
        for document, score in results
    ]


def retrieve(
    query: str,
    jurisdiction: str | None = None,
    quotas: dict[str, int] | None = None,
) -> list[RetrievedChunk]:
    """Retrieve a balanced set of chunks across document kinds.

    One query per kind, merged and re-sorted by score. Duplicates are
    dropped, since a chunk can satisfy only one kind's quota.

    This is the entry point agents should use. `search` remains available
    for cases that genuinely want unbalanced ranking.
    """
    per_kind = quotas if quotas is not None else DEFAULT_QUOTAS

    merged: list[RetrievedChunk] = []
    seen: set[str] = set()

    for kind, limit in per_kind.items():
        if limit <= 0:
            continue
        for hit in search(query, jurisdiction=jurisdiction, document_kinds=[kind], k=limit):
            key = f"{hit.citation}|{hit.text[:80]}"
            if key in seen:
                continue
            seen.add(key)
            merged.append(hit)

    return sorted(merged, key=lambda hit: hit.score, reverse=True)


def main() -> None:
    """Ingest the corpus and report what was written."""
    written = ingest_corpus()
    print(f"ingested {written} chunks into namespace '{NAMESPACE}'")


if __name__ == "__main__":
    main()
