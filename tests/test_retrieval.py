"""Live retrieval tests against the vector store.

Skipped unless Pinecone is configured, so the offline suite stays runnable
without credentials. These assert the two properties that pure similarity
search does not give you: that every layer of the legal picture is
represented in a result set, and that one country's law never reaches
another country's requester.
"""

import pytest

from rti_engine.config.settings import get_settings
from rti_engine.knowledge.vectorstore import GLOBAL_JURISDICTION, retrieve, search

settings = get_settings()

pytestmark = pytest.mark.skipif(
    not (settings.pinecone_api_key and settings.pinecone_index),
    reason="Pinecone is not configured",
)

COUNTRIES = ("DE", "FR", "ES")

NATIONAL_QUERY = "Has this country transposed the directive into national law yet?"
"""Phrased in the national notes' own vocabulary, not the directive's.

A query worded like the legislation matches the legislation, which is the
failure this test exists to catch.
"""


@pytest.mark.parametrize("country", COUNTRIES)
def test_national_status_is_reachable_for_each_country(country: str) -> None:
    """The note saying a country has not transposed must be retrievable."""
    results = retrieve(NATIONAL_QUERY, jurisdiction=country)
    national = [r for r in results if r.jurisdiction == country]

    assert national, f"no {country} status chunk retrieved for a national query"
    assert all(r.document_kind == "national_status" for r in national)


@pytest.mark.parametrize("country", COUNTRIES)
def test_no_other_country_law_reaches_a_requester(country: str) -> None:
    """Isolation, asserted against a query that does surface national content."""
    foreign = set(COUNTRIES) - {country}
    results = retrieve(NATIONAL_QUERY, jurisdiction=country)

    assert results, "retrieval returned nothing; the corpus may not be ingested"
    assert not (foreign & {r.jurisdiction for r in results})


def test_every_document_kind_is_represented() -> None:
    """The point of quotas: no layer is crowded out by a larger one."""
    results = retrieve("What are my rights regarding pay information?", jurisdiction="ES")
    kinds = {r.document_kind for r in results}

    assert {"legislation", "national_status", "company_policy"} <= kinds


def test_unbalanced_search_still_favours_the_largest_document() -> None:
    """Records why quotas exist: plain search is dominated by the directive."""
    results = search("What are my rights regarding pay information?", jurisdiction="ES", k=5)
    assert results
    assert all(r.jurisdiction == GLOBAL_JURISDICTION for r in results)


def test_spanish_justification_threshold_is_retrievable() -> None:
    """Spain's 25% rule must be findable, not buried under the 5% trigger."""
    results = retrieve(
        "When must an employer justify a pay difference under national law?",
        jurisdiction="ES",
    )
    spanish = [r for r in results if r.jurisdiction == "ES"]

    assert spanish
    assert any("25" in r.text for r in spanish)


def test_quotas_are_respected() -> None:
    results = retrieve("pay gap reporting", jurisdiction="DE", quotas={"legislation": 2})
    assert len(results) <= 2
    assert all(r.document_kind == "legislation" for r in results)
