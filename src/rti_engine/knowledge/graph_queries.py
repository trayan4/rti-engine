"""The fixed set of graph queries agents may run.

Agents select a template by name and supply parameters. They cannot
compose, extend or supply Cypher, because an agent that can write Cypher
can read the entire graph regardless of what its tier permits — and a
prompt-injected one will. The guarantee lives here rather than in an
instruction, for the same reason authorization does.

Adding a capability means adding a template and a test, which is friction
by design.
"""

from typing import Any

from pydantic import BaseModel, ConfigDict

from rti_engine.knowledge.graph import graph_session


class GraphQueryError(RuntimeError):
    """Raised when a query is requested that cannot be run as asked."""


class QueryTemplate(BaseModel):
    """One permitted query: its Cypher, its parameters, and what it answers."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    description: str
    parameters: tuple[str, ...]
    cypher: str


TEMPLATES: tuple[QueryTemplate, ...] = (
    QueryTemplate(
        name="jurisdiction_status",
        description=(
            "Whether a country has transposed the directive, what is expected, "
            "and from when public-sector employers are subject to direct effect."
        ),
        parameters=("jurisdiction",),
        cypher="""
            MATCH (j:Jurisdiction {code: $jurisdiction})
            RETURN j.code AS jurisdiction,
                   j.name AS country,
                   j.transposed AS transposed,
                   j.status AS status,
                   j.expected AS expected,
                   j.direct_effect_from AS direct_effect_from
        """,
    ),
    QueryTemplate(
        name="national_position_on_article",
        description=(
            "What national law currently provides in respect of one article, "
            "for one country. The query behind any statement about whether an "
            "obligation actually applies to this employer today."
        ),
        parameters=("jurisdiction", "article"),
        cypher="""
            MATCH (j:Jurisdiction {code: $jurisdiction})
            MATCH (a:Article {number: $article})
            OPTIONAL MATCH (p:NationalProvision)-[:IN_JURISDICTION]->(j)
            WHERE (p)-[:CORRESPONDS_TO]->(a)
            RETURN j.code AS jurisdiction,
                   j.transposed AS transposed,
                   j.status AS status,
                   a.number AS article,
                   a.heading AS article_heading,
                   p.instrument AS instrument,
                   p.title AS provision,
                   p.summary AS summary,
                   p.threshold AS threshold
            ORDER BY p.title
        """,
    ),
    QueryTemplate(
        name="article_context",
        description=(
            "One article with the articles it references and those referencing "
            "it, so a provision is read alongside what qualifies it."
        ),
        parameters=("article",),
        cypher="""
            MATCH (a:Article {number: $article})
            OPTIONAL MATCH (a)-[:REFERENCES]->(out:Article)
            WITH a, collect(DISTINCT out.number) AS references
            OPTIONAL MATCH (inbound:Article)-[:REFERENCES]->(a)
            RETURN a.number AS article,
                   a.heading AS heading,
                   a.citation AS citation,
                   references AS references,
                   collect(DISTINCT inbound.number) AS referenced_by
        """,
    ),
    QueryTemplate(
        name="policy_sections_for_article",
        description=(
            "Which sections of the employer's compensation policy implement an "
            "article, so a response can point at the employer's own commitment."
        ),
        parameters=("article",),
        cypher="""
            MATCH (s:PolicySection)-[:IMPLEMENTS]->(a:Article {number: $article})
            RETURN s.number AS section, s.title AS title,
                   a.number AS article, a.heading AS article_heading
            ORDER BY s.number
        """,
    ),
    QueryTemplate(
        name="articles_without_national_basis",
        description=(
            "Articles with no corresponding provision in a country's current "
            "law: the obligations that exist under the directive but have no "
            "national footing there yet."
        ),
        parameters=("jurisdiction",),
        cypher="""
            MATCH (a:Article)
            WHERE NOT EXISTS {
                MATCH (p:NationalProvision)-[:CORRESPONDS_TO]->(a)
                MATCH (p)-[:IN_JURISDICTION]->(:Jurisdiction {code: $jurisdiction})
            }
            AND a.number IN $articles
            RETURN a.number AS article, a.heading AS heading
            ORDER BY a.number
        """,
    ),
    QueryTemplate(
        name="provisions_in_jurisdiction",
        description="Every national provision recorded for one country.",
        parameters=("jurisdiction",),
        cypher="""
            MATCH (p:NationalProvision)-[:IN_JURISDICTION]->(
                :Jurisdiction {code: $jurisdiction}
            )
            OPTIONAL MATCH (p)-[:CORRESPONDS_TO]->(a:Article)
            RETURN p.provision_id AS provision_id,
                   p.instrument AS instrument,
                   p.title AS title,
                   p.summary AS summary,
                   p.threshold AS threshold,
                   collect(DISTINCT a.number) AS corresponds_to
            ORDER BY p.provision_id
        """,
    ),
)

TEMPLATES_BY_NAME: dict[str, QueryTemplate] = {t.name: t for t in TEMPLATES}

SUBSTANTIVE_ARTICLES: tuple[int, ...] = (4, 5, 6, 7, 8, 9, 10)
"""The transparency obligations, used as the default scope for gap queries.

Procedural and final provisions have no meaningful national counterpart,
so including them would report a gap that is not one.
"""


def available_queries() -> list[dict[str, Any]]:
    """Describe every permitted query, for presentation to an agent."""
    return [
        {
            "name": template.name,
            "description": template.description,
            "parameters": list(template.parameters),
        }
        for template in TEMPLATES
    ]


def run_query(name: str, **parameters: Any) -> list[dict[str, Any]]:
    """Run one named template with the parameters it declares.

    Unknown names and unexpected parameters are refused rather than
    ignored, so a caller cannot smuggle anything past the template.
    """
    template = TEMPLATES_BY_NAME.get(name)
    if template is None:
        permitted = ", ".join(sorted(TEMPLATES_BY_NAME))
        raise GraphQueryError(f"unknown query {name!r}; permitted: {permitted}")

    supplied = set(parameters)
    expected = set(template.parameters)

    if missing := expected - supplied:
        raise GraphQueryError(f"{name}: missing parameters: {', '.join(sorted(missing))}")
    if unexpected := supplied - expected:
        raise GraphQueryError(f"{name}: unexpected parameters: {', '.join(sorted(unexpected))}")

    if name == "articles_without_national_basis":
        parameters["articles"] = list(SUBSTANTIVE_ARTICLES)

    with graph_session() as session:
        return [dict(record) for record in session.run(template.cypher, **parameters)]
