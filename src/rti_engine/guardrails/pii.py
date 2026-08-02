"""Detect and redact personal data, in both directions.

Two places it matters. An employee's request may name a colleague — "how
much does Maria earn" — and that name has no reason to reach a model's
context or a stored transcript. And a generated letter must carry no
identifying detail about anyone, including the requester: the recipient
already knows who they are, so a name in the body is a liability without
a benefit.

Presidio combines pattern rules with spaCy's entity recognition, because
the two catch different things. An email address is a pattern; a person's
name is not.

No model is involved, so this cannot be argued out of a finding. That is
the point of putting it here rather than in a prompt.
"""

from functools import lru_cache
from typing import Any, cast

from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_analyzer.predefined_recognizers import PhoneRecognizer
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig
from pydantic import BaseModel, ConfigDict

LANGUAGE = "en"

SPACY_MODEL = "en_core_web_lg"
"""Named explicitly rather than left to Presidio's default.

Presidio downloads its default model the first time an analyzer is built,
which on a fresh deployment means a several-hundred-megabyte download
during someone's first request. Naming it here makes it a deployment
step, and makes the choice visible.
"""

EMPLOYEE_ID = "EMPLOYEE_ID"

DETECTED_ENTITIES: list[str] = [
    "PERSON",
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "IBAN_CODE",
    "CREDIT_CARD",
    "US_SSN",
    EMPLOYEE_ID,
]
"""What is looked for.

Deliberately narrow. Presidio recognises far more, and every extra entity
is another chance to redact something that was not personal.

LOCATION is excluded for that reason: it classified "Engineering" and
"L3" as places, which are the job family and level every request turns
on. A request with those removed cannot be answered at all — a guardrail
that destroys the thing it protects is worse than none.
"""

PHONE_REGIONS: list[str] = ["DE", "FR", "ES", "GB", "US"]
"""Where this employer's people are, plus two common others.

Presidio's phone recogniser defaults to US numbering, so a German mobile
in a request would pass through unredacted — which is most of the numbers
this system will ever see.
"""

MINIMUM_SCORE = 0.5
"""Below this, a detection is more likely noise than a name."""

REDACTION_TEMPLATE = "[{entity}]"


class Finding(BaseModel):
    """One piece of personal data found in a text."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    entity_type: str
    text: str
    score: float
    start: int
    end: int


class ScanResult(BaseModel):
    """What a scan found, and the text with it removed."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    findings: list[Finding]
    redacted: str

    @property
    def clean(self) -> bool:
        return not self.findings

    @property
    def entity_types(self) -> list[str]:
        """The kinds found, for the audit trail. Never the values."""
        return sorted({finding.entity_type for finding in self.findings})

    def summary(self) -> dict[str, Any]:
        """Describe a scan without repeating what it found.

        Recording the values would put the personal data into the audit
        trail, which is the opposite of the point.
        """
        return {
            "pii_found": not self.clean,
            "entity_types": self.entity_types,
            "count": len(self.findings),
        }


def _employee_id_recogniser() -> PatternRecognizer:
    """Recognise this employer's own identifier format.

    Presidio does not know it, and an employee id is as identifying as a
    name in a system where every record is keyed by one.
    """
    return PatternRecognizer(
        supported_entity=EMPLOYEE_ID,
        patterns=[Pattern(name="employee_id", regex=r"\bEMP-\d{5}\b", score=0.9)],
    )


@lru_cache
def get_analyzer() -> AnalyzerEngine:
    """Return the analyzer, built once per process.

    Construction loads a spaCy language model, which is slow enough that
    doing it per request would dominate the cost of a scan.
    """
    provider = NlpEngineProvider(
        nlp_configuration={
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": LANGUAGE, "model_name": SPACY_MODEL}],
        }
    )
    engine = AnalyzerEngine(nlp_engine=provider.create_engine())
    engine.registry.add_recognizer(_employee_id_recogniser())
    engine.registry.add_recognizer(PhoneRecognizer(supported_regions=PHONE_REGIONS))
    return engine


@lru_cache
def get_anonymizer() -> AnonymizerEngine:
    """Return the anonymizer, built once per process."""
    return AnonymizerEngine()  # type: ignore[no-untyped-call]


def scan(text: str, entities: list[str] | None = None) -> ScanResult:
    """Find personal data in a text and return it with that data removed.

    Redaction names the kind rather than blanking it: "[PERSON]" tells a
    reader something was removed and what sort of thing it was, where an
    empty space reads as a typo.
    """
    if not text.strip():
        return ScanResult(findings=[], redacted=text)

    results = get_analyzer().analyze(
        text=text,
        entities=entities if entities is not None else DETECTED_ENTITIES,
        language=LANGUAGE,
        score_threshold=MINIMUM_SCORE,
    )

    findings = [
        Finding(
            entity_type=item.entity_type,
            text=text[item.start : item.end],
            score=round(item.score, 3),
            start=item.start,
            end=item.end,
        )
        for item in results
    ]

    if not findings:
        return ScanResult(findings=[], redacted=text)

    anonymized = get_anonymizer().anonymize(
        text=text,
        # The two packages define structurally identical RecognizerResult
        # classes in separate modules, so mypy sees them as unrelated.
        analyzer_results=cast(list[Any], results),
        operators={
            "DEFAULT": OperatorConfig(
                "replace", {"new_value": REDACTION_TEMPLATE.format(entity="REDACTED")}
            ),
            **{
                entity: OperatorConfig(
                    "replace", {"new_value": REDACTION_TEMPLATE.format(entity=entity)}
                )
                for entity in DETECTED_ENTITIES
            },
        },
    )

    return ScanResult(findings=findings, redacted=anonymized.text)


def redact(text: str) -> str:
    """Return a text with personal data replaced by its entity type."""
    return scan(text).redacted
