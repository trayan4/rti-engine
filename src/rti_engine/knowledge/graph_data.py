"""Authored graph content: jurisdictions, national provisions, policy mappings.

Article nodes and the cross-references between them are derived
mechanically from the directive text. The records here cannot be: deciding
that Germany's individual right to information corresponds to Article 7,
or that a policy section implements Article 9, is a legal judgment.

Those judgments are written by hand, in one place, where they can be
reviewed and corrected — rather than inferred by a model at runtime, where
a wrong correspondence would be invisible and would propagate into a
statutory document.

Every record here must remain consistent with the corresponding corpus
document. If the graph and the notes disagree, traversal and retrieval
answer the same question differently.
"""

from pydantic import BaseModel, ConfigDict


class StrictRecord(BaseModel):
    """Base: unknown fields rejected, instances immutable."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class JurisdictionRecord(StrictRecord):
    """A member state and its transposition position."""

    code: str
    name: str
    transposed: bool
    status: str
    expected: str | None = None
    direct_effect_from: str | None = None
    """Date from which public-sector employers may rely on the directive."""


class ProvisionRecord(StrictRecord):
    """A provision of national law, and the article it corresponds to."""

    provision_id: str
    jurisdiction: str
    instrument: str
    title: str
    summary: str
    corresponds_to: tuple[int, ...]
    """Articles of the directive this provision addresses, in whole or part."""

    threshold: str | None = None
    """Any numeric trigger, kept distinct from the directive's own."""


class PolicySectionRecord(StrictRecord):
    """A section of the employer's compensation policy."""

    number: int
    title: str
    implements: tuple[int, ...]


JURISDICTIONS: tuple[JurisdictionRecord, ...] = (
    JurisdictionRecord(
        code="DE",
        name="Germany",
        transposed=False,
        status="deadline missed; no draft published as at the transposition deadline",
        expected="cabinet review expected during 2026; right to information "
        "recommended to apply first in 2027",
        direct_effect_from="2026-06-08",
    ),
    JurisdictionRecord(
        code="FR",
        name="France",
        transposed=False,
        status="deadline missed; draft law published",
        expected="entry into force targeted for 1 January 2027",
        direct_effect_from="2026-06-08",
    ),
    JurisdictionRecord(
        code="ES",
        name="Spain",
        transposed=False,
        status="deadline missed; prior consultation closed, no draft text published",
        expected="not announced",
        direct_effect_from="2026-06-08",
    ),
)


NATIONAL_PROVISIONS: tuple[ProvisionRecord, ...] = (
    ProvisionRecord(
        provision_id="de-entgtranspg-information",
        jurisdiction="DE",
        instrument="Entgelttransparenzgesetz (EntgTranspG)",
        title="Individual right to information",
        summary=(
            "Employees in establishments with more than 200 employees may request "
            "the median pay of the opposite sex in a comparable group of at least "
            "six employees. Narrower than the directive in threshold, comparator "
            "and the measure disclosed."
        ),
        corresponds_to=(7,),
        threshold="more than 200 employees; comparator group of at least six",
    ),
    ProvisionRecord(
        provision_id="de-entgtranspg-reporting",
        jurisdiction="DE",
        instrument="Entgelttransparenzgesetz (EntgTranspG)",
        title="Reporting obligation",
        summary=(
            "Employers with more than 500 employees report on equality and equal "
            "pay measures within existing management reporting cycles. There is "
            "no category-level gap disclosure of the kind the directive requires."
        ),
        corresponds_to=(9,),
        threshold="more than 500 employees",
    ),
    ProvisionRecord(
        provision_id="fr-index-egalite",
        jurisdiction="FR",
        instrument="Index de l'égalité professionnelle",
        title="Professional equality index",
        summary=(
            "Employers with 50 or more employees publish an annual composite "
            "score out of 100 across weighted indicators. A composite score is "
            "not a set of category-level gap disclosures, and the regime carries "
            "no individual right to comparator pay information."
        ),
        corresponds_to=(9,),
        threshold="50 or more employees; corrective measures below a defined score",
    ),
    ProvisionRecord(
        provision_id="es-rd902-registro",
        jurisdiction="ES",
        instrument="Royal Decree 902/2020",
        title="Pay register (registro retributivo)",
        summary=(
            "Every employer, regardless of size, maintains a register of average "
            "values and pay ranges by sex for each group, category, level and "
            "post of equal value. Employees access it through their legal "
            "representatives rather than by individual request."
        ),
        corresponds_to=(7, 9),
        threshold="all employers, regardless of size",
    ),
    ProvisionRecord(
        provision_id="es-rd902-auditoria",
        jurisdiction="ES",
        instrument="Royal Decree 902/2020",
        title="Pay audit (auditoría retributiva)",
        summary=(
            "Employers with 50 or more employees conduct a pay audit within their "
            "equality plan, including job evaluation and an action plan where "
            "disparities are found."
        ),
        corresponds_to=(10,),
        threshold="50 or more employees",
    ),
    ProvisionRecord(
        provision_id="es-rd902-justification",
        jurisdiction="ES",
        instrument="Royal Decree 902/2020",
        title="Justification threshold",
        summary=(
            "Where the average pay of one sex exceeds the other by 25 per cent or "
            "more, the employer must justify that the difference is unrelated to "
            "sex. This is a different trigger from the directive's joint pay "
            "assessment threshold and the two must not be conflated."
        ),
        corresponds_to=(10,),
        threshold="25 per cent average pay difference",
    ),
)


POLICY_SECTIONS: tuple[PolicySectionRecord, ...] = (
    PolicySectionRecord(number=2, title="Job architecture", implements=(4,)),
    PolicySectionRecord(number=4, title="How base pay is set", implements=(6,)),
    PolicySectionRecord(number=5, title="Pay progression", implements=(6,)),
    PolicySectionRecord(number=6, title="Working patterns", implements=(4,)),
    PolicySectionRecord(number=7, title="Categories of work of equal value", implements=(4,)),
    PolicySectionRecord(number=8, title="Employee right to pay information", implements=(7, 8, 12)),
    PolicySectionRecord(number=9, title="Pay gap monitoring", implements=(9, 10)),
    PolicySectionRecord(number=10, title="Recruitment", implements=(5,)),
)
