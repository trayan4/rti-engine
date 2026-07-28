"""Prompts as versioned code.

A prompt is a typed object, not a string built where it happens to be
needed. That buys three things. A change is a reviewable diff rather than
an edit buried in agent logic. A missing variable fails when the prompt is
constructed, instead of leaving a blank in text the model then reasons
over confidently. And every prompt can be measured against a token budget
before a call is made.

Rules that must hold for every agent live in shared blocks and are
composed in. Copied into each prompt they would drift, and the one that
drifted would be the one that mattered.
"""

from string import Formatter
from typing import Any, Self

from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic import BaseModel, ConfigDict, Field, model_validator

from rti_engine.llm.tokens import count_tokens

GROUNDING_RULES = """\
## Figures

You do not calculate. Every number you state must have come from a tool
result in this conversation, quoted exactly as the tool returned it.

- Do not compute, estimate, round, convert or combine figures.
- Do not restate a percentage in different terms, or infer one figure
  from another.
- If a number you need is not in a tool result, say so and stop. Do not
  supply an approximation.

A figure that cannot be traced to a tool result must not appear in your
output."""

CITATION_RULES = """\
## Sources

Every statement about what the law or a policy requires must cite the
source it came from, using the citation string the tool returned.

- Cite the specific article, provision or policy section, never the
  corpus generally.
- Do not attribute a statement to a source that does not support it.
- If retrieval returned nothing relevant, say the question cannot be
  answered from the available sources."""

TOOL_FAILURE_RULES = """\
## Tool results

A tool may return an error instead of data. Errors arrive as ordinary
text beginning "Error calling tool", not as an interruption.

- Treat such a result as a refusal, never as content to summarise.
- Do not retry with different arguments to obtain data that was refused,
  and do not construct the answer from another source instead.
- Report that the information could not be obtained, and why."""

JURISDICTION_RULES = """\
## Jurisdiction

An obligation under the Directive is not automatically an obligation
under the law the employer is subject to today. Several member states
have not yet transposed it.

- Establish the national position before stating what is required.
- Do not present a national threshold as the Directive's, or the
  reverse.
- Where the employer's own policy commits to more than national law
  compels, say which is which."""


class PromptError(ValueError):
    """Raised when a prompt is rendered with the wrong inputs."""


class Prompt(BaseModel):
    """One versioned prompt, with declared inputs and a token budget."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    version: int = Field(ge=1)
    description: str
    template: str
    inputs: tuple[str, ...] = ()
    max_tokens: int = Field(default=4000, gt=0)

    @model_validator(mode="after")
    def check_inputs_match_template(self) -> Self:
        """Require declared inputs and template placeholders to agree.

        A placeholder with no declared input renders as a missing key; a
        declared input with no placeholder is dead configuration that
        suggests the template says something it does not.
        """
        found = {field for _, field, _, _ in Formatter().parse(self.template) if field is not None}
        declared = set(self.inputs)

        if undeclared := found - declared:
            raise ValueError(
                f"{self.name}: template uses undeclared inputs: {', '.join(sorted(undeclared))}"
            )
        if unused := declared - found:
            raise ValueError(
                f"{self.name}: declared inputs never used: {', '.join(sorted(unused))}"
            )
        return self

    def render(self, **values: Any) -> str:
        """Fill the template, refusing missing or unexpected values."""
        supplied = set(values)
        expected = set(self.inputs)

        if missing := expected - supplied:
            raise PromptError(f"{self.name}: missing inputs: {', '.join(sorted(missing))}")
        if unexpected := supplied - expected:
            raise PromptError(f"{self.name}: unexpected inputs: {', '.join(sorted(unexpected))}")

        return self.template.format(**values)

    def token_count(self, **values: Any) -> int:
        """Count the tokens this prompt occupies once rendered."""
        return count_tokens(self.render(**values))

    def fits(self, **values: Any) -> bool:
        """Report whether the rendered prompt is within its own budget."""
        return self.token_count(**values) <= self.max_tokens

    def to_chat_prompt(self, **values: Any) -> ChatPromptTemplate:
        """Return this prompt as a LangChain chat prompt.

        The rendered text becomes the system message and conversation
        history is appended, which is the shape a graph node needs.

        A SystemMessage object is passed rather than a ("system", text)
        tuple: the tuple form re-templates the string, so a literal brace
        in the prompt — a JSON example, a set — would be read as a
        variable and fail. The text is already rendered here, so any
        further templating is wrong.
        """
        return ChatPromptTemplate.from_messages(
            [SystemMessage(content=self.render(**values)), MessagesPlaceholder("messages")]
        )

    @property
    def identifier(self) -> str:
        """Stable identifier, recorded in the audit trail with each run."""
        return f"{self.name}@v{self.version}"


class PromptRegistry(BaseModel):
    """Every prompt in the system, addressable by name."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    prompts: tuple[Prompt, ...]

    @model_validator(mode="after")
    def check_names_are_unique(self) -> Self:
        names = [prompt.name for prompt in self.prompts]
        if len(names) != len(set(names)):
            raise ValueError("prompt names must be unique")
        return self

    def get(self, name: str) -> Prompt:
        """Return one prompt by name, refusing an unknown one."""
        for prompt in self.prompts:
            if prompt.name == name:
                return prompt
        available = ", ".join(sorted(p.name for p in self.prompts))
        raise PromptError(f"unknown prompt {name!r}; available: {available}")

    def names(self) -> list[str]:
        return sorted(prompt.name for prompt in self.prompts)


REGISTRY = PromptRegistry(prompts=())
"""Populated as each agent is built. Every agent registers its prompt here
so the full set is inspectable, diffable and testable in one place."""
