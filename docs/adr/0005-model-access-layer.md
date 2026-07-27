# 0005 — Model access by role, over the v1 API

- Status: accepted
- Date: 2026-07-27

## Context

Six agents need language models. Some do cheap, mechanical classification;
some do regulatory reasoning and drafting; one reviews another's output.
Two questions had to be settled before any agent code was written: how
agents obtain a model, and how the Azure connection is expressed.

Left unaddressed, each agent would construct its own client with its own
model name, temperature and timeout. Changing a model would then mean
finding every construction site, and missing one would leave a silent
inconsistency.

Separately, Azure OpenAI historically required a dated `api-version`
string on every call. Microsoft's v1 GA API removes that requirement: the
standard OpenAI client is used with the endpoint as a base URL, and no
version parameter is passed.

## Decision

**Models are requested by role, not by name.** A caller asks for
`REASONING`, `CLASSIFICATION` or `REVIEW`. Which deployment that resolves
to, what it falls back to, and with what timeout and temperature, is
decided in one module. No agent names a vendor or a deployment.

**Fallbacks cross vendors.** A chain from one Azure deployment to another
Azure deployment does not survive an Azure outage, so the backup for an
Azure model is Anthropic or Groq, reached over separate infrastructure.
The review role inverts this deliberately: it runs on a different vendor
from the drafter, so the reviewer does not share the drafter's blind
spots.

**The Azure connection uses the v1 API** with the standard chat client and
a base URL, rather than the Azure-specific client and a dated
`api-version`. The version path is derived from the configured endpoint in
one property, so the API surface stays an implementation detail rather
than something a person must type into an environment file.

**Temperature is zero where the provider accepts it.** This system
produces statutory documents; identical input should yield identical
output. Some newer models reject the parameter as deprecated, in which
case the provider default applies and the divergence is noted in code.

**Credentials are wrapped in `SecretStr`** so a key cannot be printed by a
traceback, a log line, or an accidental repr.

## Consequences

- Changing a model is one line in one module.
- A provider outage degrades to another vendor rather than failing.
- Costs stay proportionate: classification runs on the cheap tier by
  construction, not by remembering to choose it.
- No dated API version to update as Azure releases new ones.
- Determinism is not fully uniform across providers, since one of them no
  longer accepts a temperature setting. That is a property of the
  provider, and is recorded rather than papered over.
- The role vocabulary becomes an interface. Adding a genuinely new kind of
  work means adding a role, not passing a model name through.

## Alternatives considered

**Let each agent construct its own model.** Rejected: guarantees drift and
scatters cost decisions across the codebase.

**`AzureChatOpenAI` with a dated `api-version`.** Rejected: it is built
around the parameter the v1 API exists to eliminate, and the version string
would need maintaining indefinitely.

**Fallback to a second Azure deployment.** Rejected: shares a failure
domain with the primary, so the fallback is decorative.

**Route Claude through Microsoft Foundry** for a single gateway. Not
available — the model had no quota in any region tried. Using the vendor
API directly turned out to be the stronger choice anyway, for the
failure-domain reason above. The trade-off is that this one provider sits
outside the Azure gateway, which is noted where the gateway is described.
