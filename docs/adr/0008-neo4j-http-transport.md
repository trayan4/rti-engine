# 0008 — Neo4j reached over HTTP, not the Bolt driver

- Status: accepted
- Date: 2026-08-07

## Context

Neo4j was originally reached with the official `neo4j` Python driver
over Bolt, its native binary protocol, and that worked without issue
locally and in CI. It did not work once deployed.

Every request touching the knowledge graph in Azure failed with a raw
TCP timeout. The container itself was confirmed healthy — Bolt listening
on `0.0.0.0:7687`, `HealthState: Healthy`, logs showing a normal startup.
DNS resolved the internal hostname correctly. The Bicep ingress
configuration matched the Azure Portal's own Ingress screen exactly,
field for field. An NSG inbound check came back clean. Removing a
redundant `exposedPort` setting made no difference. A raw socket connect
attempted from another container app inside the same environment still
timed out after 134 seconds, even with the client timeout deliberately
raised to 240 seconds — long enough to rule out a slow handshake and
confirm the connection was never going to complete.

Two independent investigations converged on the same explanation. A
second diagnostic pass and a separate web search both identified this as
a known class of Azure Container Apps platform bug specific to
TCP-transport internal ingress: the environment's ingress dataplane does
not reliably forward raw TCP traffic to an otherwise healthy backend,
while HTTP-transport ingress to other apps in the same environment
worked without issue throughout. This is a limitation of the platform
in this configuration, not a defect in this project's Bicep or
application code.

One further path was tried and ruled out directly: switching the
ingress traffic setting to "Accepting traffic from anywhere" to test
whether external TCP ingress behaved differently. The Azure Portal
silently forced the ingress type from TCP to HTTP the moment "anywhere"
was selected, refusing to offer external TCP at all — confirming the
Container Apps environment is not VNet-injected, and that external TCP
ingress genuinely is not available here regardless of any Bicep change.

## Decision

**Reach Neo4j over its HTTP Cypher transaction endpoint
(`/db/neo4j/tx/commit`) instead of the Bolt driver**, using the same
HTTP-transport ingress pattern already proven working for the API and
both MCP servers.

The Bolt driver (`neo4j` package, `GraphDatabase.driver`) was replaced
with a small `httpx`-based client (`knowledge/graph.py`) that opens one
`httpx.Client` per process and posts each Cypher statement to the
transaction endpoint directly. `graph_queries.py`, which selects from a
fixed catalog of named query templates, needed no changes at all: its
`[dict(record) for record in session.run(...)]` pattern already expected
plain dictionaries back, which is what the new client's session object
returns.

Getting this working end to end in the deployed environment required
two further fixes beyond the transport change itself, both worth
recording since neither was obvious in advance:

**The ingress URL scheme had to match the platform's actual routing,
not the container's actual port.** The first attempt set `NEO4J_URI` to
`http://<fqdn>:7474` — the plaintext scheme, with Neo4j's real listening
port appended explicitly. This produced a `ConnectTimeout` identical in
shape to the original Bolt failure. Comparing a working sibling app's
connection (instant, on port 443) against Neo4j's (timeout, on 7474)
against the identical internal load-balancer IP revealed the cause:
Azure Container Apps' internal HTTP ingress multiplexes every internal
app onto the environment's shared load balancer on port 443, routing by
hostname (SNI) to each app's real `targetPort` behind the scenes — it
does not expose each app's port directly. The fix was matching the
already-working `ANALYTICS_MCP_URL`/`KNOWLEDGE_MCP_URL` pattern exactly:
`https://`, no explicit port.

**The CD pipeline's graph-seeding step needed external, not internal,
ingress.** Once connectivity worked from inside the environment, the
GitHub Actions step that seeds the graph (`make graph`, run on a
GitHub-hosted runner) still failed — traced first to a stale test
password (the deployed `NEO4J_PASSWORD` differs from the local one, and
had to be read from Key Vault directly to confirm), and after that was
fixed, to the real remaining cause: Neo4j's ingress was `external:
false`. A GitHub-hosted runner has no network path to an internal-only
Azure address at all — `az containerapp exec`, used for every manual
diagnostic test throughout this investigation, runs from inside the
environment and was never subject to this limitation. Neo4j's ingress
was changed to `external: true`, matching the API and UI. Application-
layer password authentication still gates real access regardless of
network-level visibility.

## Consequences

- Neo4j is reachable from both the deployed application and the CD
  pipeline's seeding step, on infrastructure already proven reliable for
  every other internal service in this environment.
- The TCP-ingress platform bug is avoided entirely rather than worked
  around — HTTP-transport ingress is a different code path in Azure's
  own implementation, not a retry or a longer timeout against the same
  broken one.
- Neo4j's ingress is now externally reachable, which it was not before.
  This is a real, deliberate widening of network exposure, mitigated by
  the fact that access still requires the real database credential —
  the same trade already accepted for the API and UI.
- `archive_bundle`-style silent failures were a real risk here too: an
  early version of the archival fix caught only a narrow exception type
  with no logging at all, and a Bolt-to-HTTP-style connectivity failure
  could plausibly have gone unnoticed the same way if this migration had
  reused that pattern. It did not — every failure in this chain
  surfaced as a real error, on a real timeout, propagated to the caller.
- The `neo4j` package was removed from `pyproject.toml` entirely; it is
  no longer a dependency of this project.

## Alternatives considered

**Fix the TCP ingress configuration instead of migrating transport.**
Rejected: the underlying cause was confirmed to be a platform-level
Azure Container Apps limitation, not a Bicep misconfiguration. No
combination of ingress settings restored working TCP connectivity;
HTTP-transport ingress was the only path proven reliable in this
environment.

**Request external TCP ingress via a VNet-injected environment.**
Rejected as unnecessary scope: it would require reprovisioning the
Container Apps environment with a VNet, a substantially larger change
for a portfolio deployment, when the application-layer HTTP endpoint
Neo4j already exposes achieves the same result with infrastructure
already proven working.

**File an Azure support ticket and wait for a platform fix.** Rejected
for this project's timeline: a genuine option for a production system
with a support contract, but not a reasonable path for unblocking a
portfolio deployment on a practical timescale.
