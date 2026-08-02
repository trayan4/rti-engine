"""Tests for the HTTP interface.

The identity tests matter most. Every guarantee below them assumes the
principal is who the session says — a caller able to submit as another
employee would walk straight through the authorization layer, because
that layer trusts the identity it is handed.

Requests run in the background, so these assert what the endpoint returns
and who it lets in, not what the graph eventually produces.
"""

from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient

from rti_engine.api.app import app
from rti_engine.api.security import EMPLOYEE_HEADER, REVIEWER_HEADER
from rti_engine.config.settings import get_settings

needs_postgres = pytest.mark.skipif(
    not get_settings().postgres_dsn, reason="Postgres is not configured"
)

EMPLOYEE = {EMPLOYEE_HEADER: "EMP-00001"}
OTHER_EMPLOYEE = {EMPLOYEE_HEADER: "EMP-00002"}
REVIEWER = {EMPLOYEE_HEADER: "HR-0007", REVIEWER_HEADER: "true"}


@pytest.fixture
def client() -> Iterator[TestClient]:
    """A client that does not run background tasks.

    Submitting would otherwise start a full pipeline — minutes of model
    calls — for a test asserting the shape of a 202.
    """
    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client


@pytest.fixture
def submitted(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Submit a request without letting the graph run."""
    from fastapi import BackgroundTasks

    monkeypatch.setattr(BackgroundTasks, "add_task", lambda *_, **__: None)

    response = client.post(
        "/requests", json={"request_text": "What is my salary?"}, headers=EMPLOYEE
    )
    assert response.status_code == 202
    return dict(response.json())


# --- identity ---


def test_a_request_without_identity_is_refused(client: TestClient) -> None:
    """Every guarantee below this assumes the principal is authenticated."""
    response = client.post("/requests", json={"request_text": "What is my salary?"})
    assert response.status_code == 401


def test_an_employee_id_cannot_be_supplied_in_the_body(client: TestClient) -> None:
    """Accepting one would let a caller submit as someone else."""
    response = client.post(
        "/requests",
        json={"request_text": "What is my salary?", "employee_id": "EMP-00042"},
        headers=EMPLOYEE,
    )
    assert response.status_code == 422


def test_approvals_require_a_reviewer(client: TestClient) -> None:
    """A requester may not approve the disclosure of their own comparison."""
    assert client.get("/approvals", headers=EMPLOYEE).status_code == 403


def test_a_reviewer_may_list_approvals(client: TestClient) -> None:
    assert client.get("/approvals", headers=REVIEWER).status_code == 200


# --- submission ---


def test_an_empty_request_is_refused(client: TestClient) -> None:
    response = client.post("/requests", json={"request_text": ""}, headers=EMPLOYEE)
    assert response.status_code == 422


@needs_postgres
def test_submitting_returns_an_id_immediately(submitted: dict[str, Any]) -> None:
    """202, not 200: accepted rather than answered."""
    assert submitted["status"] == "accepted"
    assert submitted["request_id"]


@needs_postgres
def test_a_submitted_request_is_visible_at_once(
    client: TestClient, submitted: dict[str, Any]
) -> None:
    """Recorded before the graph runs, so a failure mid-pipeline is not lost."""
    response = client.get(f"/requests/{submitted['request_id']}", headers=EMPLOYEE)

    assert response.status_code == 200
    assert response.json()["status"] == "received"


@needs_postgres
def test_a_request_lists_under_its_own_requester(
    client: TestClient, submitted: dict[str, Any]
) -> None:
    listed = client.get("/requests", headers=EMPLOYEE).json()
    assert submitted["request_id"] in {item["request_id"] for item in listed}


@needs_postgres
def test_another_employees_request_is_not_found(
    client: TestClient, submitted: dict[str, Any]
) -> None:
    """404 rather than 403: a distinct error would confirm it exists."""
    response = client.get(f"/requests/{submitted['request_id']}", headers=OTHER_EMPLOYEE)
    assert response.status_code == 404


@needs_postgres
def test_another_employees_request_does_not_list(
    client: TestClient, submitted: dict[str, Any]
) -> None:
    listed = client.get("/requests", headers=OTHER_EMPLOYEE).json()
    assert submitted["request_id"] not in {item["request_id"] for item in listed}


# --- lookups ---


def test_an_unknown_request_is_not_found(client: TestClient) -> None:
    response = client.get("/requests/not-a-uuid", headers=EMPLOYEE)
    assert response.status_code == 404


def test_an_approval_that_is_not_pending_is_not_found(client: TestClient) -> None:
    response = client.get("/approvals/not-a-uuid", headers=REVIEWER)
    assert response.status_code == 404


def test_deciding_on_an_unknown_request_is_not_found(client: TestClient) -> None:
    response = client.post(
        "/approvals/not-a-uuid/decision",
        json={"decision": "approved"},
        headers=REVIEWER,
    )
    assert response.status_code == 404


def test_an_unknown_decision_is_refused(client: TestClient) -> None:
    """A malformed decision must not release a statutory disclosure."""
    response = client.post(
        "/approvals/not-a-uuid/decision",
        json={"decision": "looks_fine"},
        headers=REVIEWER,
    )
    assert response.status_code == 422


def test_deciding_requires_a_reviewer(client: TestClient) -> None:
    response = client.post(
        "/approvals/not-a-uuid/decision",
        json={"decision": "approved"},
        headers=EMPLOYEE,
    )
    assert response.status_code == 403


# --- health ---


def test_health_reports_tracing(client: TestClient) -> None:
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert "tracing" in body
