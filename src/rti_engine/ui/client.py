"""Talking to the API from the UI.

The UI holds no logic of its own. Every decision — which tier a request
takes, whether a group is large enough to report on, what a reviewer sees
— already lives behind the API, and duplicating any of it here would give
two answers to the same question.

Identity travels as headers because that is what the API expects. In a
deployment these would come from a signed session; here they come from a
sidebar, which is honest about being a stand-in rather than pretending to
be authentication.
"""

from typing import Any

import httpx

from rti_engine.api.security import EMPLOYEE_HEADER, REVIEWER_HEADER

DEFAULT_BASE_URL = "http://localhost:8000"
TIMEOUT_SECONDS = 30.0


class ApiError(RuntimeError):
    """Raised when the API refuses or cannot answer a request."""


class ApiClient:
    """A thin client for one identity."""

    def __init__(
        self,
        employee_id: str,
        is_reviewer: bool = False,
        base_url: str = DEFAULT_BASE_URL,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.headers = {EMPLOYEE_HEADER: employee_id}
        if is_reviewer:
            self.headers[REVIEWER_HEADER] = "true"

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        """Send one request, turning a failure into a readable message.

        The API's own detail is preferred over a status code: "this action
        requires a reviewer" tells a person what to do, where "403" does
        not.
        """
        try:
            with httpx.Client(timeout=TIMEOUT_SECONDS) as client:
                response = client.request(
                    method, f"{self.base_url}{path}", headers=self.headers, **kwargs
                )
        except httpx.RequestError as error:
            raise ApiError(f"could not reach the service at {self.base_url}") from error

        if response.is_error:
            detail = _detail(response)
            raise ApiError(f"{response.status_code}: {detail}")

        return response.json()

    def health(self) -> dict[str, Any]:
        return dict(self._request("GET", "/health"))

    def submit(self, request_text: str) -> dict[str, Any]:
        return dict(self._request("POST", "/requests", json={"request_text": request_text}))

    def my_requests(self) -> list[dict[str, Any]]:
        return list(self._request("GET", "/requests"))

    def request_detail(self, request_id: str) -> dict[str, Any]:
        return dict(self._request("GET", f"/requests/{request_id}"))

    def approvals(self) -> list[dict[str, Any]]:
        return list(self._request("GET", "/approvals"))

    def approval_detail(self, request_id: str) -> dict[str, Any]:
        return dict(self._request("GET", f"/approvals/{request_id}"))

    def decide(self, request_id: str, decision: str, comment: str | None = None) -> dict[str, Any]:
        return dict(
            self._request(
                "POST",
                f"/approvals/{request_id}/decision",
                json={"decision": decision, "comment": comment},
            )
        )


def _detail(response: httpx.Response) -> str:
    """Pull the API's own error message out of a response."""
    try:
        body = response.json()
    except ValueError:
        return response.text[:200] or response.reason_phrase

    if isinstance(body, dict) and "detail" in body:
        return str(body["detail"])
    return str(body)[:200]
