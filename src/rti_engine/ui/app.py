"""The web interface.

Three screens: submit a request, watch it, and decide on one. Each is a
client of the API and holds no logic of its own — the tier a request
takes and the findings a reviewer sees are decided behind the API, and a
second implementation here would give two answers to the same question.

Run with:

    uv run streamlit run src/rti_engine/ui/app.py
"""

import time
from typing import Any

import streamlit as st

from rti_engine.ui.client import ApiClient, ApiError

TITLE = "Pay Information Requests"

DECISIONS = {
    "approved": "Approve and send",
    "changes_requested": "Send back for changes",
    "rejected": "Reject",
}

TERMINAL_STATUSES = {"completed", "approved", "rejected", "failed"}


def sidebar() -> ApiClient:
    """Collect the identity every request is made under.

    A stand-in for authentication, and labelled as one. Making it look
    like a login would suggest the identity is verified here, when in a
    deployment it would arrive from a signed session.
    """
    st.sidebar.header("Session")
    st.sidebar.caption(
        "Stands in for an identity provider. In deployment these come "
        "from a signed token rather than a text box."
    )

    employee_id = st.sidebar.text_input("Employee ID", value="EMP-00001")
    is_reviewer = st.sidebar.checkbox("Sign in as a reviewer", value=False)
    base_url = st.sidebar.text_input("API", value="http://localhost:8000")

    client = ApiClient(employee_id, is_reviewer=is_reviewer, base_url=base_url)

    try:
        health = client.health()
        st.sidebar.success(f"Connected — tracing {'on' if health['tracing'] else 'off'}")
    except ApiError as error:
        st.sidebar.error(str(error))

    return client


POLL_SECONDS = 2.0
POLL_LIMIT_SECONDS = 210.0
"""Long enough for a disclosure request to reach a decision point.

A tier 0 or 1 request settles in about twenty seconds. A tier 2 request
runs the full pipeline first, which takes minutes, so a shorter limit
would give up just before the interesting part.
"""


def _await_outcome(client: ApiClient, request_id: str) -> None:
    """Show a request's progress until it settles.

    Polling rather than streaming: the work happens in a background task
    behind the API, and a status endpoint is a smaller thing to get right
    than a channel held open across a three-minute pipeline.
    """
    status_area = st.empty()
    result_area = st.empty()
    waited = 0.0

    while waited < POLL_LIMIT_SECONDS:
        try:
            detail = client.request_detail(request_id)
        except ApiError as error:
            status_area.error(str(error))
            return

        status = detail["status"]
        status_area.info(f"{status.replace('_', ' ').title()} — {_status_note(status)}")

        if status == "awaiting_approval":
            # The employee sees nothing here, and that is the design: a
            # disclosure involving colleagues' pay is not sent until a
            # person has read it.
            result_area.warning(
                "Your request compares pay across colleagues, so a person "
                "reviews the response before it is sent. You will be "
                "notified when that is done."
            )
            return

        if status in TERMINAL_STATUSES:
            if detail.get("letter"):
                with result_area.container():
                    st.markdown("**Response**")
                    st.text(detail["letter"])
                    if detail.get("citations"):
                        st.markdown("**Sources**")
                        for citation in detail["citations"]:
                            st.caption(citation)
            elif status == "failed":
                result_area.warning(
                    "This could not be completed automatically. A person will follow up."
                )
            return

        time.sleep(POLL_SECONDS)
        waited += POLL_SECONDS

    status_area.info(
        "Still working. Your request is safe — check My requests for the answer when it is ready."
    )


def submit_screen(client: ApiClient) -> None:
    """Where an employee asks for pay information, and sees the answer."""
    st.header("Make a request")
    st.write(
        "Ask about your own pay, about how pay is set, or about how your pay "
        "compares with colleagues doing work of equal value."
    )

    request_text = st.text_area(
        "Your request",
        height=140,
        placeholder="What is the average pay for men and women at my level?",
    )

    if st.button("Submit", type="primary", disabled=not request_text.strip()):
        try:
            result = client.submit(request_text)
        except ApiError as error:
            st.error(str(error))
            return

        st.session_state["last_request_id"] = result["request_id"]

    request_id = st.session_state.get("last_request_id")
    if not request_id:
        return

    st.divider()
    st.caption(f"Reference {request_id}")
    _await_outcome(client, request_id)


def _status_note(status: str) -> str:
    """Explain a status in terms the requester cares about."""
    return {
        "received": "Received and being classified.",
        "in_progress": "Being worked on.",
        "awaiting_approval": "Waiting for a person to review the response.",
        "approved": "Approved. The response below is final.",
        "rejected": "The response was not approved. Someone will be in touch.",
        "completed": "Answered.",
        "failed": "Could not be completed automatically; a person will follow up.",
    }.get(status, status)


def my_requests_screen(client: ApiClient) -> None:
    """Where an employee follows what happened to their requests."""
    st.header("My requests")

    try:
        requests = client.my_requests()
    except ApiError as error:
        st.error(str(error))
        return

    if not requests:
        st.info("You have not made any requests yet.")
        return

    for summary in requests:
        label = f"{summary['status']} · {summary['created_at'][:16]}"
        with st.expander(label, expanded=summary["status"] in TERMINAL_STATUSES):
            st.caption(_status_note(summary["status"]))

            try:
                detail = client.request_detail(summary["request_id"])
            except ApiError as error:
                st.error(str(error))
                continue

            if detail.get("letter"):
                st.markdown("**Response**")
                st.text(detail["letter"])

                if detail.get("citations"):
                    st.markdown("**Sources**")
                    for citation in detail["citations"]:
                        st.caption(citation)

            if detail.get("errors"):
                st.warning("This request did not complete automatically.")


def _findings(title: str, findings: list[dict[str, Any]]) -> None:
    """Show what the automated review objected to."""
    if not findings:
        return

    st.markdown(f"**{title}**")
    for finding in findings:
        st.markdown(f"- *{finding['kind']}* — {finding['problem']}")
        if quote := finding.get("quote"):
            st.caption(f"“{quote}”")


def approval_screen(client: ApiClient) -> None:
    """Where a reviewer decides on a statutory disclosure.

    The letter is shown alongside what the automated review objected to
    and which figures were used. A decision made without those is a rubber
    stamp, which is the thing this screen exists to avoid.
    """
    st.header("Awaiting approval")

    try:
        pending = client.approvals()
    except ApiError as error:
        st.error(str(error))
        st.caption("Approving requires signing in as a reviewer.")
        return

    if not pending:
        st.info("Nothing is waiting for a decision.")
        return

    st.caption(f"{len(pending)} waiting, longest first.")

    for summary in pending:
        request_id = summary["request_id"]

        with st.expander(f"{summary['requester_employee_id']} · {request_id[:8]}"):
            try:
                item = client.approval_detail(request_id)
            except ApiError as error:
                st.error(str(error))
                continue

            if item["reviewer_approved"]:
                st.success("The automated review raised no blocking findings.")
            else:
                st.warning(
                    f"The automated review did not approve this draft after "
                    f"{item['revisions_used']} revisions."
                )

            st.markdown("**Proposed response**")
            st.text(item["letter"])

            _findings("Blocking findings", item["blocking_findings"])
            _findings("Advisory findings", item["advisory_findings"])

            if item["figures_used"]:
                st.markdown("**Figures used**")
                st.dataframe(item["figures_used"], hide_index=True)

            if item["citations"]:
                st.markdown("**Sources cited**")
                for citation in item["citations"]:
                    st.caption(citation)

            st.divider()
            comment = st.text_area(
                "Comment",
                key=f"comment-{request_id}",
                height=90,
                placeholder="Required when sending a draft back.",
            )

            columns = st.columns(len(DECISIONS))
            for column, (decision, label) in zip(columns, DECISIONS.items(), strict=True):
                if column.button(label, key=f"{decision}-{request_id}"):
                    try:
                        client.decide(request_id, decision, comment or None)
                    except ApiError as error:
                        st.error(str(error))
                        continue

                    st.success(f"Recorded: {decision.replace('_', ' ')}")
                    st.rerun()


SCREENS = {
    "Make a request": submit_screen,
    "My requests": my_requests_screen,
    "Awaiting approval": approval_screen,
}


def main() -> None:
    st.set_page_config(page_title=TITLE, layout="wide")
    st.title(TITLE)

    client = sidebar()
    choice = st.sidebar.radio("Screen", list(SCREENS))
    SCREENS[choice](client)


main()
