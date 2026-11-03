import pytest
from sqlmodel import Session

from support_copilot.models import Confidence, Draft, Ticket, TicketStatus
from tests.conftest import TICKET, create_ticket


def set_status(client, ticket_id: int, status: TicketStatus, with_draft: bool = False) -> None:
    """Put a ticket into a state directly in the DB (drafting is covered in test_drafting)."""
    with Session(client.app.state.engine) as session:
        ticket = session.get(Ticket, ticket_id)
        ticket.status = status
        if with_draft:
            session.add(Draft(ticket_id=ticket_id, text="Ответ [1]", confidence=Confidence.answered,
                              citations=[{"file": "a.txt", "page": None, "fragment": "..."}]))
        session.add(ticket)
        session.commit()


def test_create_ticket(client):
    body = create_ticket(client)
    assert body["id"] == 1
    assert body["status"] == "new"
    assert {k: body[k] for k in TICKET} == TICKET
    assert body["created_at"].endswith("Z") or "+00:00" in body["created_at"]


def test_create_ticket_strips_whitespace(client):
    body = create_ticket(client, customer_name="  Бакыт ", question="  Где мой заказ?\n")
    assert body["customer_name"] == "Бакыт"
    assert body["question"] == "Где мой заказ?"


@pytest.mark.parametrize("question", ["", "   ", "\n\t"])
def test_empty_question_is_422(client, question):
    resp = client.post("/tickets", json=TICKET | {"question": question})
    assert resp.status_code == 422
    assert resp.json()["detail"][0]["loc"] == ["body", "question"]


@pytest.mark.parametrize(
    "field, value",
    [("customer_email", "not-an-email"), ("customer_name", ""), ("question", "x" * 5001)],
)
def test_invalid_ticket_fields_are_422(client, field, value):
    assert client.post("/tickets", json=TICKET | {field: value}).status_code == 422


def test_missing_question_is_422(client):
    payload = {k: v for k, v in TICKET.items() if k != "question"}
    assert client.post("/tickets", json=payload).status_code == 422


def test_list_tickets_and_filter_by_status(client):
    assert client.get("/tickets").json() == []
    for i in range(3):
        create_ticket(client, question=f"Вопрос {i}")
    set_status(client, 2, TicketStatus.escalated)

    assert [t["id"] for t in client.get("/tickets").json()] == [1, 2, 3]
    assert [t["id"] for t in client.get("/tickets?status=new").json()] == [1, 3]
    assert [t["id"] for t in client.get("/tickets?status=escalated").json()] == [2]
    assert client.get("/tickets?status=resolved").json() == []


def test_unknown_status_filter_is_422(client):
    assert client.get("/tickets?status=closed").status_code == 422


def test_get_ticket_with_drafts(client):
    create_ticket(client)
    assert client.get("/tickets/1").json()["drafts"] == []

    set_status(client, 1, TicketStatus.drafted, with_draft=True)
    body = client.get("/tickets/1").json()
    assert body["status"] == "drafted"
    [draft] = body["drafts"]
    assert draft["ticket_id"] == 1
    assert draft["confidence"] == "answered"
    assert draft["citations"] == [{"file": "a.txt", "page": None, "fragment": "..."}]


@pytest.mark.parametrize(
    "method, path",
    [("get", "/tickets/42"), ("post", "/tickets/42/resolve"), ("post", "/tickets/42/draft")],
)
def test_unknown_ticket_is_404(client, method, path):
    resp = getattr(client, method)(path)
    assert resp.status_code == 404
    assert resp.json()["detail"]["code"] == "ticket_not_found"


def test_resolve_drafted_ticket(client):
    create_ticket(client)
    set_status(client, 1, TicketStatus.drafted, with_draft=True)

    resp = client.post("/tickets/1/resolve")
    assert resp.status_code == 200
    assert resp.json()["status"] == "resolved"
    assert len(resp.json()["drafts"]) == 1
    assert client.get("/tickets?status=resolved").json()[0]["id"] == 1


@pytest.mark.parametrize("status", [TicketStatus.new, TicketStatus.escalated, TicketStatus.resolved])
def test_resolve_requires_drafted_status(client, status):
    create_ticket(client)
    set_status(client, 1, status)

    resp = client.post("/tickets/1/resolve")
    assert resp.status_code == 409
    assert resp.json()["detail"]["code"] == "not_drafted"
    assert client.get("/tickets/1").json()["status"] == status


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "documents": 0, "llm_enabled": False}
