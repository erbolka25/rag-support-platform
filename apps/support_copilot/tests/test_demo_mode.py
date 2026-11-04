"""Public demo mode: per-IP rate limit and demo data loaded at startup."""

import json

from support_copilot.demo_data import load_demo_data

from .conftest import create_ticket, upload


def test_draft_and_upload_share_hourly_budget(make_client):
    client = make_client(rate_limit_per_hour=2)
    ticket = create_ticket(client)
    assert upload(client).status_code == 201
    assert client.post(f"/tickets/{ticket['id']}/draft").status_code == 201
    resp = client.post(f"/tickets/{ticket['id']}/draft")
    assert resp.status_code == 429
    assert resp.json()["detail"]["code"] == "rate_limited"
    assert 0 < int(resp.headers["Retry-After"]) <= 3600
    assert upload(client).status_code == 429
    # Reads and ticket creation are not limited.
    assert client.get("/tickets").status_code == 200
    assert client.post("/tickets", json={"customer_name": "A", "customer_email": "a@example.com",
                                         "question": "?"}).status_code == 201


def test_no_limit_by_default(client):
    ticket = create_ticket(client)
    for _ in range(5):
        assert client.post(f"/tickets/{ticket['id']}/draft").status_code == 201


def test_demo_data_loaded_once(make_client, tmp_path):
    kb = tmp_path / "kb"
    kb.mkdir()
    (kb / "delivery.txt").write_text("Доставка в Ош стоит 300 сом.", encoding="utf-8")
    (kb / "returns.txt").write_text("Возврат в течение 14 дней.", encoding="utf-8")
    tickets = tmp_path / "tickets.json"
    tickets.write_text(json.dumps([{"customer_name": "Айгуль", "customer_email":
                                    "aigul@example.com", "question": "Доставка в Ош?"}]),
                       encoding="utf-8")
    client = make_client(demo_kb_dir=kb, demo_tickets_file=tickets)
    state = client.app.state
    load_demo_data(state.settings, state.rag, state.engine)
    load_demo_data(state.settings, state.rag, state.engine)  # restart: not loaded twice

    assert [d["filename"] for d in client.get("/kb").json()] == ["delivery.txt", "returns.txt"]
    assert [t["status"] for t in client.get("/tickets").json()] == ["new"]
    assert client.get("/health").json()["documents"] == 2


def test_shipped_demo_tickets_are_valid():
    from pathlib import Path

    from support_copilot.schemas import TicketCreate

    path = Path(__file__).resolve().parents[1] / "demo" / "shop_tickets.json"
    tickets = json.loads(path.read_text(encoding="utf-8"))
    assert len(tickets) == 10
    for item in tickets:
        TicketCreate.model_validate(item)
