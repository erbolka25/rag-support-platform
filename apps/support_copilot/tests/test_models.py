from datetime import UTC

import pytest
from sqlalchemy.exc import IntegrityError
from sqlmodel import Session, select

from support_copilot.config import PROJECT_ROOT, CopilotSettings
from support_copilot.models import Confidence, Draft, KnowledgeDocument, Ticket, TicketStatus


def make_ticket(**overrides) -> Ticket:
    values = {"customer_name": "Айгуль", "customer_email": "a@example.com", "question": "Где заказ?"}
    return Ticket(**(values | overrides))


def test_ticket_defaults_to_new_with_utc_timestamp(session):
    session.add(make_ticket())
    session.commit()
    session.expire_all()

    ticket = session.exec(select(Ticket)).one()
    assert ticket.id is not None
    assert ticket.status is TicketStatus.new
    assert ticket.created_at.tzinfo == UTC


def test_draft_citations_round_trip_as_json(engine):
    citations = [
        {"file": "delivery.txt", "page": None, "fragment": "Доставка по Бишкеку — 1 день."},
        {"file": "manual.pdf", "page": 3, "fragment": "Гарантия 12 месяцев."},
    ]
    with Session(engine) as session:
        ticket = make_ticket()
        session.add(ticket)
        session.commit()
        session.add(
            Draft(ticket_id=ticket.id, text="Ответ [1][2]", citations=citations,
                  confidence=Confidence.answered)
        )
        session.commit()

    with Session(engine) as session:
        draft = session.exec(select(Draft)).one()
        assert draft.citations == citations
        assert draft.confidence is Confidence.answered
        assert draft.escalation_reason is None


def test_ticket_drafts_are_ordered_oldest_first(session):
    ticket = make_ticket()
    session.add(ticket)
    session.commit()
    for reason in ("no_answer_in_kb", None):
        session.add(Draft(ticket_id=ticket.id, text="t", confidence=Confidence.no_answer,
                          escalation_reason=reason))
    session.commit()
    session.refresh(ticket)

    assert [d.escalation_reason for d in ticket.drafts] == ["no_answer_in_kb", None]
    assert all(d.ticket is ticket for d in ticket.drafts)


def test_draft_requires_existing_ticket(session):
    session.add(Draft(ticket_id=999, text="t", confidence=Confidence.answered))
    with pytest.raises(IntegrityError):
        session.commit()


def test_document_doc_id_is_unique(session):
    session.add(KnowledgeDocument(filename="a.txt", doc_id="abc"))
    session.commit()
    session.add(KnowledgeDocument(filename="b.txt", doc_id="abc"))
    with pytest.raises(IntegrityError):
        session.commit()


def test_settings_resolve_paths_inside_the_project():
    settings = CopilotSettings(_env_file=None)
    assert settings.data_dir == PROJECT_ROOT / "data"
    assert settings.resolved_database_url.endswith("support_copilot/data/copilot.db")
    assert settings.rag_data_dir == PROJECT_ROOT / "data" / "rag"
    assert settings.max_upload_bytes == 10 * 1024 * 1024


def test_settings_keep_an_explicit_database_url(tmp_path):
    settings = CopilotSettings(_env_file=None, data_dir=tmp_path, database_url="sqlite://")
    assert settings.data_dir == tmp_path
    assert settings.resolved_database_url == "sqlite://"


def test_rag_settings_pass_the_llm_base_url():
    from support_copilot.rag import rag_settings

    settings = CopilotSettings(
        _env_file=None,
        llm_provider="openai",
        llm_base_url="https://api.groq.com/openai/v1",
        llm_model="llama-3.3-70b-versatile",
    )
    rag = rag_settings(settings)
    assert rag.llm_base_url == "https://api.groq.com/openai/v1"
    assert rag.resolved_llm_model == "llama-3.3-70b-versatile"
