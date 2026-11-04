"""Fill an empty knowledge base with the demo documents and tickets at startup."""

from __future__ import annotations

import json
import logging

from rag_core.service import RagService
from sqlalchemy import Engine
from sqlmodel import Session, select

from support_copilot.config import CopilotSettings
from support_copilot.models import KnowledgeDocument, Ticket
from support_copilot.schemas import TicketCreate

logger = logging.getLogger(__name__)


def load_demo_data(settings: CopilotSettings, rag: RagService, engine: Engine) -> None:
    """Upload demo_kb_dir/*.txt and create the demo tickets, unless the KB has documents.

    Only embeddings run here, no LLM calls: drafts are made when a visitor asks for them.
    """
    if settings.demo_kb_dir is None:
        return
    with Session(engine) as session:
        if session.exec(select(KnowledgeDocument)).first() is not None:
            logger.info("Demo data: knowledge base is not empty, skipped")
            return

        for path in sorted(settings.demo_kb_dir.glob("*.txt")):
            added = rag.add_document(path.read_bytes(), path.name)
            session.add(KnowledgeDocument(filename=added.filename, doc_id=added.doc_id))
            logger.info("Demo data: %s, %d chunks", added.filename, added.chunks)

        if settings.demo_tickets_file is not None:
            raw = json.loads(settings.demo_tickets_file.read_text(encoding="utf-8"))
            for item in raw:
                session.add(Ticket(**TicketCreate.model_validate(item).model_dump()))
            logger.info("Demo data: %d tickets", len(raw))
        session.commit()
