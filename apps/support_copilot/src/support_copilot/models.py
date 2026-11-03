"""Database tables: knowledge-base documents, tickets and their reply drafts."""

from datetime import UTC, datetime
from enum import StrEnum

from sqlalchemy import JSON, Column, DateTime, TypeDecorator
from sqlmodel import Field, Relationship, SQLModel


def utcnow() -> datetime:
    return datetime.now(UTC)


class UTCDateTime(TypeDecorator):
    """SQLite drops the timezone; store UTC and hand it back as an aware datetime."""

    impl = DateTime
    cache_ok = True

    def process_bind_param(self, value, dialect):
        return value.astimezone(UTC).replace(tzinfo=None) if value else value

    def process_result_value(self, value, dialect):
        return value.replace(tzinfo=UTC) if value else value


def _created_at():
    return Field(default_factory=utcnow, sa_type=UTCDateTime, nullable=False)


class TicketStatus(StrEnum):
    new = "new"
    drafted = "drafted"
    escalated = "escalated"
    resolved = "resolved"


class Confidence(StrEnum):
    answered = "answered"
    no_answer = "no_answer"


class KnowledgeDocument(SQLModel, table=True):
    __tablename__ = "knowledge_documents"

    id: int | None = Field(default=None, primary_key=True)
    filename: str
    doc_id: str = Field(index=True, unique=True)  # content hash assigned by RagService
    created_at: datetime = _created_at()


class Ticket(SQLModel, table=True):
    __tablename__ = "tickets"

    id: int | None = Field(default=None, primary_key=True)
    customer_name: str
    customer_email: str
    question: str
    status: TicketStatus = Field(default=TicketStatus.new, index=True)
    created_at: datetime = _created_at()

    drafts: list["Draft"] = Relationship(
        back_populates="ticket",
        sa_relationship_kwargs={"order_by": "Draft.id", "cascade": "all, delete-orphan"},
    )


class Draft(SQLModel, table=True):
    __tablename__ = "drafts"

    id: int | None = Field(default=None, primary_key=True)
    ticket_id: int = Field(foreign_key="tickets.id", index=True)
    text: str
    # [{"file": ..., "page": int | None, "fragment": ...}], numbered as the [n] markers in text
    citations: list[dict] = Field(default_factory=list, sa_column=Column(JSON, nullable=False))
    confidence: Confidence
    escalation_reason: str | None = None  # set when confidence is no_answer
    created_at: datetime = _created_at()

    ticket: Ticket | None = Relationship(back_populates="drafts")
