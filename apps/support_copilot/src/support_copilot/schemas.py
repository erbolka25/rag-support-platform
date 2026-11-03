"""Request and response bodies."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated

from pydantic import BaseModel, ConfigDict, EmailStr, StringConstraints

from support_copilot.models import Confidence, TicketStatus

NonBlank = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class ErrorDetail(BaseModel):
    code: str
    message: str


class DocumentRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    filename: str
    doc_id: str
    created_at: datetime


class DocumentUploaded(DocumentRead):
    chunks: int
    duplicate: bool
    skipped_pages: list[int]


class TicketCreate(BaseModel):
    customer_name: Annotated[NonBlank, StringConstraints(max_length=200)]
    customer_email: EmailStr
    question: Annotated[NonBlank, StringConstraints(max_length=5000)]


class TicketRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    customer_name: str
    customer_email: str
    question: str
    status: TicketStatus
    created_at: datetime


class Citation(BaseModel):
    file: str
    page: int | None
    fragment: str


class DraftRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    ticket_id: int
    text: str
    citations: list[Citation]
    confidence: Confidence
    escalation_reason: str | None
    created_at: datetime


class TicketDetail(TicketRead):
    drafts: list[DraftRead]


class DraftResult(BaseModel):
    ticket: TicketRead
    draft: DraftRead


class Health(BaseModel):
    status: str
    documents: int
    llm_enabled: bool
