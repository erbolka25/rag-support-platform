"""FastAPI dependencies: everything comes from app.state, set up by create_app."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Annotated

from fastapi import Depends, HTTPException, Request
from rag_core.service import RagService
from sqlmodel import Session

from support_copilot.config import CopilotSettings
from support_copilot.models import Ticket


def get_session(request: Request) -> Iterator[Session]:
    with Session(request.app.state.engine) as session:
        yield session


def get_rag(request: Request) -> RagService:
    return request.app.state.rag


def get_settings(request: Request) -> CopilotSettings:
    return request.app.state.settings


SessionDep = Annotated[Session, Depends(get_session)]
RagDep = Annotated[RagService, Depends(get_rag)]
SettingsDep = Annotated[CopilotSettings, Depends(get_settings)]


def api_error(status_code: int, code: str, message: str) -> HTTPException:
    return HTTPException(status_code, detail={"code": code, "message": message})


def get_ticket_or_404(ticket_id: int, session: SessionDep) -> Ticket:
    ticket = session.get(Ticket, ticket_id)
    if ticket is None:
        raise api_error(404, "ticket_not_found", f"Ticket {ticket_id} does not exist.")
    return ticket


TicketDep = Annotated[Ticket, Depends(get_ticket_or_404)]
