"""Tickets: create, list, inspect, draft a reply and resolve."""

from __future__ import annotations

import logging

from fastapi import APIRouter
from sqlmodel import select

from support_copilot.api.deps import RagDep, SessionDep, SettingsDep, TicketDep, api_error
from support_copilot.drafting import draft_reply
from support_copilot.models import Draft, Ticket, TicketStatus
from support_copilot.schemas import (
    DraftRead,
    DraftResult,
    ErrorDetail,
    TicketCreate,
    TicketDetail,
    TicketRead,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/tickets", tags=["tickets"])

NOT_FOUND = {404: {"model": ErrorDetail}}


@router.post("", response_model=TicketRead, status_code=201)
def create_ticket(body: TicketCreate, session: SessionDep) -> Ticket:
    ticket = Ticket(**body.model_dump())
    session.add(ticket)
    session.commit()
    session.refresh(ticket)
    return ticket


@router.get("", response_model=list[TicketRead])
def list_tickets(session: SessionDep, status: TicketStatus | None = None) -> list[Ticket]:
    query = select(Ticket).order_by(Ticket.id)
    if status is not None:
        query = query.where(Ticket.status == status)
    return list(session.exec(query))


@router.get("/{ticket_id}", response_model=TicketDetail, responses=NOT_FOUND)
def get_ticket(ticket: TicketDep) -> Ticket:
    return ticket


@router.post(
    "/{ticket_id}/draft",
    response_model=DraftResult,
    status_code=201,
    responses={**NOT_FOUND, 409: {"model": ErrorDetail}},
)
def draft_ticket(
    ticket: TicketDep, session: SessionDep, rag: RagDep, settings: SettingsDep
) -> DraftResult:
    """Draft a reply from the knowledge base; escalate when it can't be grounded in it.

    A drafted or escalated ticket can be drafted again (e.g. after the knowledge base grew).
    """
    if ticket.status is TicketStatus.resolved:
        raise api_error(409, "already_resolved", "The ticket is already resolved.")
    outcome = draft_reply(rag, ticket.customer_name, ticket.question, settings.company_name)
    draft = Draft(
        ticket_id=ticket.id,
        text=outcome.text,
        citations=outcome.citations,
        confidence=outcome.confidence,
        escalation_reason=outcome.escalation_reason,
    )
    ticket.status = outcome.status
    session.add_all([draft, ticket])
    session.commit()
    session.refresh(draft)
    session.refresh(ticket)
    if outcome.escalation_reason:
        logger.info("Ticket %s escalated: %s", ticket.id, outcome.escalation_reason)
    return DraftResult(
        ticket=TicketRead.model_validate(ticket), draft=DraftRead.model_validate(draft)
    )


@router.post(
    "/{ticket_id}/resolve",
    response_model=TicketDetail,
    responses={**NOT_FOUND, 409: {"model": ErrorDetail}},
)
def resolve_ticket(ticket: TicketDep, session: SessionDep) -> Ticket:
    """The operator confirms the latest draft; only a drafted ticket can be resolved."""
    if ticket.status is not TicketStatus.drafted:
        raise api_error(
            409,
            "not_drafted",
            f"Only a drafted ticket can be resolved; this one is '{ticket.status}'.",
        )
    ticket.status = TicketStatus.resolved
    session.add(ticket)
    session.commit()
    session.refresh(ticket)
    return ticket
