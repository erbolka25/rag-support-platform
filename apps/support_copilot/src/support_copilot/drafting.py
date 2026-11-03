"""Turn a RagService answer into a customer reply draft, or escalate the ticket.

A ticket is escalated whenever the reply could not be tied to the knowledge base:
the LLM or the embedder is down, nothing relevant was found, the model said NO_ANSWER,
or its reply carried no valid citation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from rag_core.generation import detect_language
from rag_core.retrieval import SearchResult
from rag_core.service import AskResult, RagService

from support_copilot.models import Confidence, TicketStatus

FRAGMENT_CHARS = 300
_CITATION = re.compile(r"\[(\d+)\]")
_CITATION_RUN = re.compile(r"(?:\[\d+\])+")


class Reason:
    """Stable escalation codes stored in Draft.escalation_reason."""

    LLM_UNAVAILABLE = "llm_unavailable"
    EMBEDDING_UNAVAILABLE = "embedding_unavailable"
    NO_RELEVANT_DOCUMENTS = "no_relevant_documents"
    NO_ANSWER_IN_KB = "no_answer_in_kb"
    NO_VALID_CITATIONS = "no_valid_citations"


# Operator-facing explanations; the operator UI is Russian, like the knowledge base.
REASON_TEXT = {
    Reason.LLM_UNAVAILABLE: "LLM недоступна ({notice}), черновик не сгенерирован.",
    Reason.EMBEDDING_UNAVAILABLE: "Модель поиска (эмбеддинги) недоступна.",
    Reason.NO_RELEVANT_DOCUMENTS: "В базе знаний не найдено релевантных фрагментов.",
    Reason.NO_ANSWER_IN_KB: "В базе знаний нет ответа на этот вопрос.",
    Reason.NO_VALID_CITATIONS: "Ответ модели не подтверждён ни одной цитатой из базы знаний.",
}


@dataclass(frozen=True)
class Template:
    greeting: str
    closing: str
    sources: str
    page: str  # format with the page number


TEMPLATES = {
    "Russian": Template(
        greeting="Здравствуйте, {name}!",
        closing=(
            "Если у вас остались вопросы, просто ответьте на это письмо — мы будем рады помочь.\n\n"
            "С уважением,\nслужба поддержки{company}"
        ),
        sources="Источники:",
        page="стр. {page}",
    ),
    "English": Template(
        greeting="Hello, {name}!",
        closing=(
            "If you have any other questions, just reply to this email — we are happy to help.\n\n"
            "Best regards,\nCustomer Support{company}"
        ),
        sources="Sources:",
        page="p. {page}",
    ),
    "Kyrgyz": Template(
        greeting="Саламатсызбы, {name}!",
        closing=(
            "Дагы суроолоруңуз болсо, ушул катка жооп жазыңыз — жардам берүүгө даярбыз.\n\n"
            "Урматтоо менен,\nколдоо кызматы{company}"
        ),
        sources="Булактар:",
        page="{page}-бет",
    ),
}
DEFAULT_LANGUAGE = "Russian"


@dataclass
class DraftOutcome:
    status: TicketStatus
    confidence: Confidence
    text: str
    citations: list[dict] = field(default_factory=list)
    escalation_reason: str | None = None


def _shorten(text: str) -> str:
    text = " ".join(text.split())
    return text if len(text) <= FRAGMENT_CHARS else text[: FRAGMENT_CHARS - 1].rstrip() + "…"


def source_key(result: SearchResult) -> tuple[str, int | None]:
    return result.chunk.filename, result.chunk.page


def group_by_source(results: list[SearchResult]) -> list[list[SearchResult]]:
    """Chunks of the same file and page form one source, in order of first appearance."""
    groups: dict[tuple[str, int | None], list[SearchResult]] = {}
    for result in results:
        groups.setdefault(source_key(result), []).append(result)
    return list(groups.values())


def citation_dict(group: list[SearchResult]) -> dict:
    chunk = group[0].chunk
    # A shortened part already ends with "…", so it doubles as the separator.
    fragment = " … ".join(_shorten(r.chunk.text).removesuffix("…") for r in group[:-1])
    fragment = f"{fragment} … " if fragment else ""
    fragment += _shorten(group[-1].chunk.text)
    return {"file": chunk.filename, "page": chunk.page, "fragment": fragment}


def renumber_citations(text: str, numbers: dict[int, int]) -> str:
    """Rewrite each [n] to [numbers[n]], drop unknown ones and repeats like [1][1]."""

    def replace(match: re.Match) -> str:
        n = numbers.get(int(match.group(1)))
        return f"[{n}]" if n else ""

    def dedupe(match: re.Match) -> str:
        unique = dict.fromkeys(_CITATION.findall(match.group(0)))
        return "".join(f"[{n}]" for n in unique)

    return _CITATION_RUN.sub(dedupe, _CITATION.sub(replace, text))


def compose_reply(
    customer_name: str, question: str, answer: str, citations: list[dict], company: str = ""
) -> str:
    """Polite reply in the question's language, with the numbered sources at the end."""
    template = TEMPLATES[detect_language(question) or DEFAULT_LANGUAGE]
    sources = []
    for i, citation in enumerate(citations, 1):
        location = citation["file"]
        if citation["page"] is not None:
            location += ", " + template.page.format(page=citation["page"])
        sources.append(f"[{i}] {location}")
    return "\n\n".join(
        [
            template.greeting.format(name=customer_name),
            answer.strip(),
            template.closing.format(company=f" {company}" if company else ""),
            "\n".join([template.sources, *sources]),
        ]
    )


def escalate(reason: str, detail: str | None = None, results=(), notice: str | None = None):
    lines = ["Эскалация: " + REASON_TEXT[reason].format(notice=notice)]
    if detail:
        lines.append(f"Пояснение модели: {detail}")
    citations = [citation_dict(g) for g in group_by_source(list(results))]
    if citations:
        # Not an answer: passages the operator may use to reply by hand.
        lines.append("Найденные фрагменты приложены к черновику для оператора.")
    return DraftOutcome(
        status=TicketStatus.escalated,
        confidence=Confidence.no_answer,
        text="\n".join(lines),
        citations=citations,
        escalation_reason=reason,
    )


def outcome_from_answer(
    result: AskResult, customer_name: str, question: str, company: str = ""
) -> DraftOutcome:
    if result.mode == "retrieval_only":
        if result.notice == "embedding_unavailable":
            return escalate(Reason.EMBEDDING_UNAVAILABLE)
        return escalate(Reason.LLM_UNAVAILABLE, results=result.results, notice=result.notice)
    if not result.results:
        return escalate(Reason.NO_RELEVANT_DOCUMENTS)
    if not result.answer_found:
        # generate_answer keeps the model's NO_ANSWER explanation, and blanks a reply
        # it discarded for lack of a valid citation.
        if result.answer:
            return escalate(Reason.NO_ANSWER_IN_KB, detail=result.answer)
        return escalate(Reason.NO_VALID_CITATIONS)

    # The answer's [n] point into result.results; the reply numbers sources 1, 2, ... in
    # order of first use, with every chunk of one file and page under a single number.
    cited = [c for c in result.citations if c in result.results]
    groups = group_by_source(cited)
    number = {source_key(g[0]): i for i, g in enumerate(groups, 1)}
    numbers = {result.results.index(c) + 1: number[source_key(c)] for c in cited}
    answer = renumber_citations(result.answer or "", numbers)
    if not cited or not _CITATION.search(answer):
        return escalate(Reason.NO_VALID_CITATIONS)
    citations = [citation_dict(g) for g in groups]
    return DraftOutcome(
        status=TicketStatus.drafted,
        confidence=Confidence.answered,
        text=compose_reply(customer_name, question, answer, citations, company),
        citations=citations,
    )


def draft_reply(rag: RagService, customer_name: str, question: str, company: str = "") -> DraftOutcome:
    return outcome_from_answer(rag.ask(question), customer_name, question, company)
