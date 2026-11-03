import pytest
from rag_core.generation import NO_ANSWER, LLMError
from rag_core.ingestion import Chunk
from rag_core.retrieval import SearchResult
from rag_core.service import AskResult

from support_copilot.drafting import (
    Reason,
    citation_dict,
    compose_reply,
    outcome_from_answer,
    renumber_citations,
)
from support_copilot.models import Confidence, TicketStatus
from tests.conftest import FakeLLM, create_ticket, upload

DELIVERY = "Доставка в Ош стоит 300 сом и занимает 2 дня. Доставка по Бишкеку бесплатная."
RETURNS = "Возврат товара возможен в течение 14 дней с момента покупки."


def make_kb_client(make_client, llm=None, **overrides):
    client = make_client(llm, **overrides)
    assert upload(client, "delivery.txt", DELIVERY).status_code == 201
    assert upload(client, "returns.txt", RETURNS).status_code == 201
    return client


def draft(client, ticket_id: int = 1) -> dict:
    resp = client.post(f"/tickets/{ticket_id}/draft")
    assert resp.status_code == 201, resp.text
    return resp.json()


def _result(n: int, filename: str, page: int | None = None, text: str = "текст") -> SearchResult:
    return SearchResult(Chunk("d", filename, page, n, 0, len(text), text), 0.9, n + 1)


# --- answered -----------------------------------------------------------------


def test_answered_ticket_is_drafted_with_sources(make_client):
    llm = FakeLLM("Доставка в Ош стоит 300 сом и занимает 2 дня [1].")
    client = make_kb_client(make_client, llm, company_name="TechNomad")
    create_ticket(client)

    body = draft(client)
    assert body["ticket"]["status"] == "drafted"
    result = body["draft"]
    assert result["confidence"] == "answered"
    assert result["escalation_reason"] is None
    assert result["citations"] == [{"file": "delivery.txt", "page": None, "fragment": DELIVERY}]
    assert result["text"] == (
        "Здравствуйте, Айгуль!\n\n"
        "Доставка в Ош стоит 300 сом и занимает 2 дня [1].\n\n"
        "Если у вас остались вопросы, просто ответьте на это письмо — мы будем рады помочь.\n\n"
        "С уважением,\nслужба поддержки TechNomad\n\n"
        "Источники:\n[1] delivery.txt"
    )
    # The question went to the LLM together with the retrieved passage.
    [(_, prompt)] = llm.calls
    assert "Сколько стоит доставка в Ош?" in prompt and "300 сом" in prompt

    detail = client.get("/tickets/1").json()
    assert detail["status"] == "drafted"
    assert [d["id"] for d in detail["drafts"]] == [result["id"]]


def test_citations_are_renumbered_to_the_sources_list(make_client):
    # Both documents are retrieved; the reply cites only the second one.
    client = make_kb_client(make_client, FakeLLM("Вернуть товар можно в течение 14 дней [2]."))
    create_ticket(client, question="Сколько стоит доставка в Ош и можно ли вернуть товар?")

    result = draft(client)["draft"]
    assert [c["file"] for c in result["citations"]] == ["returns.txt"]
    assert "14 дней [1]." in result["text"]
    assert result["text"].endswith("Источники:\n[1] returns.txt")


@pytest.mark.parametrize(
    "question, reply, greeting, sources",
    [
        ("How much is delivery to Osh?", "Delivery to Osh costs 300 som [1].",
         "Hello, Aigul!", "Sources:\n[1] delivery.txt"),
        ("Ошко жеткирүү канча турат?", "Ошко жеткирүү 300 сом турат [1].",
         "Саламатсызбы, Aigul!", "Булактар:\n[1] delivery.txt"),
    ],
)
def test_reply_follows_the_question_language(make_client, question, reply, greeting, sources):
    client = make_kb_client(make_client, FakeLLM(reply))
    create_ticket(client, customer_name="Aigul", question=question)

    text = draft(client)["draft"]["text"]
    assert text.startswith(greeting)
    assert reply in text
    assert text.endswith(sources)


# --- escalation ---------------------------------------------------------------


def test_no_answer_in_kb_is_escalated_with_the_reason(make_client):
    llm = FakeLLM(f"{NO_ANSWER}: В документах нет информации о доставке в Казахстан.")
    client = make_kb_client(make_client, llm)
    create_ticket(client, question="Доставляете ли вы в Казахстан?")

    body = draft(client)
    assert body["ticket"]["status"] == "escalated"
    result = body["draft"]
    assert result["confidence"] == "no_answer"
    assert result["escalation_reason"] == Reason.NO_ANSWER_IN_KB
    assert result["citations"] == []
    assert "нет ответа на этот вопрос" in result["text"]
    assert "нет информации о доставке в Казахстан" in result["text"]
    assert client.get("/tickets?status=escalated").json()[0]["id"] == 1


@pytest.mark.parametrize(
    "reply",
    [
        "Доставка в Казахстан стоит 1000 сом.",  # no citation at all
        "Доставка в Казахстан стоит 1000 сом [7].",  # cites a source that does not exist
        "   ",  # empty reply
    ],
)
def test_reply_without_valid_citation_is_escalated(make_client, reply):
    client = make_kb_client(make_client, FakeLLM(reply))
    create_ticket(client, question="Доставляете ли вы в Казахстан?")

    body = draft(client)
    assert body["ticket"]["status"] == "escalated"
    assert body["draft"]["escalation_reason"] == Reason.NO_VALID_CITATIONS
    assert "1000 сом" not in body["draft"]["text"]  # the ungrounded claim never reaches a draft


def test_empty_knowledge_base_is_escalated(make_client):
    llm = FakeLLM("не должен вызываться [1]")
    client = make_client(llm)
    create_ticket(client)

    body = draft(client)
    assert body["draft"]["escalation_reason"] == Reason.NO_RELEVANT_DOCUMENTS
    assert llm.calls == []


def test_nothing_above_min_score_is_escalated(make_client):
    client = make_kb_client(make_client, FakeLLM("ответ [1]"), min_score=0.99)
    create_ticket(client, question="Какой у вас график работы офиса?")
    assert draft(client)["draft"]["escalation_reason"] == Reason.NO_RELEVANT_DOCUMENTS


@pytest.mark.parametrize(
    "error, notice",
    [(LLMError("network", "timeout"), "network"), (LLMError("auth", "bad key"), "auth"),
     (RuntimeError("SDK surprise"), "api")],
)
def test_llm_failure_escalates_with_retrieved_fragments(make_client, error, notice):
    client = make_kb_client(make_client, FakeLLM(error=error))
    create_ticket(client)

    resp = client.post("/tickets/1/draft")
    assert resp.status_code == 201  # a failing LLM is an escalation, not a server error
    body = resp.json()
    assert body["ticket"]["status"] == "escalated"
    result = body["draft"]
    assert result["escalation_reason"] == Reason.LLM_UNAVAILABLE
    assert f"LLM недоступна ({notice})" in result["text"]
    # retrieval-only results are attached so the operator can answer by hand
    assert result["citations"][0]["file"] == "delivery.txt"


def test_no_llm_configured_escalates(make_client):
    client = make_kb_client(make_client, llm=None)
    create_ticket(client)
    result = draft(client)["draft"]
    assert result["escalation_reason"] == Reason.LLM_UNAVAILABLE
    assert "llm_disabled" in result["text"]


def test_embedding_failure_escalates(make_client, monkeypatch):
    client = make_kb_client(make_client, FakeLLM("ответ [1]"))
    create_ticket(client)

    def broken(text):
        raise RuntimeError("model crashed")

    monkeypatch.setattr(client.app.state.rag.embedder, "embed_query", broken)
    result = draft(client)["draft"]
    assert result["escalation_reason"] == Reason.EMBEDDING_UNAVAILABLE
    assert result["citations"] == []


# --- lifecycle ----------------------------------------------------------------


def test_escalated_ticket_can_be_drafted_again_then_resolved(make_client):
    llm = FakeLLM(error=LLMError("network", "timeout"))
    client = make_kb_client(make_client, llm)
    create_ticket(client)
    assert draft(client)["ticket"]["status"] == "escalated"
    assert client.post("/tickets/1/resolve").status_code == 409

    llm.error, llm.reply = None, "Доставка в Ош стоит 300 сом [1]."
    assert draft(client)["ticket"]["status"] == "drafted"

    resolved = client.post("/tickets/1/resolve").json()
    assert resolved["status"] == "resolved"
    assert [d["confidence"] for d in resolved["drafts"]] == ["no_answer", "answered"]


def test_resolved_ticket_cannot_be_drafted(make_client):
    client = make_kb_client(make_client, FakeLLM("Доставка в Ош стоит 300 сом [1]."))
    create_ticket(client)
    draft(client)
    client.post("/tickets/1/resolve")

    resp = client.post("/tickets/1/draft")
    assert resp.status_code == 409
    assert resp.json()["detail"]["code"] == "already_resolved"
    assert len(client.get("/tickets/1").json()["drafts"]) == 1


# --- pure helpers -------------------------------------------------------------


def test_renumber_citations():
    text = "Первое [3]. Второе [1][3]. Лишнее [2]."
    assert renumber_citations(text, {3: 1, 1: 2}) == "Первое [1]. Второе [2][1]. Лишнее ."


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Доставка 350 сом [1][2].", "Доставка 350 сом [1]."),
        ("Срок [2][3][1]. Цена [1][1].", "Срок [1][2]. Цена [1]."),
        ("Отдельно [1] и [2].", "Отдельно [1] и [1]."),  # separate sentences keep their marker
    ],
)
def test_renumber_citations_merges_repeats(text, expected):
    assert renumber_citations(text, {1: 1, 2: 1, 3: 2}) == expected


def test_chunks_of_one_file_and_page_share_a_source_number():
    results = [
        _result(0, "delivery.txt", text="Ош — 350 сом."),
        _result(1, "delivery.txt", text="Ош — 2–3 рабочих дня."),
        _result(2, "returns.txt", text="Возврат 14 дней."),
        _result(3, "manual.pdf", page=1, text="Страница один."),
        _result(4, "manual.pdf", page=2, text="Страница два."),
    ]
    answer = "Доставка 350 сом, 2–3 дня [1][2]. Инструкция [4]. Ещё [5]. Возврат [3][2]."
    result = AskResult("rag", answer, True, [results[i] for i in (0, 1, 3, 4, 2)], results)

    outcome = outcome_from_answer(result, "Айгуль", "Сколько стоит доставка?")
    assert outcome.status is TicketStatus.drafted
    assert "Доставка 350 сом, 2–3 дня [1]. Инструкция [2]. Ещё [3]. Возврат [4][1]." in outcome.text
    assert outcome.citations == [
        {"file": "delivery.txt", "page": None, "fragment": "Ош — 350 сом. … Ош — 2–3 рабочих дня."},
        {"file": "manual.pdf", "page": 1, "fragment": "Страница один."},
        {"file": "manual.pdf", "page": 2, "fragment": "Страница два."},  # other page: own number
        {"file": "returns.txt", "page": None, "fragment": "Возврат 14 дней."},
    ]
    assert outcome.text.endswith(
        "Источники:\n[1] delivery.txt\n[2] manual.pdf, стр. 1\n[3] manual.pdf, стр. 2\n"
        "[4] returns.txt"
    )


def test_two_chunks_of_one_file_via_api(make_client):
    long_delivery = (
        "Доставка в Ош стоит 350 сом. " * 12 + "\n\n" + "Срок доставки в Ош 2–3 рабочих дня. " * 12
    )
    client = make_client(FakeLLM("Доставка в Ош стоит 350 сом и занимает 2–3 дня [1][2]."))
    assert upload(client, "delivery.txt", long_delivery).json()["chunks"] >= 2
    create_ticket(client)

    result = draft(client)["draft"]
    assert [c["file"] for c in result["citations"]] == ["delivery.txt"]
    assert "занимает 2–3 дня [1]." in result["text"]
    assert result["text"].endswith("Источники:\n[1] delivery.txt")


def test_merged_fragment_has_no_double_ellipsis():
    long_text = "слово " * 100
    group = [_result(0, "a.txt", text=long_text), _result(1, "a.txt", text="второй фрагмент")]
    fragment = citation_dict(group)["fragment"]
    assert "… …" not in fragment
    assert fragment.endswith("слово … второй фрагмент")


def test_escalation_fragments_are_grouped_by_source():
    results = [_result(0, "a.txt", text="один"), _result(1, "a.txt", text="два"),
               _result(2, "b.txt", text="три")]
    outcome = outcome_from_answer(
        AskResult("retrieval_only", None, True, results=results, notice="network"), "Имя", "?"
    )
    assert outcome.escalation_reason == Reason.LLM_UNAVAILABLE
    assert [(c["file"], c["fragment"]) for c in outcome.citations] == [
        ("a.txt", "один … два"), ("b.txt", "три")
    ]


def test_compose_reply_shows_pdf_pages_and_defaults_to_russian():
    citations = [
        {"file": "warranty.pdf", "page": 3, "fragment": "..."},
        {"file": "faq.txt", "page": None, "fragment": "..."},
    ]
    text = compose_reply("Бакыт", "12345?", "Гарантия 12 месяцев [1][2].", citations)
    assert text.startswith("Здравствуйте, Бакыт!")  # no letters: fall back to Russian
    assert text.endswith("Источники:\n[1] warranty.pdf, стр. 3\n[2] faq.txt")


def test_long_fragments_are_shortened():
    long_text = "слово " * 200
    result = AskResult("rag", "Ответ [1].", True, [_result(0, "a.txt", text=long_text)],
                       [_result(0, "a.txt", text=long_text)])
    outcome = outcome_from_answer(result, "Имя", "Вопрос?")
    assert outcome.status is TicketStatus.drafted
    assert outcome.confidence is Confidence.answered
    fragment = outcome.citations[0]["fragment"]
    assert len(fragment) <= 300 and fragment.endswith("…")


def test_found_answer_without_citations_is_escalated():
    # Guard in case the RAG service ever reports a found answer without citations.
    result = AskResult("rag", "Ответ без ссылок.", True, [], [_result(0, "a.txt")])
    outcome = outcome_from_answer(result, "Имя", "Вопрос?")
    assert outcome.status is TicketStatus.escalated
    assert outcome.escalation_reason == Reason.NO_VALID_CITATIONS
