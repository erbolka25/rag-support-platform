import json
import re
import sys
import zlib
from pathlib import Path

import numpy as np
import pytest
from rag_core.generation import BaseLLMClient
from rag_core.retrieval import BaseEmbedder, create_index
from rag_core.retrieval.embeddings import l2_normalize
from rag_core.service import RagService

EVAL_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EVAL_DIR))

import run_eval


class HashingEmbedder(BaseEmbedder):
    """Deterministic bag-of-words embedder: no model download, no network."""

    def __init__(self, dim: int = 512) -> None:
        self.dim = dim

    def _vector(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        for word in re.findall(r"\w+", text.lower()):
            vec[zlib.crc32(word.encode()) % self.dim] += 1.0
        return vec

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return l2_normalize(np.stack([self._vector(t) for t in texts]))

    def embed_query(self, text: str) -> np.ndarray:
        return l2_normalize(self._vector(text))


class EchoLLM(BaseLLMClient):
    """Answers with the first sentence of source [1], or NO_ANSWER when asked to."""

    provider = "fake"
    model = "fake-model"

    def complete(self, system: str, user: str) -> str:
        if "Reference answer" in user:
            return "YES"
        if "вакансии" in user:
            return "NO_ANSWER: нет информации."
        passage = user.split('">', 1)[1].split("</source>", 1)[0].strip()
        return passage.split(".")[0] + ". [1]"


@pytest.mark.parametrize(
    ("fact", "answer", "expected"),
    [
        ("150", "Доставка стоит 150 сомов.", True),
        ("2", "Стоимость 250 сомов.", False),
        ("50000", "You need 50,000 som a year.", True),
        ("30000", "до 30 000 сомов", True),
        ("тот же день", "Привезём в тот же день.", True),
        ("Элкарт", "картой элкарт", True),
        ("20", "open daily from 10:00 AM to 8:00 PM", True),
        ("10", "open daily from 10:00 AM to 8:00 PM", True),
        ("20", "from 10 a.m. to 8 p.m.", True),
        ("22", "support works 8am–10pm", True),
        ("0", "from 12 AM", True),
        ("12", "closes at 12:30 PM", True),
        ("20", "open from 10:00 to 18:00", False),
    ],
)
def test_fact_present(fact, answer, expected):
    assert run_eval.fact_present(fact, answer) is expected


def test_questions_file_is_consistent():
    questions = run_eval.load_questions(EVAL_DIR / "questions.jsonl")
    kb = {p.name for p in (EVAL_DIR / "dataset" / "kb").iterdir()}
    assert 40 <= len(questions) <= 50
    assert len({q.id for q in questions}) == len(questions)
    for q in questions:
        if q.answerable:
            assert q.expected_source in kb and q.key_facts, q.id
        else:
            assert q.expected_source is None and not q.key_facts, q.id
    assert {q.lang for q in questions} == {"ru", "ky", "en"}


def make_row(**kw):
    base = {"id": "x", "lang": "ru", "answerable": True, "retrieved": [], "top_score": 0.9,
            "hit_at_k": True, "reciprocal_rank": 1.0, "refused": False, "latency_ms": 10.0}
    return run_eval.Row(**(base | kw))


def test_compute_metrics_full_mode():
    rows = [
        make_row(answer="a [1]", facts_ok=True, citation_statuses=["ok"]),
        make_row(answer="b [1]", facts_ok=False, citation_statuses=["unsupported"], hit_at_k=False,
                 reciprocal_rank=0.0),
        make_row(refused=True, facts_ok=False, latency_ms=30.0),
        make_row(answerable=False, hit_at_k=None, reciprocal_rank=None, refused=True),
        make_row(answerable=False, hit_at_k=None, reciprocal_rank=None, answer="c [1]",
                 citation_statuses=["ok"]),
    ]
    m = run_eval.compute_metrics(rows, use_llm=True, llm_judge=False)
    assert m["recall_at_k"] == pytest.approx(2 / 3, abs=1e-3)
    assert m["answer_accuracy"] == pytest.approx(1 / 3, abs=1e-3)
    assert m["citation_validity"] == pytest.approx(2 / 3, abs=1e-3)
    assert m["no_answer_precision"] == 0.5  # 2 refusals, one of them wrong
    assert m["no_answer_recall"] == 0.5
    assert m["hallucination_rate"] == 0.5
    assert m["false_refusal_rate"] == pytest.approx(1 / 3, abs=1e-3)


def test_retrieval_mode_renames_refusal_metrics():
    m = run_eval.compute_metrics([make_row(), make_row(answerable=False, refused=True)], False, False)
    assert "answer_accuracy" not in m and "citation_validity" not in m
    assert m["retrieval_gate_no_answer_recall"] == 1.0


@pytest.mark.parametrize("llm", [None, EchoLLM()])
def test_main_end_to_end(tmp_path, monkeypatch, llm):
    def fake_build_service(settings):
        index = create_index("numpy")
        return RagService(settings, HashingEmbedder(), index, llm)

    monkeypatch.setattr(run_eval, "build_service", fake_build_service)
    args = ["--out-dir", str(tmp_path), "--report", str(tmp_path / "REPORT.md"),
            "--env-file", str(tmp_path / "missing.env"), "--min-score", "0.1", "--delay", "0"]
    if llm:
        args.append("--llm-judge")
    run_eval.main(args)

    [result] = list(tmp_path.glob("*.json"))
    report = json.loads(result.read_text(encoding="utf-8"))
    assert report["config"]["mode"] == ("full" if llm else "retrieval")
    assert len(report["rows"]) == report["config"]["questions"]
    assert 0 <= report["metrics"]["recall_at_k"] <= 1
    if llm:
        assert report["metrics"]["answer_accuracy_llm_judge"] is not None
    assert "| Metric | Value |" in (tmp_path / "REPORT.md").read_text(encoding="utf-8")


class FailingLLM(BaseLLMClient):
    provider = "fake"
    model = "fake-model"

    def complete(self, system: str, user: str) -> str:
        from rag_core.generation import LLMError

        raise LLMError("quota", "no credits")


def test_llm_errors_are_not_counted_as_answers():
    rows = [make_row(llm_error="quota"), make_row(answerable=False, hit_at_k=None,
                                                  reciprocal_rank=None, llm_error="quota"),
            make_row(answer="a [1]", facts_ok=True, citation_statuses=["ok"])]
    m = run_eval.compute_metrics(rows, use_llm=True, llm_judge=False)
    assert m["llm_errors"] == 2
    assert m["answer_accuracy"] == 1.0
    assert m["hallucination_rate"] is None  # no unanswerable question got a real LLM verdict
    assert m["recall_at_k"] == 1.0  # retrieval metrics still use every row


def test_main_stops_when_the_llm_keeps_failing(tmp_path, monkeypatch):
    monkeypatch.setattr(run_eval, "build_service", lambda s: RagService(
        s, HashingEmbedder(), create_index("numpy"), FailingLLM()))
    with pytest.raises(SystemExit, match="failed on the first 3"):
        run_eval.main(["--out-dir", str(tmp_path), "--report", str(tmp_path / "R.md"),
                       "--env-file", str(tmp_path / "x.env"), "--min-score", "0.1", "--delay", "0"])
    assert not list(tmp_path.glob("*.json"))


class FlakyLLM(BaseLLMClient):
    """Rate-limited on the first ``failures`` calls, then answers from source [1]."""

    provider = "openai"
    model = "llama-test"
    base_url = "https://api.groq.com/openai/v1"

    def __init__(self, failures: int) -> None:
        self.failures = failures
        self.calls = 0

    def complete(self, system: str, user: str) -> str:
        from rag_core.generation import LLMError

        self.calls += 1
        if self.calls <= self.failures:
            raise LLMError("rate_limit", "429")
        return EchoLLM().complete(system, user)


def _service(llm, tmp_path):
    from rag_core.config import Settings

    settings = Settings(_env_file=None, data_dir=tmp_path, min_score=0.1)
    service = RagService(settings, HashingEmbedder(), create_index("numpy"), llm)
    run_eval.index_kb(service, EVAL_DIR / "dataset" / "kb")
    return service


def test_pacer_waits_before_each_call_and_backs_off_exponentially(tmp_path):
    slept = []
    pacer = run_eval.Pacer(delay_s=2, max_retries=4, backoff_s=5, sleep=slept.append)
    llm = FlakyLLM(failures=2)
    q = run_eval.load_questions(EVAL_DIR / "questions.jsonl")[0]
    row = run_eval.evaluate_question(_service(llm, tmp_path), q, 5, True, False, pacer)
    assert row.llm_error is None and not row.refused
    assert slept == [2, 5, 2, 10, 2]  # delay, backoff, delay, longer backoff, delay
    assert pacer.retries == 2 and llm.calls == 3


def test_rate_limit_that_never_clears_is_reported_not_retried_forever(tmp_path):
    pacer = run_eval.Pacer(delay_s=0, max_retries=2, sleep=lambda s: None)
    q = run_eval.load_questions(EVAL_DIR / "questions.jsonl")[0]
    row = run_eval.evaluate_question(_service(FlakyLLM(failures=99), tmp_path), q, 5, True, False, pacer)
    assert row.llm_error == "rate_limit" and pacer.retries == 2


def test_judge_retries_rate_limits(tmp_path):
    pacer = run_eval.Pacer(delay_s=0, sleep=lambda s: None)
    service = _service(FlakyLLM(failures=1), tmp_path)
    q = run_eval.load_questions(EVAL_DIR / "questions.jsonl")[0]
    assert run_eval.judge(service, q, "150 сомов [1]", pacer) is True
    assert pacer.retries == 1


def test_llm_label_names_model_and_host(tmp_path):
    assert run_eval.llm_label(_service(FlakyLLM(0), tmp_path)) == "llama-test @ api.groq.com"
    assert run_eval.llm_label(_service(EchoLLM(), tmp_path)) == "fake-model @ fake"


def test_rescore_changes_only_the_fact_check():
    baseline = {
        "config": {"llm_judge": False},
        "rows": [
            {**vars(make_row(id="a34", answer="Open 10:00 AM to 8:00 PM [1]", facts_ok=False,
                             missing_facts=["20"], citation_statuses=["ok"])),
             "question": "hours?"},
            {**vars(make_row(id="a04", refused=True, facts_ok=False)), "question": "q"},
        ],
    }
    questions = [
        run_eval.Question("a34", "en", "hours?", "10 to 20", "x.txt", ["10", "20"]),
        run_eval.Question("a04", "ru", "q", "a", "x.txt", ["350"]),
    ]
    m = run_eval.rescore(baseline, questions)
    assert m["answer_accuracy"] == 0.5  # a34 now passes; the refusal stays wrong
    assert m["false_refusal_rate"] == 0.5


def test_fmt_delta():
    assert run_eval.fmt_delta("answer_accuracy", 0.75, 0.722) == "+2.8 pp"
    assert run_eval.fmt_delta("latency_p50_ms", 800.0, 832.0) == "-32 ms"
    assert run_eval.fmt_delta("mrr", 0.9, 0.9) == "+0.000"
    assert run_eval.fmt_delta("llm_errors", 0, 2) == "-2"
    assert run_eval.fmt_delta("citation_validity", None, 1.0) == "—"


def test_render_rebuilds_report_with_baseline_comparison(tmp_path, monkeypatch):
    monkeypatch.setattr(run_eval, "build_service", lambda s: RagService(
        s, HashingEmbedder(), create_index("numpy"), EchoLLM()))
    common = ["--out-dir", str(tmp_path), "--env-file", str(tmp_path / "x.env"),
              "--min-score", "0.1", "--delay", "0"]
    run_eval.main([*common, "--report", str(tmp_path / "first.md")])
    [result] = list(tmp_path.glob("*.json"))
    notes = tmp_path / "notes.md"
    notes.write_text("Fix A changed X.", encoding="utf-8")
    run_eval.main(["--render", str(result), "--baseline", str(result), "--notes", str(notes),
                   "--report", str(tmp_path / "R.md")])
    report = (tmp_path / "R.md").read_text(encoding="utf-8")
    assert "## Baseline → this run" in report and "Fix A changed X." in report
    assert "| Answer accuracy (key facts) |" in report
