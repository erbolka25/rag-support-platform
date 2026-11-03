"""End-to-end quality evaluation of rag_core on eval/dataset.

Usage (from the repository root):
    python eval/run_eval.py                       # full RAG if an LLM key is set, else retrieval-only
    python eval/run_eval.py --mode retrieval      # no API key needed: retrieval metrics only
    python eval/run_eval.py --mode full --llm-judge --env-file packages/rag_core/.env
    python eval/run_eval.py --mode full --delay 7   # free tiers: stay under the per-minute limit

Every run indexes eval/dataset/kb into a temporary directory, asks every question in
eval/questions.jsonl, writes eval/results/<date>.json and regenerates eval/REPORT.md.

Metrics
    Recall@5            expected_source is among the top-5 retrieved files (ranking only,
                        before the min_score threshold); answerable questions only.
    MRR                 reciprocal rank of the first chunk from expected_source.
    Answer accuracy     every key fact of the question appears in the answer (numbers match
                        as whole numbers); refusals count as wrong. --llm-judge adds a second,
                        model-graded accuracy.
    Citation validity   share of answers whose every [n] marker passes rag_core's lexical
                        check_citations ("ok"); markers in a different language than their
                        passage cannot be checked and are reported separately.
    NO_ANSWER P / R     precision and recall of refusals against expected NO_ANSWER.
    Hallucination rate  share of NO_ANSWER questions that got an answer anyway.
    Latency p50 / p95   wall time per question (search only, or the full ask()); pauses and
                        rate-limit backoff are not included.

Free LLM tiers allow only a few requests per minute: full mode pauses --delay seconds before
every LLM call and retries a rate-limited call with exponential backoff (--max-retries).

In retrieval mode "refused" means no chunk cleared min_score, so the refusal metrics there
describe the retrieval gate, not the LLM.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
from rag_core.config import Settings
from rag_core.generation import LLMError, check_citations
from rag_core.service import RagService, build_service

EVAL_DIR = Path(__file__).resolve().parent
NO_ANSWER = "NO_ANSWER"

JUDGE_SYSTEM = (
    "You grade answers of a customer-support assistant. Reply with exactly one word: "
    "YES if the candidate answer states the same facts as the reference answer (wording and "
    "language may differ, extra correct detail is fine), otherwise NO."
)


@dataclass
class Question:
    id: str
    lang: str
    question: str
    expected_answer: str
    expected_source: str | None
    key_facts: list[str] = field(default_factory=list)

    @property
    def answerable(self) -> bool:
        return self.expected_answer != NO_ANSWER


@dataclass
class Row:
    id: str
    lang: str
    answerable: bool
    retrieved: list[str]
    top_score: float | None
    hit_at_k: bool | None
    reciprocal_rank: float | None
    refused: bool
    latency_ms: float
    answer: str | None = None
    facts_ok: bool | None = None
    missing_facts: list[str] = field(default_factory=list)
    judge_ok: bool | None = None
    citation_statuses: list[str] = field(default_factory=list)
    notice: str | None = None
    # Full mode only: the LLM call failed (rag_core fell back to retrieval-only). Such rows
    # say nothing about answer quality, so they are left out of the answer metrics.
    llm_error: str | None = None


def load_questions(path: Path) -> list[Question]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [Question(**json.loads(line)) for line in lines if line.strip()]


# --- answer checks ----------------------------------------------------------------

_DIGIT_SEPARATOR = re.compile(r"(?<=\d)[\s  ,](?=\d{3}\b)")


_TWELVE_HOUR = re.compile(r"\b(\d{1,2})(?::(\d{2}))?\s*([ap])\.?\s*m\b\.?", re.IGNORECASE)


def to_24_hour(text: str) -> str:
    """Rewrite "8:00 PM" / "8 p.m." as "20:00", so key facts in 24-hour form still match."""

    def convert(match: re.Match) -> str:
        hour = int(match.group(1)) % 12 + (12 if match.group(3).lower() == "p" else 0)
        return f"{hour}:{match.group(2) or '00'}"

    return _TWELVE_HOUR.sub(convert, text)


def normalize(text: str) -> str:
    return _DIGIT_SEPARATOR.sub("", to_24_hour(text).lower().replace("ё", "е"))


def fact_present(fact: str, answer: str) -> bool:
    """Numbers match as whole numbers (so "2" is not found in "250"); words as substrings."""
    fact, answer = normalize(fact), normalize(answer)
    if fact.isdigit():
        return re.search(rf"(?<!\d){fact}(?!\d)", answer) is not None
    return fact in answer


@dataclass
class Pacer:
    """Spaces out LLM calls and backs off on rate limits, to fit free-tier quotas."""

    delay_s: float = 2.0
    max_retries: int = 4
    backoff_s: float = 5.0
    sleep: Callable[[float], None] = time.sleep
    retries: int = 0  # rate-limited calls retried during the run

    def before_call(self) -> None:
        if self.delay_s > 0:
            self.sleep(self.delay_s)

    def backoff(self, attempt: int, what: str) -> None:
        wait = self.backoff_s * 2**attempt
        self.retries += 1
        print(f"  rate limited on {what}; retrying in {wait:.0f} s", file=sys.stderr)
        self.sleep(wait)


def judge(service: RagService, q: Question, answer: str, pacer: Pacer) -> bool | None:
    user = f"Question: {q.question}\nReference answer: {q.expected_answer}\nCandidate answer: {answer}"
    for attempt in range(pacer.max_retries + 1):
        pacer.before_call()
        try:
            verdict = service.llm.complete(JUDGE_SYSTEM, user)
        except LLMError as exc:
            if exc.code == "rate_limit" and attempt < pacer.max_retries:
                pacer.backoff(attempt, f"judge {q.id}")
                continue
            print(f"  judge failed on {q.id}: {exc.code}", file=sys.stderr)
            return None
        except Exception as exc:  # noqa: BLE001 - a failed grade is missing data, not a wrong answer
            print(f"  judge failed on {q.id}: {exc}", file=sys.stderr)
            return None
        return verdict.strip().upper().startswith("YES")
    return None


# --- running ----------------------------------------------------------------------


def build(args: argparse.Namespace, data_dir: Path) -> RagService:
    overrides = {"data_dir": data_dir, "top_k": args.top_k}
    if args.min_score is not None:
        overrides["min_score"] = args.min_score
    if args.mode == "retrieval":
        overrides["llm_provider"] = "none"
    settings = Settings(_env_file=args.env_file, **overrides)
    service = build_service(settings)
    if args.mode == "full" and service.llm is None:
        sys.exit(
            "--mode full needs an LLM key: set LLM_PROVIDER and OPENAI_API_KEY/ANTHROPIC_API_KEY "
            "in the environment or pass --env-file."
        )
    return service


def index_kb(service: RagService, kb_dir: Path) -> int:
    files = sorted(p for p in kb_dir.iterdir() if p.suffix in {".txt", ".md", ".pdf"})
    for path in files:
        service.add_document(path.read_bytes(), path.name)
    return len(files)


def evaluate_question(
    service: RagService, q: Question, k: int, use_llm: bool, llm_judge: bool, pacer: Pacer | None = None
) -> Row:
    pacer = pacer or Pacer(delay_s=0)
    # Ranking without the threshold, for Recall@k and MRR.
    ranked = service.index.search(service.embedder.embed_query(q.question), k)
    retrieved = [r.chunk.filename for r in ranked]
    hit = rr = None
    if q.answerable:
        hit = q.expected_source in retrieved
        rr = next((1 / (i + 1) for i, f in enumerate(retrieved) if f == q.expected_source), 0.0)

    if use_llm:
        for attempt in range(pacer.max_retries + 1):
            pacer.before_call()
            start = time.perf_counter()
            result = service.ask(q.question, k)
            latency_ms = (time.perf_counter() - start) * 1000
            # ask() turns LLM failures into a retrieval-only result with the error as notice.
            if result.notice != "rate_limit" or attempt == pacer.max_retries:
                break
            pacer.backoff(attempt, q.id)
        refused, answer, notice = not result.answer_found, result.answer, result.notice
        passages = result.results
        llm_error = notice if result.mode != "rag" else None
    else:
        start = time.perf_counter()
        passages = service.search(q.question, k)
        latency_ms = (time.perf_counter() - start) * 1000
        refused, answer, notice, llm_error = not passages, None, None, None

    row = Row(
        id=q.id,
        lang=q.lang,
        answerable=q.answerable,
        retrieved=retrieved,
        top_score=round(ranked[0].score, 4) if ranked else None,
        hit_at_k=hit,
        reciprocal_rank=rr,
        refused=refused,
        latency_ms=round(latency_ms, 1),
        answer=answer,
        notice=notice,
        llm_error=llm_error,
    )
    if llm_error:
        return row
    if use_llm and not refused and answer:
        row.citation_statuses = [c.status for c in check_citations(answer, passages)]
        if q.answerable:
            row.missing_facts = [f for f in q.key_facts if not fact_present(f, answer)]
            row.facts_ok = not row.missing_facts
            if llm_judge:
                row.judge_ok = judge(service, q, answer, pacer)
    elif q.answerable and use_llm:
        row.facts_ok = False
        row.judge_ok = False if llm_judge else None
    return row


# --- metrics ----------------------------------------------------------------------


def ratio(num: int, den: int) -> float | None:
    return round(num / den, 3) if den else None


def compute_metrics(rows: list[Row], use_llm: bool, llm_judge: bool) -> dict:
    all_answerable = [r for r in rows if r.answerable]  # retrieval metrics use every row
    llm_errors = sum(bool(r.llm_error) for r in rows)
    rows = [r for r in rows if not r.llm_error]
    answerable = [r for r in rows if r.answerable]
    unanswerable = [r for r in rows if not r.answerable]
    refused = [r for r in rows if r.refused]
    latencies = [r.latency_ms for r in rows] or [0.0]
    m: dict = {
        "recall_at_k": ratio(sum(bool(r.hit_at_k) for r in all_answerable), len(all_answerable)),
        "mrr": round(float(np.mean([r.reciprocal_rank for r in all_answerable])), 3),
        "no_answer_precision": ratio(sum(not r.answerable for r in refused), len(refused)),
        "no_answer_recall": ratio(sum(r.refused for r in unanswerable), len(unanswerable)),
        "false_refusal_rate": ratio(sum(r.refused for r in answerable), len(answerable)),
        "hallucination_rate": ratio(sum(not r.refused for r in unanswerable), len(unanswerable)),
        "latency_p50_ms": round(float(np.percentile(latencies, 50)), 1),
        "latency_p95_ms": round(float(np.percentile(latencies, 95)), 1),
    }
    if use_llm:
        m["answer_accuracy"] = ratio(sum(bool(r.facts_ok) for r in answerable), len(answerable))
        if llm_judge:
            graded = [r for r in answerable if r.judge_ok is not None]
            m["answer_accuracy_llm_judge"] = ratio(sum(r.judge_ok for r in graded), len(graded))
        cited = [r for r in rows if r.citation_statuses]
        answered = [r for r in rows if not r.refused and r.answer]
        m["citation_validity"] = ratio(
            sum(all(s == "ok" for s in r.citation_statuses) for r in cited), len(answered)
        )
        m["citation_unverifiable_share"] = ratio(
            sum("cross_language" in r.citation_statuses for r in cited), len(answered)
        )
        m["llm_errors"] = llm_errors
    else:
        # In retrieval mode "refused" is the min_score gate, not the model's NO_ANSWER.
        m = {f"retrieval_gate_{k}" if k.startswith(("no_answer", "false_ref", "halluc")) else k: v
             for k, v in m.items()}
    return m


def metrics_by_lang(rows: list[Row], use_llm: bool) -> dict:
    out = {}
    for lang in sorted({r.lang for r in rows}):
        sub = [r for r in rows if r.lang == lang]
        ans = [r for r in sub if r.answerable]
        una = [r for r in sub if not r.answerable]
        out[lang] = {
            "questions": len(sub),
            "recall_at_k": ratio(sum(bool(r.hit_at_k) for r in ans), len(ans)),
            "refused_answerable": sum(r.refused for r in ans),
            "answered_unanswerable": sum(not r.refused for r in una),
        }
        if use_llm:
            out[lang]["answer_accuracy"] = ratio(sum(bool(r.facts_ok) for r in ans), len(ans))
    return out


# --- output -----------------------------------------------------------------------

LABELS = {
    "recall_at_k": "Retrieval Recall@{k}",
    "mrr": "Retrieval MRR",
    "answer_accuracy": "Answer accuracy (key facts)",
    "answer_accuracy_llm_judge": "Answer accuracy (LLM judge)",
    "citation_validity": "Citation validity",
    "citation_unverifiable_share": "  answers with cross-language citations (unchecked)",
    "no_answer_precision": "NO_ANSWER precision",
    "no_answer_recall": "NO_ANSWER recall",
    "false_refusal_rate": "False refusal rate",
    "hallucination_rate": "Hallucination rate",
    "retrieval_gate_no_answer_precision": "Retrieval gate: refusal precision",
    "retrieval_gate_no_answer_recall": "Retrieval gate: refusal recall",
    "retrieval_gate_false_refusal_rate": "Retrieval gate: false refusal rate",
    "retrieval_gate_hallucination_rate": "Retrieval gate: passes on unanswerable",
    "latency_p50_ms": "Latency p50, ms",
    "latency_p95_ms": "Latency p95, ms",
    "llm_errors": "LLM errors (left out of answer metrics)",
}


def fmt(key: str, value) -> str:
    if value is None:
        return "—"
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    return f"{value:.0f}" if key.startswith("latency") else f"{value:.1%}" if key != "mrr" else f"{value:.3f}"


def metric_table(metrics: dict, k: int) -> list[tuple[str, str]]:
    return [(LABELS.get(key, key).format(k=k), fmt(key, v)) for key, v in metrics.items()]


def print_table(report: dict) -> None:
    rows = metric_table(report["metrics"], report["config"]["top_k"])
    width = max(len(name) for name, _ in rows)
    print(f"\nrag_core eval — mode={report['config']['mode']}, "
          f"{report['config']['questions']} questions, {report['config']['documents']} documents")
    print("-" * (width + 12))
    for name, value in rows:
        print(f"{name:<{width}}  {value:>8}")
    print("-" * (width + 12))


def display_path(path: Path) -> str:
    """Repository-relative when possible, so REPORT.md holds no machine-specific paths."""
    try:
        return path.resolve().relative_to(EVAL_DIR.parent).as_posix()
    except ValueError:
        return path.as_posix()


def write_report(report: dict, path: Path, result_file: Path) -> None:
    cfg, rows = report["config"], report["rows"]
    lines = [
        "# Evaluation report",
        "",
        f"Generated by `eval/run_eval.py` on {cfg['date']} — mode **{cfg['mode']}**, "
        f"{cfg['questions']} questions ({cfg['answerable']} answerable, "
        f"{cfg['questions'] - cfg['answerable']} NO_ANSWER), {cfg['documents']} documents, "
        f"embedding `{cfg['embedding_model']}`, top_k={cfg['top_k']}, min_score={cfg['min_score']}"
        + (f", LLM **`{cfg['llm']}`** (delay {cfg['delay_s']:g} s, "
           f"{cfg['rate_limit_retries']} rate-limit retries)" if cfg["llm"] else ", no LLM")
        + f". Raw results: `{display_path(result_file)}`.",
        "",
        "| Metric | Value |",
        "|---|---|",
        *[f"| {name.strip()} | {value} |" for name, value in metric_table(report["metrics"], cfg["top_k"])],
        "",
        "## By language",
        "",
        "| Lang | Questions | Recall@k | Refused answerable | Answered NO_ANSWER |"
        + (" Accuracy |" if "answer_accuracy" in report["metrics"] else ""),
        "|---|---|---|---|---|" + ("---|" if "answer_accuracy" in report["metrics"] else ""),
    ]
    for lang, m in report["by_language"].items():
        line = (f"| {lang} | {m['questions']} | {fmt('r', m['recall_at_k'])} | "
                f"{m['refused_answerable']} | {m['answered_unanswerable']} |")
        if "answer_accuracy" in m:
            line += f" {fmt('a', m['answer_accuracy'])} |"
        lines.append(line)

    failures = []
    for r in rows:
        problems = []
        if r["llm_error"]:
            problems.append(f"LLM error: {r['llm_error']}")
        if r["answerable"] and not r["hit_at_k"]:
            problems.append(f"source not in top-{cfg['top_k']} ({', '.join(dict.fromkeys(r['retrieved']))})")
        if r["answerable"] and r["refused"] and not r["llm_error"]:
            problems.append("refused although answerable" + (f" ({r['notice']})" if r["notice"] else ""))
        if not r["answerable"] and not r["refused"] and not r["llm_error"]:
            problems.append("answered a NO_ANSWER question")
        if r["missing_facts"]:
            problems.append(f"missing facts {r['missing_facts']}")
        bad = [s for s in r["citation_statuses"] if s not in ("ok", "cross_language")]
        if bad:
            problems.append(f"citations {bad}")
        if problems:
            failures.append(f"| {r['id']} | {r['lang']} | {r['question']} | {'; '.join(problems)} |")
    lines += ["", "## Failures", ""]
    if failures:
        lines += ["| ID | Lang | Question | Problem |", "|---|---|---|---|", *failures]
    else:
        lines.append("None.")
    lines += [
        "",
        "## How to read this",
        "",
        ("- Answer accuracy is a strict lexical check: every key fact (mostly numbers) must "
         "appear in the answer. It under-counts correct paraphrases and over-counts answers "
         "that mention the right number in a wrong statement; use `--llm-judge` for a second opinion."),
        ("- Citation validity uses rag_core's own lexical check (a cited sentence must share a "
         "number or content-word stem with the passage). It cannot verify answers in a "
         "different language from the source (Kyrgyz/English answers over Russian documents)."),
        "- In retrieval mode refusals come from the `min_score` threshold alone.",
        ("- 50 questions over 7 short documents: one question is 2–3 percentage points, so "
         "differences of a few points between runs are noise."),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def llm_label(service: RagService) -> str:
    """Model and where it runs, e.g. "llama-3.3-70b-versatile @ api.groq.com"."""
    base_url = getattr(service.llm, "base_url", None)
    host = urlparse(base_url).netloc if base_url else service.llm.provider
    return f"{service.llm.model} @ {host}"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mode", choices=["auto", "retrieval", "full"], default="auto")
    parser.add_argument("--llm-judge", action="store_true", help="also grade answers with the LLM")
    parser.add_argument("--kb", type=Path, default=EVAL_DIR / "dataset" / "kb")
    parser.add_argument("--questions", type=Path, default=EVAL_DIR / "questions.jsonl")
    parser.add_argument("--env-file", type=Path, default=None,
                        help="settings file with LLM_PROVIDER and the API key (default: environment "
                             "variables and packages/rag_core/.env)")
    parser.add_argument("--delay", type=float, default=2.0,
                        help="seconds to wait before every LLM call (full mode; default 2)")
    parser.add_argument("--max-retries", type=int, default=4,
                        help="retries of a rate-limited LLM call, with backoff 5, 10, 20, 40 s")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--min-score", type=float, default=None)
    parser.add_argument("--out-dir", type=Path, default=EVAL_DIR / "results")
    parser.add_argument("--report", type=Path, default=EVAL_DIR / "REPORT.md")
    args = parser.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):  # Windows consoles default to a legacy code page
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    if args.env_file is None:
        from rag_core.config import ENV_FILE

        args.env_file = ENV_FILE

    questions = load_questions(args.questions)
    with tempfile.TemporaryDirectory(prefix="rag-eval-") as tmp:
        service = build(args, Path(tmp))
        use_llm = service.llm is not None
        if args.llm_judge and not use_llm:
            sys.exit("--llm-judge needs an LLM key.")
        mode = "full" if use_llm else "retrieval"
        documents = index_kb(service, args.kb)
        service.embedder.embed_query("warm-up")  # keep model loading out of latency
        pacer = Pacer(delay_s=args.delay if use_llm else 0, max_retries=args.max_retries)
        rows = []
        for i, q in enumerate(questions, 1):
            row = evaluate_question(service, q, args.top_k, use_llm, args.llm_judge, pacer)
            rows.append(row)
            if len(rows) == 3 and all(r.llm_error for r in rows):
                sys.exit(f"The LLM failed on the first 3 questions ({row.llm_error}); "
                         "fix the key or quota, or run with --mode retrieval.")
            status = "refused" if row.refused else "answered"
            print(f"[{i:>2}/{len(questions)}] {q.id} {q.lang} {status:<8} {row.latency_ms:>7.0f} ms",
                  file=sys.stderr)
        settings = service.settings

    report = {
        "config": {
            "date": datetime.now().isoformat(timespec="seconds"),
            "mode": mode,
            "questions": len(questions),
            "answerable": sum(q.answerable for q in questions),
            "documents": documents,
            "embedding_model": settings.embedding_model,
            "index_backend": settings.index_backend,
            "top_k": args.top_k,
            "min_score": settings.min_score,
            "llm": llm_label(service) if use_llm else None,
            "llm_judge": args.llm_judge,
            "llm_base_url": settings.llm_base_url or None,
            "delay_s": pacer.delay_s,
            "rate_limit_retries": pacer.retries,
        },
        "metrics": compute_metrics(rows, use_llm, args.llm_judge),
        "by_language": metrics_by_lang(rows, use_llm),
        "rows": [asdict(r) | {"question": q.question} for r, q in zip(rows, questions, strict=True)],
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    result_file = args.out_dir / f"{datetime.now():%Y-%m-%d_%H%M%S}_{mode}.json"
    result_file.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(report, args.report, result_file)
    print_table(report)
    print(f"Saved {result_file} and {args.report}")


if __name__ == "__main__":
    main()
