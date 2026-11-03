"""Experiment: dense vs BM25 vs hybrid (RRF) retrieval on the same chunks. No API key needed.

    python eval/compare_retrieval.py

rag_core itself is dense-only; BM25 and the fusion live here, so the comparison changes
nothing in the engine. Two datasets:

    store   eval/questions.jsonl over eval/dataset/kb (RU documents; RU/KY/EN questions)
    hr      packages/rag_core/data/eval/eval_set.json over data/samples (the engine's own set)

A question is a hit when a chunk of its expected file is in the top-k (answerable questions
only). Writes eval/RETRIEVAL_COMPARISON.md.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
from rag_core.config import Settings
from rag_core.ingestion import Chunk, ingest_document
from rag_core.retrieval import SentenceTransformerEmbedder

EVAL_DIR = Path(__file__).resolve().parent
ROOT = EVAL_DIR.parent
RRF_K = 60
_WORD = re.compile(r"\w+")


@dataclass
class Case:
    question: str
    source: str
    lang: str


def load_store() -> tuple[Path, list[Case]]:
    cases = []
    for line in (EVAL_DIR / "questions.jsonl").read_text(encoding="utf-8").splitlines():
        item = json.loads(line)
        if item["expected_source"]:
            cases.append(Case(item["question"], item["expected_source"], item["lang"]))
    return EVAL_DIR / "dataset" / "kb", cases


def load_hr() -> tuple[Path, list[Case]]:
    data = ROOT / "packages" / "rag_core" / "data"
    items = json.loads((data / "eval" / "eval_set.json").read_text(encoding="utf-8"))
    cases = []
    for item in items:
        if item.get("answerable", True) is False:
            continue
        lang = "en" if re.search(r"[a-z]", item["question"], re.IGNORECASE) else "ru"
        cases.append(Case(item["question"], item["relevant"][0]["filename"], lang))
    return data / "samples", cases


DATASETS = {"store": load_store, "hr": load_hr}


def tokens(text: str) -> list[str]:
    """Lower-cased words cut to 5 characters: a crude stemmer for Russian inflection."""
    out = []
    for word in _WORD.findall(text.lower().replace("ё", "е")):
        out.append(word if word.isdigit() or len(word) < 4 else word[:5])
    return out


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75) -> None:
        self.docs = [Counter(d) for d in docs]
        self.lengths = np.array([len(d) for d in docs], dtype=float)
        self.avg = self.lengths.mean()
        df = Counter(t for d in docs for t in set(d))
        n = len(docs)
        self.idf = {t: math.log((n - f + 0.5) / (f + 0.5) + 1) for t, f in df.items()}
        self.k1, self.b = k1, b

    def scores(self, query: list[str]) -> np.ndarray:
        out = np.zeros(len(self.docs))
        norm = self.k1 * (1 - self.b + self.b * self.lengths / self.avg)
        for term in set(query):
            idf = self.idf.get(term)
            if idf is None:
                continue
            tf = np.array([d[term] for d in self.docs], dtype=float)
            out += idf * tf * (self.k1 + 1) / (tf + norm)
        return out


def rrf(*rankings: np.ndarray) -> np.ndarray:
    """Reciprocal rank fusion of full rankings (arrays of chunk positions, best first)."""
    fused = np.zeros(len(rankings[0]))
    for ranking in rankings:
        fused[ranking] += 1 / (RRF_K + np.arange(1, len(ranking) + 1))
    return np.argsort(-fused, kind="stable")


def chunk_kb(kb: Path, settings: Settings) -> list[Chunk]:
    chunks = []
    for path in sorted(kb.iterdir()):
        if path.suffix in {".txt", ".md", ".pdf"}:
            chunks += ingest_document(
                path.read_bytes(),
                path.name,
                max_chars=settings.chunk_max_chars,
                overlap_chars=settings.chunk_overlap_chars,
            ).chunks
    return chunks


def evaluate(name: str, embedder, settings: Settings, k: int) -> list[dict]:
    kb, cases = DATASETS[name]()
    chunks = chunk_kb(kb, settings)
    files = np.array([c.filename for c in chunks])
    doc_vectors = embedder.embed_documents([c.text for c in chunks])
    bm25 = BM25([tokens(c.text) for c in chunks])

    per_method: dict[str, list[tuple[Case, float, bool]]] = {"dense": [], "bm25": [], "hybrid": []}
    for case in cases:
        dense = np.argsort(-(doc_vectors @ embedder.embed_query(case.question)), kind="stable")
        lexical = np.argsort(-bm25.scores(tokens(case.question)), kind="stable")
        for method, ranking in (("dense", dense), ("bm25", lexical), ("hybrid", rrf(dense, lexical))):
            ranked_files = files[ranking]
            hits = np.flatnonzero(ranked_files == case.source)
            rr = 1 / (hits[0] + 1) if len(hits) else 0.0
            per_method[method].append((case, rr, bool(len(hits)) and hits[0] < k))

    rows = []
    for method, results in per_method.items():
        row = {"dataset": name, "method": method, "questions": len(results), "chunks": len(chunks)}
        row["recall"] = float(np.mean([hit for _, _, hit in results]))
        row["mrr"] = float(np.mean([rr for _, rr, _ in results]))
        for lang in sorted({c.lang for c, _, _ in results}):
            sub = [hit for c, _, hit in results if c.lang == lang]
            row[f"recall_{lang}"] = (float(np.mean(sub)), len(sub))
        row["misses"] = [c.question for c, _, hit in results if not hit]
        rows.append(row)
    return rows


def render(rows: list[dict], k: int, model: str) -> str:
    langs = ["ru", "ky", "en"]
    lines = [
        "# Retrieval: dense vs BM25 vs hybrid",
        "",
        (f"Generated by `eval/compare_retrieval.py` on {date.today()}. Embeddings `{model}`, "
         f"BM25 (k1=1.5, b=0.75, 5-character stems), hybrid = reciprocal rank fusion "
         f"(k={RRF_K}) of the two full rankings. Hit = a chunk of the expected file in the "
         f"top-{k}; answerable questions only. rag_core in production is **dense-only**."),
        "",
        (f"Caveat: each knowledge base has only 13–15 chunks, so the top-{k} covers a third of "
         "it and Recall saturates; MRR separates the methods better. One question is 3–4 "
         "points of Recall."),
        "",
        f"| Dataset | Method | Recall@{k} | MRR | " + " | ".join(f"Recall {lang.upper()}" for lang in langs) + " |",
        "|---|---|---|---|" + "---|" * len(langs),
    ]
    for r in rows:
        by_lang = []
        for lang in langs:
            value = r.get(f"recall_{lang}")
            by_lang.append(f"{value[0]:.1%} (n={value[1]})" if value else "—")
        lines.append(
            f"| {r['dataset']} ({r['questions']} q, {r['chunks']} chunks) | {r['method']} | "
            f"{r['recall']:.1%} | {r['mrr']:.3f} | " + " | ".join(by_lang) + " |"
        )
    lines += ["", "## Misses", ""]
    for r in rows:
        if r["misses"]:
            lines.append(f"- **{r['dataset']} / {r['method']}**: " + "; ".join(r["misses"]))
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--out", type=Path, default=EVAL_DIR / "RETRIEVAL_COMPARISON.md")
    args = parser.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    settings = Settings(_env_file=None)
    embedder = SentenceTransformerEmbedder(settings.embedding_model)
    rows = [row for name in DATASETS for row in evaluate(name, embedder, settings, args.top_k)]
    report = render(rows, args.top_k, settings.embedding_model)
    args.out.write_text(report, encoding="utf-8")
    print(report.split("## Misses")[0])
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
