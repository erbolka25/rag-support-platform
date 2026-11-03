import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import compare_retrieval as cr


def test_tokens_cut_words_to_stems_and_keep_numbers():
    assert cr.tokens("Доставка доставки 3000 сомов в Ош") == ["доста", "доста", "3000", "сомов", "в", "ош"]


def test_bm25_ranks_the_document_with_the_rare_term_first():
    docs = [cr.tokens(t) for t in ("гарантия на чайник", "доставка по городу", "доставка в регионы")]
    scores = cr.BM25(docs).scores(cr.tokens("гарантия"))
    assert scores.argmax() == 0 and scores[1] == scores[2] == 0


def test_rrf_rewards_agreement_between_rankings():
    dense = np.array([0, 1, 2])
    lexical = np.array([1, 2, 0])
    assert cr.rrf(dense, lexical)[0] == 1  # ranked 2nd and 1st beats 1st and 3rd


def test_datasets_load():
    for load in cr.DATASETS.values():
        kb, cases = load()
        files = {p.name for p in kb.iterdir()}
        assert cases and all(c.source in files for c in cases)
