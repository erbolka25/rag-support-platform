All changes below fix the **measurement**, not the system: the knowledge base, the
questions and the labels stay the same, only what the checker requires of an answer
changes. They follow the labeling rule in [LABELING.md](LABELING.md), written before the
labels were re-reviewed and applied to all 50 questions.

**2026-11-03 — key facts of a05, a07, a12 (after the second full run)**

| ID | Question | Key facts before → after | Why |
|---|---|---|---|
| a07 | Можно ли оплатить наличными заказ на 40 000 сомов? | `30` → `30000` | The checker joins "30 000" into "30000", so a correct answer quoting the 30 000 som limit failed the fact "30". Numbers are written whole. |
| a12 | Можно ли вернуть шампунь надлежащего качества? | `космет` → `не подлеж` | The answer only has to say that it cannot be returned; demanding the word "cosmetics" failed a correct "шампунь возврату не подлежит". The fact is now the document's negation. |
| a05 | Сколько стоит подъём холодильника на 5 этаж без лифта? | `100` → `500` | Found by applying the rule to every question: the question asks for the total for 5 floors (5 × 100 = 500 som), so the per-floor price is an intermediate step a correct answer may omit. |

**Labels: no change.** All 50 labels were re-checked against the rule; none flipped.

**n13 "Can I pay with Bitcoin?" — disputed label, kept as NO_ANSWER.** `payment.txt`
says the settlement currency is only the Kyrgyz som. If Bitcoin counts as a currency, that
sentence excludes it and the question is answerable ("no"); if it counts as a payment
method, the list of methods is open and the question is unanswerable. The rule does not
settle it, so the label stays. Consequence: an answer like "No, Bitcoin is not accepted
[1]" is counted as a hallucination although it is defensible.
