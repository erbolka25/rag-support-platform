# Labeling rule

Written on 2026-11-03, before re-reviewing the labels, and applied to all 50 questions
(the review is below). A label never changes because of a model's answer; only this rule
decides it.

## When does the knowledge base answer a question?

A question is **answerable** (`expected_source` set) if a careful support agent who reads
only `eval/dataset/kb` can give the correct answer, and it falls into one of these:

- **A. Stated.** The answer is written in a document, in any wording or language.
- **B. One-step inference.** The answer follows from stated facts by one arithmetic step,
  comparison, or classification of the asked item into a category the document names
  ("5 floors × 100 som per floor", "40 000 > the 30 000 cash limit", "shampoo is cosmetics,
  and cosmetics cannot be returned").
- **C. Explicit exclusion.** The answer is "no", and a document excludes the asked option
  in words: "only", "not", "no", "cannot" (e.g. "large appliances are delivered **only** in
  Bishkek" answers "do you deliver a fridge to Osh?").

A question is **unanswerable** (`expected_answer: NO_ANSWER`) otherwise. In particular:

- The documents are silent about the topic.
- A "no" would rest only on the option being **absent from a list**. Lists are read as
  open: "you can pay by card, QR or cash" does not say other ways are impossible.
- The answer needs a fact outside the documents (geography, law, prices of other shops).

## Key facts

`key_facts` hold the minimum that a correct answer must contain, in a form that depends on
wording as little as possible: numbers as whole numbers (`30000`, not `30`); for a computed
answer, the result the question asks for, not an intermediate step (`500` for "5 floors",
not `100`); for "no" answers the negation the document uses, cut to a stem (`не подлеж`);
other facts as the document's own term. A key fact must not demand a word the question
already makes unnecessary.

## Review of all 50 questions against this rule

| ID | Label | Rule | Verdict |
|---|---|---|---|
| a01, a02, a04, a06, a08–a11, a13–a15, a17–a24, a26 | answerable | A | kept |
| a16 | answerable | B (washing machine → large appliance, as delivery.txt names it) | kept |
| a03 | answerable | B (12:00 is before the 14:00 cut-off) | kept |
| a05 | answerable | B (5 × 100 som) | kept; key fact fixed |
| a07 | answerable | B (40 000 > 30 000) | kept; key fact fixed |
| a12 | answerable | B + C (shampoo → cosmetics; "не подлежат возврату") | kept; key fact fixed |
| a25 | answerable | A | kept |
| a27–a31 (KY) | answerable | A, asked in Kyrgyz | kept |
| a32–a36 (EN) | answerable | A, asked in English; a33 is A ("chargers" are listed as accessories) | kept |
| n01, n09 courier jobs | NO_ANSWER | silent | kept |
| n02, n10 gift certificates | NO_ANSWER | silent | kept |
| n03, n11 parking | NO_ANSWER | silent | kept |
| n04 instalments | NO_ANSWER | only an open list of payment methods | kept |
| n05 iPhone app | NO_ANSWER | silent (the bank apps are payment channels, not a store app) | kept |
| n06 wardrobe assembly | NO_ANSWER | silent | kept |
| n07 company director | NO_ANSWER | silent | kept |
| n08 delivery to Almaty | NO_ANSWER | delivery lists Kyrgyz towns, no "only Kyrgyzstan"; Almaty being abroad is outside knowledge | kept |
| n12 franchise | NO_ANSWER | silent | kept |
| n13 Bitcoin | NO_ANSWER | **disputed**, see below | kept by decision |
| n14 student discount | NO_ANSWER | silent; the loyalty program does not exclude other discounts | kept |

**n13 "Can I pay with Bitcoin?"** `payment.txt` says "Валюта расчётов — только
кыргызский сом" (the settlement currency is only the Kyrgyz som). If Bitcoin counts as a
currency, rule C applies and the correct answer is "no": the question would be answerable.
If it counts as a payment method, the list of methods is open and the question is
unanswerable. The rule does not settle this, so the label stays `NO_ANSWER` and the
question is marked disputed: an answer "no, only som / not among the methods" is counted
as a hallucination by the metrics but is defensible. Reports state this next to the
hallucination rate.
