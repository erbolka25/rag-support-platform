---
title: Support Copilot
emoji: 🎧
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
short_description: Source-cited support replies in RU/EN/KG via RAG
---

# AI Customer Support Copilot — demo

Drafts polite, source-cited replies to customer tickets in Russian, English and Kyrgyz from a
company knowledge base; when an answer can't be backed by a citation, the ticket is escalated
to a human operator.

The demo starts with the knowledge base of **TechNomad**, a fictional electronics store
(delivery, returns, payment, warranty), and ten customer tickets: pick one and press
"Draft". Seven of them are answered by the documents, three should be escalated.

- LLM: `openai/gpt-oss-20b` on Groq, embeddings: `intfloat/multilingual-e5-small` on CPU.
- Limit: 20 drafts and uploads per hour per IP. Data is reset when the Space restarts.
- API docs: `/docs`.

Source code: <https://github.com/erbolka25/rag-support-platform> (`apps/support_copilot`, engine in `packages/rag_core`)
