# Product Scope

## Problem Statement
Build a chatbot that answers user questions grounded in static, multi-modal company content (initially text/markdown/pdf). The bot must reduce hallucinations by retrieval grounding and provide transparent source references.

## Product Goal
Deliver an assistant that runs on local infrastructure and can:
- answer domain questions with citation-style references,
- query a simple external service via a typed tool call,
- ingest and index static documents reliably,
- run with minimal setup for developer experimentation.

## Target Users
- Primary: developers and technical stakeholders evaluating RAG quality.
- Secondary: internal users exploring knowledge retrieval on static documents.

## In Scope (MVP)
- Chat UI with conversation history and streaming responses (Chainlit).
- Local model runtime via Ollama.
- RAG over txt, md, and pdf documents.
- Multi-modal ingestion via document content extraction.
- Type-specific ingestion conversion (txt/text, md/markdown) with sidecar metadata support (`<document>.meta.json`).
- Vector storage in Qdrant.
- One typed external tool call ("vacation days left" simulation) with simple username/password context.
- Source attribution in answers.
- Basic observability for quality checks and troubleshooting.

## Out of Scope (MVP)
- Production-grade IAM/SSO.
- Multi-tenant architecture.
- Dynamic document upload by end users.
- MCP-based tool ecosystem.
- Full cloud deployment hardening.

## UI Strategy
- Primary UI: Chainlit.
- Dev utility interface: lightweight CLI scripts for ingestion and evaluation runs.
- Fallback UI: minimal Gradio chat in case Chainlit blocks progress.

## Success Criteria (MVP)
- 80%+ correctness on a curated evaluation set of at least 20 domain questions.
- 90%+ of answers include at least one relevant source reference when a source exists.
- p95 end-to-end response time <= 10s on local setup for typical queries.
- Reproducible setup from README on a clean machine.

## Constraints and Assumptions
- Models and services run locally for developer experimentation, but the architecture must not prevent substitution with hosted models or cloud services later.
- Ingestion embedding must be consumed through an injected boundary so provider/runtime can be swapped without changing orchestration logic.
- No strict regulatory privacy requirements for MVP.
