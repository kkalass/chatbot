# Architecture

## High-Level Design
The application is a locally-run RAG chatbot with explicit separation of concerns.

## Components
- UI Layer (Chainlit)
  - Handles user interaction, streaming output, and source display.
- Application Layer (Chat Orchestrator)
  - Coordinates retrieval, prompt construction, model generation, and optional tool calling.
- Retrieval Layer
  - Query embedding, vector search (Qdrant), reranking/filtering, source packaging.
- Ingestion Layer
  - File discovery, content extraction (multi-modal), chunking, embedding, indexing.
- Tool Integration Layer
  - Typed tool interface (Pydantic models), authentication/context handling, adapter to external service simulation.
- Model Runtime
  - Ollama-hosted local models for generation and extraction support.

## Data Flow
1. Ingestion flow:
   - Load corpus files (txt/md/pdf).
   - Extract normalized textual representation.
   - Chunk content with metadata (document id, section, page if available).
   - Generate embeddings and upsert vectors to Qdrant.
2. Query flow:
   - User message arrives via Chainlit.
   - Orchestrator classifies whether tool call is needed.
   - For knowledge queries: retrieve top-k chunks from Qdrant.
   - Build grounded prompt with retrieved evidence and policy instructions.
   - Generate answer via LLM and return with citations.
   - For tool-eligible queries: invoke typed tool and merge result into response.

## Authentication Flows (MVP)
- The Chainlit UI is accessible without mandatory global app-level login in MVP.
- Authentication is required only for the external service simulation and happens at tool-call time.
- Credentials are scoped to the active chat session and not persisted as long-term application user accounts.
- Tool-call authentication failures are returned as user-safe errors without exposing sensitive details.
- Optional app-level Chainlit authentication can be added later as a separate access-control concern.

## Auth-Protected Tool Sequence (MVP)
1. The orchestrator detects that a user request requires the vacation-days tool.
2. The orchestrator checks server-side session state for valid external-service credentials.
3. If credentials are missing, the chat flow requests username/password and stores them in session scope.
4. The orchestrator resumes the pending intent and invokes the tool adapter.
5. The tool adapter receives typed auth context as explicit function parameters from the orchestrator.
6. The external service is called, and the structured result is merged into the final assistant response.

Implementation rule:
- Credentials must never be sourced from model-generated tool arguments or retrieval context; only orchestrator-managed session state is allowed.

## Key Design Decisions
- UI is Chainlit-first for chat-focused UX and rapid conversational iteration.
- CLI remains available for ingestion/evaluation workflows.
- Gradio fallback is documented to de-risk UI framework blockers. Switch trigger: Chainlit blocks a key user-facing feature for > 1 working days with no viable workaround.
- Multi-modal strategy uses extraction-first approach rather than shared embedding tricks.
- MCP is intentionally excluded from MVP to reduce complexity.

## Suggested Module Boundaries
- src/ui/
- src/app/
- src/retrieval/
- src/ingestion/
- src/tools/
- src/config/
- tests/unit/
- tests/integration/

## Configuration
Use environment variables for:
- model names,
- Ollama endpoint,
- Qdrant host/port,
- corpus path,
- retrieval parameters (top-k, score thresholds).

Runtime note:
- Per-user credentials for the simulated external service are exclusively session-scoped runtime state handled by the orchestrator and are not sourced from environment variables.

## Risk Register (Architecture)
- Local model quality variance may reduce answer quality.
- PDF extraction quality can dominate downstream retrieval quality.
- Prompt injection in documents can affect generation behavior.
- Latency can increase with large corpora if chunking/retrieval strategy is not tuned.
