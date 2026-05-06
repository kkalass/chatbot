# Evaluation

This directory contains scripts and datasets for offline evaluation of the RAG chatbot
using [Arize Phoenix](https://phoenix.arize.com).

---

## Phoenix Feature Landscape

Phoenix offers several overlapping features. Here is how they relate:

### Tracing (already active)
Live spans from the running chatbot are sent to Phoenix via OTLP. Every production
conversation is visible under **Traces** in the Phoenix UI. This is the observation
layer — it shows what happened, but does not systematically score it.

### Experiments (primary eval workflow)
Experiments are the systematic, offline evaluation workflow:

1. **Dataset** — a curated table of `{input, expected_output}` examples.
2. **Task** — a function that runs the system against one input and returns an output.
3. **Evaluators** — functions that score the task output (code-based or LLM-as-judge).

`client.experiments.run_experiment(dataset, task, evaluators)` runs the full matrix
and records results in Phoenix. Multiple experiment runs against the same dataset are
compared side-by-side in the UI. This is the main tool for measuring whether a code,
prompt, or retrieval change improved quality.

→ **Start here.** See `run_experiment.py`.

### Annotations
Annotations are labels (scores, categories, notes) attached to existing spans/traces.
They can be created:
- **Manually in the UI** — reviewers click thumbs-up/down or add notes to live traces.
- **Programmatically** — `client.annotations.add_span_annotation(...)` for automated
  post-hoc LLM-as-judge scoring of production traffic.
- **From the UI: "Add to Dataset"** — select a span in the trace view and export it
  directly into a dataset. This is the primary way to grow your evaluation dataset
  from real conversations.

Use annotations to curate your dataset over time, not for primary experiment runs.

### Evaluations (arize-phoenix-evals SDK)
`arize-phoenix-evals` provides **LLM-as-judge** templates for metrics like
*faithfulness*, *relevance*, *Q&A correctness*, *toxicity*, etc. These can be used:
- As evaluator functions passed to `run_experiment()` (most useful here).
- Post-hoc on exported trace DataFrames via `run_evals(df, ...)`.

They are a library, not a Phoenix UI feature. Use them as evaluators inside experiments.

### Prompt Hub (Prompts)
Phoenix can store and version prompt templates centrally. You pull a prompt version
via `client.prompts.get_prompt(name, tag="production")` and inject it at runtime.
Useful if you want to manage prompt versions in Phoenix rather than in Git.

→ **Not needed for this project** — prompts live in `src/chatbot/app/prompts.py`
and are versioned via Git. Consider using Prompt Hub only if you want to run A/B
prompt experiments directly from the Phoenix Playground.

### Prompt Playground
The interactive web UI for testing prompt templates against a dataset without code.
Great for quick iteration on the system prompt phrasing. It cannot easily drive the
full RAG pipeline (Qdrant retrieval + tool calling), so it is not the primary eval
surface for this chatbot.

→ **Optional / future use.** Useful for isolated prompt fragment testing.

---

## Setup

Install the eval dependency group:

```sh
uv sync --group eval
```

Make sure Phoenix is running locally (started alongside the chatbot):

```sh
# If not already running via docker-compose, start manually:
uv tool run --from arize-phoenix python -m phoenix.server.main serve
```

Configure `.env` to enable tracing and point to Phoenix:

```dotenv
OTEL_ENABLED=true
OTEL_EXPORT_PHOENIX=true
OTEL_PHOENIX_OTLP_ENDPOINT=http://localhost:6006/v1/traces
PHOENIX_BASE_URL=http://localhost:6006
```

The experiment runner reads the same `.env` as the application.

---

## Running an Experiment

```sh
# Default: loads eval/datasets/rag_questions.jsonl
uv run --group eval python eval/run_experiment.py

# Custom dataset file and name
uv run --group eval python eval/run_experiment.py \
  --dataset-file eval/datasets/rag_questions.jsonl \
  --dataset-name rag-questions-v1 \
  --experiment-name "retrieval-top-k-5"

# Dry run (sanity-check task function against 1 example, no Phoenix upload)
uv run --group eval python eval/run_experiment.py --dry-run

# Re-run only evaluators on an existing experiment (tasks do not re-execute)
uv run --group eval python eval/run_experiment.py --experiment-id <ID>

# Replay: create a NEW experiment from cached task outputs of an existing one
uv run --group eval python eval/run_experiment.py \
  --replay-from <SOURCE_EXPERIMENT_ID> \
  --experiment-name "phase11-n3a-judge-qwen25coder"
```

The experiment ID is shown in the Phoenix UI on the experiment detail page.

**`--experiment-id`** — re-runs evaluators on the *same* existing Phoenix experiment.
The evaluator columns are overwritten in place (upsert by evaluator name).
Use this to quickly rescore a run after fixing a buggy evaluator.

**`--replay-from`** — fetches the cached task outputs from the source experiment and
creates a **new** Phoenix experiment entry using `run_experiment`.  The task function
is not re-executed; the RAG pipeline is not called.  The new experiment gets its own
name, description, and metadata (reflecting the current `.env` judge config), and its
evaluator columns are fully independent of the source experiment.
Use this to compare different LLM judges side-by-side in the Phoenix experiment
comparison view without re-running the full RAG pipeline.

After the run, open the Phoenix UI at `http://localhost:6006` and navigate to
**Datasets → rag-questions-v1** to compare experiment runs.

Every experiment run is tagged with a `metadata` dict so runs with different
configurations can be filtered and compared in the UI:

| Key | Source |
|-----|--------|
| `chat_model` | `CHAT_MODEL` env var |
| `chat_model_provider` | `CHAT_MODEL_PROVIDER` env var |
| `embedding_model` | `EMBEDDING_MODEL` env var |
| `embedding_model_provider` | `EMBEDDING_MODEL_PROVIDER` env var |
| `eval_judge_model` | `EVAL_JUDGE_MODEL` env var |
| `eval_judge_provider` | `EVAL_JUDGE_PROVIDER` env var |
| `git_commit` | `git rev-parse --short HEAD` |

---

## Dataset Format

The JSONL format has one JSON object per line:

```jsonc
{"query": "...", "reference_answer": "..."}   // with reference answer
{"query": "..."}                               // without (evaluators are code-only)
```

`query` is the required input key. `reference_answer` is optional; use it for
evaluators that compare output against a ground-truth string.

---

## Adding Evaluators

Edit `run_experiment.py` and add evaluator functions to the `evaluators` list in
`main()`. Evaluators receive the task output dict and return a `bool | float`.

**Convention: `True` / `1.0` = good, `False` / `0.0` = bad.** This ensures that
Phoenix displays consistent score bars — a high score always means high quality.
Never write an evaluator where `True` signals a problem; invert the logic and
rename accordingly (e.g. `no_hallucinated_citations` instead of `has_hallucinations`).

Built-in code evaluators:

| Name | Returns `True` when … |
|------|----------------------|
| `has_citations` | answer contains at least one `[N]` marker |
| `is_non_empty` | answer is longer than 50 characters |
| `no_unsubstantiated_claims` | `unsubstantiated_claim_count == 0` |
| `no_hallucinated_citations` | `hallucinated_citation_count == 0` |

Example:

```python
def has_citations(output: dict[str, object]) -> bool:
    """Check that the answer contains at least one citation marker [N]."""
    return bool(re.search(r"\[\d+\]", str(output.get("answer", ""))))
```

For LLM-as-judge evaluators, see `arize-phoenix-evals`:
- `phoenix.evals.run_evals` for faithfulness, Q&A correctness, etc.
- Wire them up via `evaluate_experiment()` after the run if you prefer to separate
  the task run from the scoring step.
