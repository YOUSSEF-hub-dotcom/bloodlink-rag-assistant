# BloodLink RAG service — handoff guide

FastAPI service that answers blood-donation and BloodLink-platform questions from a local Markdown knowledge base. It uses local `BAAI/bge-m3` embeddings, PostgreSQL + pgvector hybrid retrieval, Groq streaming generation, conversation memory, and deterministic guardrails.

Status: Phases 0–6 implemented. The optional Node/Express proxy is **not** deployed in this repository because the Node backend repo was not available; a ready-to-use snippet is provided below.

---

## 1. Architecture

```text
data/*.md  (4 Markdown files; medical source of truth)
   │
   ▼  ingestion.parse → clean → chunk (### boundaries) → embed (BAAI/bge-m3, 1024 dim)
   │
   ▼  rag.chunks  (vector(1024), HNSW index using vector_cosine_ops)
   │
   ├── Dense leg:  pgvector cosine score 1 - (embedding <=> query)
   │               language filter, then one cross-lingual retry if best < SCORE_THRESHOLD
   ├── Sparse leg: native tsvector / ts_rank_cd ('simple' + Arabic normalization
   │               + exact blood-type boosts) over the same threshold-passing candidates
   ▼
   RRF fusion (k=60) → top_k results
   │
   ▼  generation/pipeline.py  (ordered chat turn)
   │   1 auth (API dependency) + input length check
   │   2 emergency regex (AR/EN) → canned answer, zero LLM calls
   │   3 load buffered history + rolling summary; rewrite follow-up with
   │     Groq openai/gpt-oss-20b strict JSON → standalone_question / language / intent
   │   4 intent routing (off-topic / smalltalk / injection → canned, zero answer LLM)
   │   5 hybrid retrieval above (no results → canned NO_INFO, zero answer LLM)
   │   6 streamed Groq answer (openai/gpt-oss-120b, fallback openai/gpt-oss-20b)
   │   7 persist user/assistant text → rolling summary → non-blocking post-checks
   ▼
   FastAPI SSE: meta / token / done / error events
   ▼
   Optional Express /api/chat proxy (port 5000) → React/Vite frontend
```

Key design choices:

- **Hybrid Search** = dense pgvector cosine + sparse native full-text search fused by Reciprocal Rank Fusion (`RRF, k=60`). This is retrieval fusion, not a learned reranker.
- `SCORE_THRESHOLD=0.55` is applied to the **dense cosine score before fusion**. Sparse terms can re-rank candidates that already passed the gate but cannot rescue chunks below it.
- `meta.sources` come only from retrieval metadata — never from the LLM.
- The original user message goes to the answer LLM; the standalone rewrite is used only for retrieval.
- Rolling summaries and post-stream guardrails (number grounding, diagnosis/dosage phrases) only **log** warnings; no output filter blocks the streamed answer. Answers are not claimed to be medically verified.

Verified local environment: Python 3.11.8, `torch 2.15.0+cu132`, CUDA 13.2, RTX 4050 Laptop (6 GB), PostgreSQL 16.16, pgvector 0.8.7, `BAAI/bge-m3` (1024 dim), `rag` schema.

---

## 2. Prerequisites

- Python 3.11.x with `venv`.
- NVIDIA GPU + driver compatible with CUDA 13.2 (or set `EMBEDDING_DEVICE=cpu`; slower).
- PostgreSQL with the `pgvector` extension. Local development uses `docker compose up -d` with `pgvector/pgvector:pg16`.
- A Groq API key. Model IDs are verified against `GET /openai/v1/models` at API startup.

---

## 3. Local setup

```powershell
# 1. Virtual environment
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip

# 2. CUDA build of PyTorch (matches requirements.txt / cu132 index)
.\.venv\Scripts\python.exe -m pip install "torch==2.15.0+cu132" --extra-index-url https://download.pytorch.org/whl/cu132

# 3. All pinned dependencies (includes httpx; optional Streamlit block is marked)
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

# 4. Confirm CUDA before ingestion
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"

# 5. PostgreSQL + pgvector (local)
docker compose up -d

# 6. Configuration
copy .env.example .env
```

Edit `.env` before starting anything:

- `GROQ_API_KEY` — your key (never commit `.env`).
- `DATABASE_URL` — must match the local container (`postgresql://postgres:postgres@localhost:5432/bloodlink_db`).
- `JWT_SECRET` — must be the **same secret** the Node backend uses.
- `AUTH_ENABLED=false` for local development (no token required; `user_id = null`).
- `EMBEDDING_BATCH_SIZE=16` is safer on a 6 GB GPU; the template default is 32.

`bootstrap_schema()` creates the extension, `rag` schema, `chunks`, `conversations`, `messages`, and HNSW/FTS indexes automatically on the first ingestion or API start. No manual SQL is needed locally.

### Configuration reference

All values are read exclusively through `core/config.py`; nothing is hardcoded.

| Variable | Default | Notes |
| --- | --- | --- |
| `GROQ_API_KEY` | empty | Required for rewrite and generation. Never logged. |
| `GROQ_PRIMARY_MODEL` | `openai/gpt-oss-120b` | Answer model; verified live. |
| `GROQ_FALLBACK_MODEL` | `openai/gpt-oss-20b` | Answer fallback + query rewrite + rolling summary. |
| `LLM_TIMEOUT_SECONDS` | `30` | Groq timeout. |
| `DATABASE_URL` | local placeholder | PostgreSQL connection string. |
| `RAG_DB_SCHEMA` | `rag` | Schema holding all three tables. |
| `EMBEDDING_MODEL` | `BAAI/bge-m3` | Local sentence-transformer. |
| `EMBEDDING_DEVICE` | `cuda` | `cuda` or `cpu`. |
| `EMBEDDING_BATCH_SIZE` | `32` | Lower to 16 on 6 GB VRAM. |
| `MAX_CHUNK_TOKENS` | `400` | Chunking target. |
| `CHUNK_OVERLAP_TOKENS` | `40` | Chunk overlap. |
| `TOP_K` | `5` | Retrieval candidates after fusion. See §9. |
| `SCORE_THRESHOLD` | `0.55` | Dense-score gate before fusion. See §9. |
| `HNSW_EF_SEARCH` | `40` | Set with `SET LOCAL` in the search transaction. |
| `HISTORY_MAX_MESSAGES` | `8` | Buffered turns sent to prompts. |
| `SUMMARY_TRIGGER_MESSAGES` | `16` | Rolling-summary threshold. |
| `MAX_USER_MESSAGE_CHARS` | `1000` | Blocking input length cap. |
| `AUTH_ENABLED` | `true` | Set `false` only for local development. |
| `JWT_SECRET` | empty | Shared with Node backend. |
| `JWT_ALGORITHM` | `HS256` | Node controller uses HS256. |
| `JWT_USER_CLAIM` | `id` | Node signs `jwt.sign({ id }, ...)`. |
| `CORS_ALLOW_ORIGINS` | Vite dev origins | Comma-separated; add the production origin. |
| `CHAT_RATE_LIMIT` | `20` | Requests per window per user/IP. |
| `CHAT_RATE_WINDOW_SECONDS` | `60` | Sliding-window length. |
| `LOG_LEVEL` | `INFO` | Structured JSON logs; message text only at DEBUG. |

`.env.example` contains placeholders only, and `.gitignore` ignores `.env` / `.env.*` (except `.env.example`), `.venv/`, caches, and model artifacts.

---

## 4. Ingestion

The four Markdown files in `data/` are the medical source of truth. Do not edit them without clinical review. `[verify]` markers are stripped from chunk text but recorded as `has_unverified_markers: true` in metadata (12 chunks currently).

```powershell
# Incremental: hash-based skip of unchanged chunks
.\.venv\Scripts\python.exe -m ingestion.run

# Full rebuild: truncate rag.chunks and re-embed every chunk
.\.venv\Scripts\python.exe -m ingestion.run --reset
```

Verified result after Phase 1: **71 chunks** — `en: 36`, `ar: 35`. A second incremental run embeds 0 chunks unless file content changes.

---

## 5. Debug CLIs and tests

```powershell
# Retrieval trace: dense_score, sparse_rank/score, rrf_score, section, language, preview.
# Ignores SCORE_THRESHOLD so you can see what is just below the gate.
.\.venv\Scripts\python.exe -m retrieval.debug "هل ينفع أتبرع وأنا بصوم؟" --lang ar --k 5

# Single-turn RAG (no memory), optional forced fallback demo
.\.venv\Scripts\python.exe -m generation.debug_turn "Can I donate?"
.\.venv\Scripts\python.exe -m generation.debug_turn "Can I donate?" --mock-fallback

# Multi-turn memory + rewrite + rolling summary (temporary low threshold)
.\.venv\Scripts\python.exe -m memory.debug_multiturn `
  --summary-trigger 2 --history-max 2 `
  "إيه الشروط الأساسية للتبرع بالدم؟" "طب لو أنا ست؟"

# Full unit suite (no live Groq, no GPU, no DB required; everything is mocked)
.\.venv\Scripts\python.exe -m pytest -q
```

`generation.debug_turn` also accepts `--conversation-id`, `--summary-trigger`, `--history-max`, `--lang`, `--no-verify`, and `--retry-delay`.

Expected test result: `57 passed`.

---

## 6. FastAPI server

```powershell
.\.venv\Scripts\python.exe -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/health` | status, device, embedding model, chunk count, Groq availability |
| `POST` | `/v1/chat` | SSE chat turn |
| `GET` | `/v1/conversations/{id}/messages` | stored messages (owner only) |
| `DELETE` | `/v1/conversations/{id}` | delete conversation (owner only) |

`POST /v1/chat` body: `{ "conversation_id": "uuid|null", "message": "..." }`. A new UUID is accepted and created by the pipeline; an existing UUID is ownership-checked when auth is enabled. SSE events:

```text
event: meta
data: {"conversation_id":"...","language":"ar","intent":"on_topic","sources":[{"doc_id":"01_donor_eligibility","section_path":"...","language":"ar","score":0.7811}]}

event: token
data: {"text":"..."}

event: done
data: {"message_id":14}
```

`error` is used for input/validation or stream failures. Canned replies (emergency, off-topic, no-context) still use `meta → token → done` with `sources: []` and perform zero answer-LLM calls. When `AUTH_ENABLED=true`, a missing/invalid Bearer token returns HTTP 401.

---

## 7. `curl -N` examples

No auth (`AUTH_ENABLED=false`):

```bash
curl -N -X POST "http://127.0.0.1:8000/v1/chat" \
  -H "Content-Type: application/json" \
  -d '{"conversation_id":null,"message":"إيه الشروط الأساسية للتبرع بالدم؟"}'
```

With a Node-issued JWT (`AUTH_ENABLED=true`):

```bash
TOKEN="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
curl -N -X POST "http://127.0.0.1:8000/v1/chat" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"message":"Can I donate?"}'
```

Same conversation, follow-up turn:

```bash
curl -N -X POST "http://127.0.0.1:8000/v1/chat" \
  -H "Content-Type: application/json" \
  -d '{"conversation_id":"aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa","message":"طب لو أنا ست؟"}'
```

Other endpoints:

```bash
curl -s "http://127.0.0.1:8000/health"
curl -s "http://127.0.0.1:8000/v1/conversations/aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa/messages"
curl -s -X DELETE "http://127.0.0.1:8000/v1/conversations/aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"
```

Add `-H "Authorization: Bearer $TOKEN"` to each when auth is enabled. `-N` disables curl buffering so tokens appear live.

---

## 8. Optional Streamlit test UI

`demo_ui.py` is a local developer tool, not the production UI. Its dependencies (`streamlit`, `httpx`) are pinned in the optional block of `requirements.txt`.

```powershell
.\.venv\Scripts\python.exe -m streamlit run demo_ui.py
```

Open the printed URL (normally `http://localhost:8501`). Features: multi-turn chat with a **New Conversation** button (fresh UUID), real-time token streaming, an expandable **Debug / Retrieval Metadata** box under each reply (`intent`, `language`, `sources` with `doc_id`, `section_path`, `score`), and sidebar inputs for the API URL and an optional Bearer JWT.

---

## 9. How to tune `TOP_K` and `SCORE_THRESHOLD` manually

1. Start from the verified values: `TOP_K=5`, `SCORE_THRESHOLD=0.55`. Restart the API after changing `.env`.
2. Use the debug CLI to inspect what is near the cut-off; it deliberately ignores the threshold:

   ```powershell
   .\.venv\Scripts\python.exe -m retrieval.debug "I take warfarin, can I donate?" --lang en --k 10
   ```

   Look at `dense_score`, `sparse_rank`, `rrf_score` and the section path of each row.
3. If correct sections fall just **below** the gate and the bot returns `NO_INFO` too often, lower `SCORE_THRESHOLD` in small steps (e.g. 0.55 → 0.50 → 0.45). If unrelated sections are being retrieved, raise it (e.g. 0.55 → 0.60 → 0.65).
4. If the right section appears at rank 6–10, raise `TOP_K` instead of lowering the threshold. Higher `TOP_K` increases prompt size and latency; keep it small (5–10).
5. Remember the gate applies to the **dense score before RRF fusion**. A sparse keyword match cannot add a chunk that failed the gate. The debug CLI prints both scores so you can tell which leg is helping.
6. Re-check both languages separately (Arabic and English) and re-run `pytest -q`. Log fields to watch: `chunks`, `top_score`, `intent`, `llm_calls`, `model`, `ttft_ms`, `total_latency_ms`.
7. Record any production change and get medical review before changing knowledge-base content to compensate for retrieval tuning.

---

## 10. Optional Node/Express SSE proxy (port 5000)

The Node backend repo was not provided, so this is a ready-to-use snippet to add to its Express app (Node 18+). Put it **after** your existing JWT auth middleware so the incoming `Authorization` header is already validated; forward it as-is (do not re-sign), and set `RAG_URL` to the FastAPI service.

```js
// routes/ragChat.js
const express = require('express');
const router = express.Router();

const RAG_URL = process.env.RAG_URL || 'http://127.0.0.1:8000';
const PASSTHROUGH = new Set([
  'content-type', 'cache-control', 'connection', 'x-accel-buffering',
]);

router.post('/api/chat', async (req, res) => {
  const headers = {
    'Content-Type': 'application/json',
    Accept: 'text/event-stream',
  };
  if (req.headers.authorization) headers.Authorization = req.headers.authorization;

  try {
    const upstream = await fetch(`${RAG_URL}/v1/chat`, {
      method: 'POST',
      headers,
      body: JSON.stringify({
        conversation_id: req.body.conversation_id ?? null,
        message: req.body.message,
      }),
    });

    res.status(upstream.status);
    upstream.headers.forEach((value, key) => {
      if (PASSTHROUGH.has(key.toLowerCase())) res.setHeader(key, value);
    });

    if (!upstream.ok || !upstream.body) {
      res.end(await upstream.text());
      return;
    }

    // Pipe SSE frames unbuffered.
    const reader = upstream.body.getReader();
    const decoder = new TextDecoder('utf-8');
    let buffer = '';
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const frames = buffer.split('\n\n');
      buffer = frames.pop() || '';
      for (const frame of frames) res.write(`${frame}\n\n`);
    }
    if (buffer.trim()) res.write(`${buffer}\n\n`);
    res.end();
  } catch (error) {
    if (!res.headersSent) res.status(502).json({ error: 'RAG service unavailable' });
    else res.end();
  }
});

module.exports = router;
```

```js
// app.js
app.use('/api', require('./routes/ragChat')); // after express.json() and auth middleware
app.listen(5000);
```

The frontend calls `POST /api/chat` on the port-5000 backend; CORS on FastAPI is only needed for direct dev calls from Vite. Keep the Node rate limiting/auth in place; the FastAPI limiter is an additional layer.

---

## 11. Railway PostgreSQL + production deployment checklist

1. **pgvector on Railway:** create the Postgres plugin and confirm the extension is installable. The app runs `CREATE EXTENSION IF NOT EXISTS vector`; the DB user must be allowed to do this. Verify `SELECT extversion FROM pg_extension WHERE extname = 'vector';` and that `vector(1024)`, HNSW with `vector_cosine_ops`, and `tsvector`/`ts_rank_cd` work. **Do not assume Railway supports pgvector — verify.** If it does not, use a Postgres host that does.
2. Set `DATABASE_URL` from Railway's connection details (SSL), plus `RAG_DB_SCHEMA=rag`.
3. Set `GROQ_API_KEY`, `GROQ_PRIMARY_MODEL`, `GROQ_FALLBACK_MODEL`; confirm availability from the startup log.
4. Set `JWT_SECRET` to the exact Node secret, `JWT_ALGORITHM=HS256`, `JWT_USER_CLAIM=id`, and `AUTH_ENABLED=true` in production.
5. Add the production frontend origin to `CORS_ALLOW_ORIGINS` if it calls FastAPI directly.
6. Run ingestion once against the production DB (`--reset` for a first load), then confirm 71 chunks / 36 en / 35 ar via `/health`.
7. Set `TOP_K=5`, `SCORE_THRESHOLD=0.55`, `HISTORY_MAX_MESSAGES=8`, `SUMMARY_TRIGGER_MESSAGES=16`, `CHAT_RATE_LIMIT=20` / `CHAT_RATE_WINDOW_SECONDS=60`; tune per §9.
8. Start with `uvicorn api.main:app --host 0.0.0.0 --port $PORT` ( Railway-injected `PORT`). Run the optional Node proxy on 5000 with `RAG_URL` pointing at the FastAPI service.
9. Smoke-test `/health`, one grounded AR turn, one grounded EN turn, an emergency turn (zero LLM calls), an out-of-scope turn, and a two-turn follow-up. Check structured logs for `model`, `top_score`, `llm_calls`, `ttft_ms`, and no secrets.
10. Ensure Railway backups include both `rag.chunks` and the conversation/message tables; restoring embeddings without the rest loses chat history and vice versa.

---

## 12. Handoff items for the team

- Confirm the JWT claim (`id`) and algorithm (`HS256`) against the live `auth.controller.js` / middleware; both are configurable.
- Verify pgvector support on the production Postgres host.
- Review the `[verify]` UI labels and the 12 chunks flagged `has_unverified_markers`.
- Tune `SCORE_THRESHOLD` / `TOP_K` with real user questions before release.
- Arrange medical review of all four `data/*.md` files; warfarin-related questions are a known coverage gap.
- Do not treat generated answers as clinically verified; post-checks only log warnings.
