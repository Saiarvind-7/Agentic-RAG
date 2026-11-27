# Agentic RAG System with Hybrid Retrieval and Self-Evaluation

A question-answering system for your own documents (.txt, .md, .pdf). It does three things
that a basic RAG demo does not:

1. **Hybrid retrieval.** Every search runs two ways at once: by meaning (embeddings in
   ChromaDB) and by exact words (BM25). The two ranked lists are merged with reciprocal
   rank fusion, so you find "annual leave" when someone asks about "vacation days" and
   also find "POL-305" when someone types the code.
2. **An agent loop.** The agent plans search queries, reads what came back, decides whether
   it is enough, and searches again with new queries if it is not.
3. **Self-evaluation.** After writing an answer it checks every claim against the retrieved
   text. If too many claims are unsupported, it rewrites the answer without them. A separate
   RAGAS script scores the whole system on a set of test questions.

## How a question flows

```
question
   |
   v
 PLAN            LLM writes up to 3 search queries (or skips search for small talk)
   |
   v
 RETRIEVE  <---------------------------+
   dense (ChromaDB + sentence-transformers)   |
   keyword (BM25)                              |  not enough? LLM writes new queries
   merged with reciprocal rank fusion          |  (up to MAX_ATTEMPTS rounds)
   |                                           |
   v                                           |
 GRADE           is this context enough? -----+
   |
   v
 ANSWER          LLM answers only from the context, with [1] [2] citations
   |
   v
 SELF-CHECK      LLM lists claims the context does not support, gives a 0-1 score
   |  score below FAITHFULNESS_THRESHOLD?
   +--> rewrite the answer once, leaving those claims out
   |
   v
 answer + sources + score + full trace
```

## Setup

You need Python 3.11 or 3.12 and an API key for any OpenAI-compatible service.

```bash
# 1. create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 2. install (the first install is large because sentence-transformers pulls in PyTorch)
pip install -r requirements.txt

# 3. add your key
cp .env.example .env               # Windows: copy .env.example .env
#    then open .env and set OPENAI_API_KEY

# 4. check the install (no API key or internet needed)
pytest

# 5. build the index from the sample documents in data/docs
python -m app.ingest
#    the first run downloads the embedding model (about 90 MB)

# 6. start the API (terminal 1)
uvicorn app.api:app --reload

# 7. start the UI (terminal 2, same virtual environment)
streamlit run ui/streamlit_app.py
```

Open http://localhost:8501 and try: *"How many days of annual leave do I get?"*,
*"What does POL-305 say about passwords?"* or *"How fast must I report a lost laptop?"*.
The API docs are at http://localhost:8000/docs.

### Using your own documents
Put files in `data/docs/` and run `python -m app.ingest` again, or use the sidebar in the UI
(Upload, then Rebuild index). Delete the sample files if you don't want them in your index.

### Using a free local model instead of OpenAI
Install Ollama, run `ollama pull llama3.1`, then set these in `.env`:
```
OPENAI_BASE_URL=http://localhost:11434/v1
OPENAI_API_KEY=ollama
LLM_MODEL=llama3.1
```
Small local models are weaker at the JSON-style planning and checking steps, so expect
less reliable self-checks than with a larger model.

## Evaluating it

```bash
python -m app.evaluate               # hybrid retrieval
python -m app.evaluate --mode all    # dense vs bm25 vs hybrid, side by side
```
This runs the questions in `data/eval_set.json` through the agent and scores them with
RAGAS: faithfulness, answer relevancy, context precision and context recall. Results go to
`results/`. It makes many LLM calls, so it costs a little. Replace the sample questions with
ones about your own documents, with a short correct answer for each.

## Project layout

```
app/
  config.py      settings (env vars)
  ingest.py      read files, chunk, build ChromaDB + BM25 data
  embeddings.py  sentence-transformers wrapper
  fusion.py      tokenizer and reciprocal rank fusion
  retriever.py   HybridRetriever (dense, bm25 or hybrid)
  llm.py         OpenAI-compatible chat helper
  agent.py       plan -> retrieve -> grade -> answer -> self-check
  api.py         FastAPI: /health /upload /ingest /ask
  evaluate.py    RAGAS evaluation script
ui/streamlit_app.py
data/docs/       sample documents (a made-up company's policies)
data/eval_set.json
tests/           run offline with faked embeddings and LLM
```

## Settings worth knowing (in .env)

| Setting | Default | What it does |
| --- | --- | --- |
| `TOP_K` | 5 | chunks the LLM sees |
| `MAX_ATTEMPTS` | 2 | retrieval rounds before answering anyway |
| `FAITHFULNESS_THRESHOLD` | 0.7 | self-check score needed to accept an answer |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | 800 / 120 | how documents are split |
| `LLM_MODEL` | gpt-4o-mini | any chat model your provider offers |

## Troubleshooting

- **409 "index has not been built"**: run `python -m app.ingest` or click Rebuild index.
- **"No API key found"**: check that `.env` is in the project root and has `OPENAI_API_KEY`.
- **Cannot reach the API in the UI**: start `uvicorn` first, and keep both terminals open.
- **Slow first start**: the embedding model is downloading. It is cached afterwards.
- **RAGAS errors after an upgrade**: RAGAS changes quickly. This project is written for
  `ragas>=0.4,<0.5`.

## Ideas to extend it

Add a cross-encoder reranker after fusion, keep chat history for follow-up questions,
stream tokens in the UI, or put the API and UI in Docker.
