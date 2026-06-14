# Shastra AI

**Shastra AI** is an AI-powered coding assistant built with [LangGraph](https://github.com/langchain-ai/langgraph).
It works like a multi-agent development team that can take a natural language request and transform it into a complete, working project, file by file, using real developer workflows.

---

## Architecture

- **Planner Agent** - Analyzes your request and generates a detailed project plan.
- **Architect Agent** - Breaks down the plan into specific engineering tasks with explicit context for each file.
- **Coder Agent** - Implements each task, writes directly into files, and uses available tools like a real developer.

<div style="text-align: center;">
    <img src="resources/coder_buddy_diagram.png" alt="Coder Agent Architecture" width="90%"/>
</div>

---

## Getting Started

### Prerequisites

- Make sure you have `uv` installed. Follow the instructions [here](https://docs.astral.sh/uv/getting-started/installation/) to install it.
- Ensure that you have created a Groq account and have your API key ready. Create an API key [here](https://console.groq.com/keys).

### Installation and Startup

- Create a virtual environment using: `uv venv`
- Activate it:
  - Windows PowerShell: `.venv\Scripts\Activate.ps1`
  - macOS/Linux: `source .venv/bin/activate`
- Install the dependencies using: `uv sync`
- Create a `.env` file using the variables from `.sample_env`

Run the application:

```bash
python main.py
```

### Runtime Options

- `GROQ_API_KEY` is required.
- `GROQ_MODEL` is optional and defaults to `openai/gpt-oss-20b`.
- You can override individual roles with `GROQ_PLANNER_MODEL`, `GROQ_ARCHITECT_MODEL`, `GROQ_CODER_MODEL`, and `GROQ_DEBUGGER_MODEL`.
- `GROQ_MODEL_FALLBACKS` is a comma-separated fallback chain. When a Groq model hits a retryable rate/token/model error, Shastra AI automatically tries the next model instead of failing immediately.
- `GROQ_AUTO_MODEL_FALLBACKS=true` appends chat-capable candidates from Groq's live `/models` API after your configured fallback chain.
- Role-specific fallback chains are supported with `GROQ_PLANNER_MODEL_FALLBACKS`, `GROQ_ARCHITECT_MODEL_FALLBACKS`, `GROQ_CODER_MODEL_FALLBACKS`, and `GROQ_DEBUGGER_MODEL_FALLBACKS`. Leave them empty to reuse `GROQ_MODEL_FALLBACKS`.
- Project edit/chat mode can use a separate key/model with `GROQ_EDITOR_API_KEY` and `GROQ_EDITOR_MODEL`. If `GROQ_EDITOR_API_KEY` is empty, it falls back to `GROQ_API_KEY`.
- Project edit/chat mode also supports `GROQ_EDITOR_MODEL_FALLBACKS`.
- You can inspect the full live Groq model list plus the resolved generation/editor fallback chains at `GET /api/models/groq`.
- Project chat history is stored in SQLite. `SHASTRA_DB_PATH` is optional and defaults to `./shastra_ai.sqlite3`; set it to a persistent mounted path when hosting in the cloud.
- `LANGCHAIN_DEBUG=true` enables verbose LangChain debug output.

### Docker Deployment

Create a `.env` file from `.sample_env` before starting Docker:

```bash
cp .sample_env .env
```

Set at least:

```env
GROQ_API_KEY=your_groq_key
GROQ_MODEL_FALLBACKS=openai/gpt-oss-120b,llama-3.3-70b-versatile,llama-3.1-8b-instant,meta-llama/llama-4-maverick-17b-128e-instruct,meta-llama/llama-4-scout-17b-16e-instruct,qwen/qwen3-32b,deepseek-r1-distill-llama-70b,gemma2-9b-it
GROQ_AUTO_MODEL_FALLBACKS=true
GROQ_EDITOR_API_KEY=your_second_groq_key_or_leave_empty_to_reuse_main_key
SHASTRA_DB_PATH=/app/data/shastra_ai.sqlite3
FRONTEND_PORT=8080
```

Then run:

```bash
docker compose up -d --build
```

The Docker setup runs:

- `backend`: FastAPI on port `8000` inside the Docker network.
- `frontend`: Nginx on public port `80`, serving the React build and proxying `/api` to the backend.
- `shastra_data`: Docker volume for SQLite chat history.
- `shastra_generated_projects`: Docker volume for generated project files.

For local Docker, open `http://localhost:8080`.

For EC2, set `FRONTEND_PORT=80` in `.env` and open inbound HTTP port `80` in the security group. Add HTTPS later with an external reverse proxy or by extending the Nginx config with Certbot.

### Example Prompts

- Create a to-do list application using HTML, CSS, and JavaScript.
- Create a simple calculator web application.
- Create a simple blog API in FastAPI with a SQLite database.

---

Copyright (c) Codebasics Inc. All rights reserved.
