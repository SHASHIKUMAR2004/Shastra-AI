import io
import json
import os
import re
import shutil
import sqlite3
import subprocess
import threading
import traceback
import zipfile
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from agent.groq_fallback import available_groq_models, make_groq_llm, resolve_groq_models
from agent.graph import _check_html_references, _extract_json
from agent.graph import agent

APP_ROOT = Path(__file__).resolve().parent
GENERATED_PROJECTS_DIR = APP_ROOT / "generated_projects"
DATABASE_PATH = Path(os.getenv("SHASTRA_DB_PATH", "shastra_ai.sqlite3"))
if not DATABASE_PATH.is_absolute():
    DATABASE_PATH = APP_ROOT / DATABASE_PATH


class AttachmentInput(BaseModel):
    name: str = Field(..., min_length=1, max_length=260)
    mime_type: str = ""
    size: int = Field(0, ge=0)
    content_text: str | None = Field(default=None, max_length=30000)


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=3, max_length=12000)
    recursion_limit: int = Field(150, ge=20, le=500)
    attachments: list[AttachmentInput] = Field(default_factory=list)


class GenerateResponse(BaseModel):
    job_id: str
    status: str


class JobSnapshot(BaseModel):
    id: str
    prompt: str
    status: Literal["queued", "running", "succeeded", "failed"]
    created_at: datetime
    updated_at: datetime
    project_dir: str | None = None
    final_status: str | None = None
    test_summary: str | None = None
    issues: list[str] = Field(default_factory=list)
    logs: str = ""
    error: str | None = None


class ProjectSummary(BaseModel):
    name: str
    path: str
    updated_at: datetime
    file_count: int


class FileSummary(BaseModel):
    path: str
    size: int
    updated_at: datetime


class ProjectFilesResponse(BaseModel):
    project: ProjectSummary
    files: list[FileSummary]


class ProjectChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=12000)
    attachments: list[AttachmentInput] = Field(default_factory=list)


class ProjectChatMessage(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    role: Literal["user", "assistant"]
    content: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ProjectFileEdit(BaseModel):
    path: str
    content: str
    reason: str = ""


class ProjectEditPlan(BaseModel):
    reply: str
    changes: list[ProjectFileEdit] = Field(default_factory=list)


class ProjectValidationResult(BaseModel):
    passed: bool
    summary: str
    issues: list[str] = Field(default_factory=list)
    commands_run: list[str] = Field(default_factory=list)
    stdout: str = ""
    stderr: str = ""


class ProjectChatResponse(BaseModel):
    message: ProjectChatMessage
    history: list[ProjectChatMessage]
    changed_files: list[str] = Field(default_factory=list)
    validation: ProjectValidationResult


class GroqModelsResponse(BaseModel):
    available_models: list[str]
    generation_models: list[str]
    editor_models: list[str]


app = FastAPI(title="Shastra AI API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1):\d+",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

jobs: dict[str, JobSnapshot] = {}
jobs_lock = threading.Lock()
generation_lock = threading.Lock()
editor_lock = threading.Lock()
db_lock = threading.Lock()
IGNORED_PROJECT_PARTS = {"__pycache__", "node_modules", ".git", ".venv", "venv"}


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _init_database() -> None:
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DATABASE_PATH) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS project_chat_messages (
                id TEXT PRIMARY KEY,
                project_name TEXT NOT NULL,
                role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
                content TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_project_chat_messages_project_created
            ON project_chat_messages (project_name, created_at)
            """
        )


def _load_project_chat(project_name: str) -> list[ProjectChatMessage]:
    with db_lock, sqlite3.connect(DATABASE_PATH) as connection:
        rows = connection.execute(
            """
            SELECT id, role, content, created_at
            FROM project_chat_messages
            WHERE project_name = ?
            ORDER BY created_at ASC, rowid ASC
            """,
            (project_name,),
        ).fetchall()

    return [
        ProjectChatMessage(
            id=row[0],
            role=row[1],
            content=row[2],
            created_at=datetime.fromisoformat(row[3]),
        )
        for row in rows
    ]


def _save_project_chat_message(project_name: str, message: ProjectChatMessage) -> None:
    with db_lock, sqlite3.connect(DATABASE_PATH) as connection:
        connection.execute(
            """
            INSERT OR REPLACE INTO project_chat_messages
                (id, project_name, role, content, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                message.id,
                project_name,
                message.role,
                message.content,
                message.created_at.isoformat(),
            ),
        )


_init_database()


def _jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    return value


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except ValueError:
        return default


def _set_job(job_id: str, **updates: Any) -> None:
    with jobs_lock:
        current = jobs[job_id]
        jobs[job_id] = current.model_copy(
            update={
                **updates,
                "updated_at": _now(),
            }
        )


def _summarize_project(project_dir: Path) -> ProjectSummary:
    files = [
        path
        for path in project_dir.rglob("*")
        if path.is_file()
        and not any(part in IGNORED_PROJECT_PARTS for part in path.parts)
    ]
    updated_at = max(
        (datetime.fromtimestamp(path.stat().st_mtime, timezone.utc) for path in files),
        default=datetime.fromtimestamp(project_dir.stat().st_mtime, timezone.utc),
    )
    return ProjectSummary(
        name=project_dir.name,
        path=str(project_dir),
        updated_at=updated_at,
        file_count=len(files),
    )


def _safe_project_dir(project_name: str) -> Path:
    project_dir = (GENERATED_PROJECTS_DIR / project_name).resolve()
    try:
        project_dir.relative_to(GENERATED_PROJECTS_DIR.resolve())
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid project name")

    if not project_dir.exists() or not project_dir.is_dir():
        raise HTTPException(status_code=404, detail="Project not found")

    return project_dir


def _is_ignored_path(path: Path) -> bool:
    return any(part in IGNORED_PROJECT_PARTS for part in path.parts)


def _safe_project_file(project_dir: Path, relative_path: str) -> Path:
    target = (project_dir / relative_path).resolve()
    try:
        target.relative_to(project_dir)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid edit path: {relative_path}")
    return target


def _build_generation_prompt(prompt: str, attachments: list[AttachmentInput]) -> str:
    if not attachments:
        return prompt

    chunks = [
        prompt,
        "\n\nAttached context from the user:",
        "Use these files as requirements, references, assets, or source material when generating the project.",
    ]
    total_chars = 0
    max_total_chars = 60000

    for index, attachment in enumerate(attachments, start=1):
        header = (
            f"\n\n--- Attachment {index}: {attachment.name} ---\n"
            f"MIME type: {attachment.mime_type or 'unknown'}\n"
            f"Size: {attachment.size} bytes\n"
        )
        chunks.append(header)

        content = (attachment.content_text or "").strip()
        if content:
            remaining = max_total_chars - total_chars
            if remaining <= 0:
                chunks.append("Content omitted because the attachment context limit was reached.\n")
                continue

            clipped = content[:remaining]
            total_chars += len(clipped)
            chunks.append(f"Content:\n{clipped}\n")
            if len(content) > len(clipped):
                chunks.append("\n[Attachment content truncated]\n")
        elif attachment.mime_type.startswith("image/"):
            chunks.append(
                "Image/screenshot attached. No OCR text was extracted yet; use the file name and prompt context.\n"
            )
        else:
            chunks.append("Binary file attached. No text content was extracted.\n")

    return "".join(chunks)


def _make_editor_llm():
    return make_groq_llm(
        model_env_name="GROQ_EDITOR_MODEL",
        fallback_env_name="GROQ_EDITOR_MODEL_FALLBACKS",
        api_key_env_names=("GROQ_EDITOR_API_KEY", "GROQ_API_KEY"),
        temperature=_env_float("GROQ_EDITOR_TEMPERATURE", _env_float("GROQ_TEMPERATURE", 0.1)),
        max_tokens=_env_int("GROQ_EDITOR_MAX_TOKENS", _env_int("GROQ_MAX_TOKENS", 4096)),
    )


def _message_text(message: Any) -> str:
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(str(item.get("text") or item.get("content") or item) for item in content)
    return str(content)


def _invoke_editor(prompt: str) -> ProjectEditPlan:
    llm = _make_editor_llm()
    try:
        return llm.with_structured_output(ProjectEditPlan, method="json_mode").invoke(prompt)
    except Exception as strict_error:
        print(f"Editor strict JSON mode failed. Retrying with local JSON parsing: {strict_error}")

    raw = llm.invoke(
        f"{prompt}\n\n"
        "Return raw JSON only. The JSON must match this shape exactly:\n"
        '{"reply":"short user-facing summary","changes":[{"path":"relative/file/path","content":"complete file content","reason":"why"}]}'
    )
    text = _message_text(raw)
    if not text.strip():
        raise ValueError("Editor model returned an empty response.")
    try:
        data = json.loads(_extract_json(text))
    except json.JSONDecodeError as e:
        preview = text[:500].replace("\n", "\\n")
        raise ValueError(f"Editor model returned invalid JSON: {preview}") from e
    return ProjectEditPlan.model_validate(data)


def _collect_project_context(project_dir: Path, max_chars: int = 70000) -> tuple[str, str]:
    files = [
        path
        for path in project_dir.rglob("*")
        if path.is_file() and not _is_ignored_path(path.relative_to(project_dir))
    ]
    files.sort(key=lambda path: str(path.relative_to(project_dir)))
    tree = "\n".join(str(path.relative_to(project_dir)) for path in files) or "No files found."

    chunks = []
    total = 0
    text_suffixes = {
        ".bat",
        ".c",
        ".cpp",
        ".cs",
        ".css",
        ".env",
        ".go",
        ".html",
        ".java",
        ".js",
        ".json",
        ".jsx",
        ".md",
        ".py",
        ".rs",
        ".sql",
        ".ts",
        ".tsx",
        ".txt",
        ".xml",
        ".yaml",
        ".yml",
    }

    for path in files:
        relative = path.relative_to(project_dir)
        if path.suffix.lower() not in text_suffixes:
            continue
        try:
            content = path.read_text(encoding="utf-8", errors="ignore")
        except Exception as e:
            content = f"[Could not read file: {e}]"

        chunk = f"\n--- {relative} ---\n{content}\n"
        if total + len(chunk) > max_chars:
            chunks.append("\n--- context truncated ---\n")
            break
        chunks.append(chunk)
        total += len(chunk)

    return tree, "".join(chunks) or "No readable text files found."


def _validate_project(project_dir: Path) -> ProjectValidationResult:
    commands_run: list[str] = []
    issues: list[str] = []
    stdout_all = ""
    stderr_all = ""

    issues.extend(_check_html_references(project_dir))

    package_json = project_dir / "package.json"
    if package_json.exists():
        try:
            json.loads(package_json.read_text(encoding="utf-8"))
        except Exception as e:
            issues.append(f"package.json is invalid JSON: {e}")

    py_files = [
        path
        for path in project_dir.rglob("*.py")
        if not _is_ignored_path(path.relative_to(project_dir))
    ]
    if py_files:
        cmd = f'"{os.sys.executable}" -m compileall -q .'
        commands_run.append(cmd)
        result = subprocess.run(
            cmd,
            shell=True,
            cwd=str(project_dir),
            capture_output=True,
            text=True,
            timeout=60,
        )
        stdout_all += result.stdout[-5000:] if result.stdout else ""
        stderr_all += result.stderr[-5000:] if result.stderr else ""
        if result.returncode != 0:
            issues.append("Python syntax validation failed. Check stderr.")

    node_path = shutil.which("node")
    if node_path:
        for js_file in project_dir.rglob("*.js"):
            if _is_ignored_path(js_file.relative_to(project_dir)):
                continue
            relative = str(js_file.relative_to(project_dir))
            cmd = f'node --check "{relative}"'
            commands_run.append(cmd)
            result = subprocess.run(
                cmd,
                shell=True,
                cwd=str(project_dir),
                capture_output=True,
                text=True,
                timeout=30,
            )
            stdout_all += result.stdout[-5000:] if result.stdout else ""
            stderr_all += result.stderr[-5000:] if result.stderr else ""
            if result.returncode != 0:
                issues.append(f"JavaScript syntax validation failed for {relative}")

    passed = len(issues) == 0
    return ProjectValidationResult(
        passed=passed,
        summary="Validation passed." if passed else "Validation failed.",
        issues=issues,
        commands_run=commands_run,
        stdout=stdout_all,
        stderr=stderr_all,
    )


def _build_editor_prompt(
    project_name: str,
    user_message: str,
    attachments: list[AttachmentInput],
    history: list[ProjectChatMessage],
    tree: str,
    context: str,
) -> str:
    attachment_context = _build_generation_prompt("", attachments).strip()
    recent_history = "\n".join(
        f"{message.role}: {message.content}" for message in history[-8:]
    )
    return f"""
You are the Shastra AI project editor.

You are editing an existing generated project named {project_name}.

User request:
{user_message}

Recent project chat:
{recent_history or "No previous chat."}

Project file tree:
{tree}

Project file contents:
{context}

{attachment_context}

Rules:
- Modify only files needed to satisfy the user's request.
- Preserve working behavior and existing project structure.
- Return complete file contents for each changed file.
- Do not edit files outside the project.
- If no file changes are needed, return an empty changes array and explain why.
- In the reply, explain the result conversationally like a coding assistant, not like a terse build log.
- Keep the reply concise, user-facing, and focused on what changed or what blocked the change.

Return only JSON matching this schema:
{ProjectEditPlan.model_json_schema()}
"""


def _run_generation(job_id: str, prompt: str, recursion_limit: int) -> None:
    _set_job(job_id, status="running")
    stdout = io.StringIO()

    try:
        with generation_lock:
            with redirect_stdout(stdout), redirect_stderr(stdout):
                result = agent.invoke(
                    {"user_prompt": prompt},
                    {"recursion_limit": recursion_limit},
                )

        result = _jsonable(result)
        test_result = result.get("test_result") or {}
        final_status = result.get("status")
        api_status = "succeeded" if final_status == "TEST_PASSED" else "failed"
        _set_job(
            job_id,
            status=api_status,
            project_dir=result.get("project_dir"),
            final_status=final_status,
            test_summary=test_result.get("summary"),
            issues=test_result.get("issues") or [],
            logs=stdout.getvalue()[-20000:],
            error=None if api_status == "succeeded" else "Generation finished with validation issues.",
        )
    except Exception as exc:
        _set_job(
            job_id,
            status="failed",
            error=str(exc),
            logs=(stdout.getvalue() + "\n" + traceback.format_exc())[-20000:],
        )


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/models/groq", response_model=GroqModelsResponse)
def groq_models() -> GroqModelsResponse:
    return GroqModelsResponse(
        available_models=available_groq_models(),
        generation_models=resolve_groq_models(model_env_name="GROQ_MODEL"),
        editor_models=resolve_groq_models(
            model_env_name="GROQ_EDITOR_MODEL",
            fallback_env_name="GROQ_EDITOR_MODEL_FALLBACKS",
            api_key_env_names=("GROQ_EDITOR_API_KEY", "GROQ_API_KEY"),
        ),
    )


@app.post("/api/generations", response_model=GenerateResponse)
def create_generation(request: GenerateRequest) -> GenerateResponse:
    job_id = str(uuid4())
    generation_prompt = _build_generation_prompt(request.prompt, request.attachments)
    with jobs_lock:
        jobs[job_id] = JobSnapshot(
            id=job_id,
            prompt=request.prompt,
            status="queued",
            created_at=_now(),
            updated_at=_now(),
        )

    thread = threading.Thread(
        target=_run_generation,
        args=(job_id, generation_prompt, request.recursion_limit),
        daemon=True,
    )
    thread.start()
    return GenerateResponse(job_id=job_id, status="queued")


@app.get("/api/generations", response_model=list[JobSnapshot])
def list_generations() -> list[JobSnapshot]:
    with jobs_lock:
        return sorted(jobs.values(), key=lambda job: job.created_at, reverse=True)


@app.get("/api/generations/{job_id}", response_model=JobSnapshot)
def get_generation(job_id: str) -> JobSnapshot:
    with jobs_lock:
        job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Generation job not found")
    return job


@app.get("/api/projects", response_model=list[ProjectSummary])
def list_projects() -> list[ProjectSummary]:
    if not GENERATED_PROJECTS_DIR.exists():
        return []

    projects = [
        _summarize_project(path)
        for path in GENERATED_PROJECTS_DIR.iterdir()
        if path.is_dir()
    ]
    return sorted(projects, key=lambda project: project.updated_at, reverse=True)


@app.get("/api/projects/{project_name}/files", response_model=ProjectFilesResponse)
def list_project_files(project_name: str) -> ProjectFilesResponse:
    project_dir = _safe_project_dir(project_name)

    files = []
    for path in project_dir.rglob("*"):
        if not path.is_file():
            continue
        if any(part in IGNORED_PROJECT_PARTS for part in path.parts):
            continue
        files.append(
            FileSummary(
                path=str(path.relative_to(project_dir)),
                size=path.stat().st_size,
                updated_at=datetime.fromtimestamp(path.stat().st_mtime, timezone.utc),
            )
        )

    files.sort(key=lambda file: file.path)
    return ProjectFilesResponse(project=_summarize_project(project_dir), files=files)


@app.get("/api/projects/{project_name}/chat", response_model=list[ProjectChatMessage])
def get_project_chat(project_name: str) -> list[ProjectChatMessage]:
    _safe_project_dir(project_name)
    return _load_project_chat(project_name)


@app.post("/api/projects/{project_name}/chat", response_model=ProjectChatResponse)
def chat_with_project(project_name: str, request: ProjectChatRequest) -> ProjectChatResponse:
    project_dir = _safe_project_dir(project_name)
    user_message = ProjectChatMessage(role="user", content=request.message)
    history = _load_project_chat(project_name)
    history.append(user_message)
    _save_project_chat_message(project_name, user_message)

    try:
        tree, context = _collect_project_context(project_dir)
        prompt = _build_editor_prompt(
            project_name=project_name,
            user_message=request.message,
            attachments=request.attachments,
            history=history,
            tree=tree,
            context=context,
        )

        with editor_lock:
            edit_plan = _invoke_editor(prompt)

        changed_files = []
        for change in edit_plan.changes:
            target = _safe_project_file(project_dir, change.path)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(change.content, encoding="utf-8")
            changed_files.append(str(target.relative_to(project_dir)))

        validation = _validate_project(project_dir)
        reply_sections = [edit_plan.reply.strip() or "I updated the project."]
        if changed_files:
            files = "\n".join(f"- {path}" for path in changed_files)
            reply_sections.append(f"Updated files:\n{files}")
        else:
            reply_sections.append("Updated files:\n- No file changes were needed.")

        reply_sections.append(f"Validation:\n{validation.summary}")
        if validation.issues:
            issues = "\n".join(f"- {issue}" for issue in validation.issues[:4])
            reply_sections.append(f"Needs attention:\n{issues}")

        assistant_message = ProjectChatMessage(role="assistant", content="\n\n".join(reply_sections))
        history.append(assistant_message)
        _save_project_chat_message(project_name, assistant_message)

        return ProjectChatResponse(
            message=assistant_message,
            history=_load_project_chat(project_name),
            changed_files=changed_files,
            validation=validation,
        )
    except Exception as exc:
        validation = ProjectValidationResult(
            passed=False,
            summary="Project edit failed.",
            issues=[str(exc)],
        )
        assistant_message = ProjectChatMessage(
            role="assistant",
            content=f"I could not update the project: {exc}",
        )
        history.append(assistant_message)
        _save_project_chat_message(project_name, assistant_message)
        return ProjectChatResponse(
            message=assistant_message,
            history=_load_project_chat(project_name),
            changed_files=[],
            validation=validation,
        )


@app.get("/api/projects/{project_name}/download")
def download_project(project_name: str) -> StreamingResponse:
    project_dir = _safe_project_dir(project_name)
    archive = io.BytesIO()

    with zipfile.ZipFile(archive, mode="w", compression=zipfile.ZIP_DEFLATED) as zip_file:
        for path in project_dir.rglob("*"):
            if not path.is_file():
                continue
            if any(part in IGNORED_PROJECT_PARTS for part in path.parts):
                continue

            relative_path = path.relative_to(project_dir)
            zip_file.write(path, arcname=str(Path(project_dir.name) / relative_path))

    archive.seek(0)
    headers = {
        "Content-Disposition": f'attachment; filename="{project_dir.name}.zip"',
    }
    return StreamingResponse(archive, media_type="application/zip", headers=headers)
