import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from langchain.globals import set_debug, set_verbose
from langchain_groq import ChatGroq
from langgraph.constants import END
from langgraph.graph import StateGraph
from pydantic import BaseModel, Field

from agent.prompts import (
    architect_prompt,
    coder_system_prompt,
    debugger_system_prompt,
    planner_prompt,
)
from agent.states import CoderState, Plan, TaskPlan, TestResult
from agent.tools import (
    get_project_root,
    list_files,
    read_file,
    run_shell_command,
    safe_path_for_project,
    set_project_root,
    write_file,
)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

PROJECT_DIR = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_DIR / ".env", encoding="utf-8-sig")

debug_enabled = os.getenv("LANGCHAIN_DEBUG", "false").lower() == "true"
set_debug(debug_enabled)
set_verbose(debug_enabled)


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


def make_llm(model_env_name: str) -> ChatGroq:
    model = os.getenv(model_env_name) or os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")

    return ChatGroq(
        model=model,
        temperature=_env_float("GROQ_TEMPERATURE", 0.1),
        max_tokens=_env_int("GROQ_MAX_TOKENS", 4096),
    )


planner_llm = make_llm("GROQ_PLANNER_MODEL")
architect_llm = make_llm("GROQ_ARCHITECT_MODEL")
coder_llm = make_llm("GROQ_CODER_MODEL")
debugger_llm = make_llm("GROQ_DEBUGGER_MODEL")


class GeneratedFile(BaseModel):
    path: str = Field(description="The project-relative path of the file to write")
    content: str = Field(description="The complete final file content")


class DebugFileFix(BaseModel):
    path: str = Field(description="The project-relative path of the file to fix")
    content: str = Field(description="The complete corrected file content")
    reason: str = Field(description="Short explanation of why this file needs the fix")


class DebugFixes(BaseModel):
    fixes: list[DebugFileFix] = Field(description="Files that should be rewritten to fix validation")


def _message_text(message) -> str:
    content = getattr(message, "content", message)

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                parts.append(str(item.get("text") or item.get("content") or ""))
            else:
                parts.append(str(item))
        return "\n".join(parts)

    return str(content)


def _extract_json(text: str) -> str:
    text = text.strip()

    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        text = text.strip()

    start_positions = [idx for idx in (text.find("{"), text.find("[")) if idx != -1]
    if not start_positions:
        return text

    start = min(start_positions)
    end = max(text.rfind("}"), text.rfind("]"))

    if end == -1 or end < start:
        return text

    return text[start : end + 1]


def _is_optional_html_reference(ref: str) -> bool:
    normalized = ref.split("?")[0].split("#")[0].strip().lower().lstrip("/")
    return normalized in {
        "favicon.ico",
        "apple-touch-icon.png",
        "site.webmanifest",
        "manifest.json",
    }


def _parse_generated_file_markers(text: str, default_path: str) -> GeneratedFile:
    text = _message_text(text)
    path_match = re.search(
        r"__FILE_PATH__\s*(.*?)\s*__FILE_CONTENT__",
        text,
        flags=re.DOTALL,
    )
    content_match = re.search(
        r"__FILE_CONTENT__\s*(.*?)\s*__END_FILE_CONTENT__",
        text,
        flags=re.DOTALL,
    )

    if not content_match:
        raise ValueError("Fallback coder response did not include file content markers.")

    path = path_match.group(1).strip() if path_match else default_path
    content = content_match.group(1)

    return GeneratedFile(path=path or default_path, content=content)


def _invoke_json_model(llm: ChatGroq, model: type[BaseModel], prompt: str):
    try:
        return invoke_with_retry(
            lambda: llm.with_structured_output(model, method="json_mode").invoke(prompt)
        )
    except Exception as strict_error:
        print(f"Strict JSON mode failed. Retrying with local JSON parsing: {strict_error}")

    fallback_prompt = (
        f"{prompt}\n\n"
        "Important: respond with raw JSON only. "
        "The first character must be { and the last character must be }. "
        "Escape newlines inside string values as \\n. Do not include markdown."
    )

    raw = invoke_with_retry(lambda: llm.invoke(fallback_prompt))
    text = _message_text(raw)

    if not text.strip():
        raise ValueError("Model fallback returned an empty response instead of JSON.")

    try:
        data = json.loads(_extract_json(text))
    except json.JSONDecodeError as e:
        preview = text[:500].replace("\n", "\\n")
        raise ValueError(f"Model fallback returned invalid JSON: {preview}") from e

    return model.model_validate(data)


def _invoke_generated_file(prompt: str, target_path: str) -> GeneratedFile:
    try:
        return invoke_with_retry(
            lambda: coder_llm.with_structured_output(
                GeneratedFile,
                method="json_mode",
            ).invoke(prompt)
        )
    except Exception as strict_error:
        print(f"Strict JSON mode failed. Retrying with marker format: {strict_error}")

    fallback_prompt = (
        f"{prompt}\n\n"
        "For this retry, ignore the previous instruction to return JSON.\n"
        "Return exactly this marker format, with no markdown and no extra prose:\n"
        "__FILE_PATH__\n"
        f"{target_path}\n"
        "__FILE_CONTENT__\n"
        "<complete file content here>\n"
        "__END_FILE_CONTENT__"
    )

    raw = invoke_with_retry(lambda: coder_llm.invoke(fallback_prompt))
    return _parse_generated_file_markers(raw, target_path)


def invoke_with_retry(fn, retries: int = 2):
    """
    Basic retry wrapper for temporary rate limit / transient LLM errors.
    """
    last_error = None

    for attempt in range(retries + 1):
        try:
            return fn()
        except Exception as e:
            last_error = e
            text = str(e).lower()

            is_rate_limit = (
                "rate limit" in text
                or "429" in text
                or "too many requests" in text
                or "resource_exhausted" in text
            )

            if not is_rate_limit or attempt >= retries:
                raise

            sleep_seconds = min(20, 5 * (attempt + 1))
            print(f"Rate limit or temporary model error detected. Retrying in {sleep_seconds} seconds...")
            time.sleep(sleep_seconds)

    raise last_error


def planner_agent(state: dict) -> dict:
    """
    Converts user prompt into a structured project Plan.
    """
    user_prompt = state["user_prompt"]

    resp = invoke_with_retry(
        lambda: planner_llm.with_structured_output(Plan, method="json_mode").invoke(
            planner_prompt(user_prompt, Plan.model_json_schema())
        )
    )

    if resp is None:
        raise ValueError("Planner did not return a valid response.")

    project_dir = set_project_root(resp.name)

    print(f"\nProject: {resp.name}")
    print(f"Output directory: {project_dir}")

    return {
        "plan": resp,
        "project_dir": project_dir,
        "status": "PLANNED",
    }


def architect_agent(state: dict) -> dict:
    """
    Creates file-wise implementation tasks from the Plan.
    """
    plan: Plan = state["plan"]

    resp = invoke_with_retry(
        lambda: architect_llm.with_structured_output(TaskPlan, method="json_mode").invoke(
            architect_prompt(
                plan=plan.model_dump_json(),
                output_schema=TaskPlan.model_json_schema(),
            )
        )
    )

    if resp is None:
        raise ValueError("Architect did not return a valid response.")

    resp.plan = plan

    print("\nImplementation steps:")
    for idx, step in enumerate(resp.implementation_steps, start=1):
        print(f"{idx}. {step.filepath} - {step.task_description[:120]}")

    return {
        "task_plan": resp,
        "status": "ARCHITECTED",
    }


def coder_agent(state: dict) -> dict:
    """
    Code-generation agent. Runs once per implementation step.
    """
    coder_state: CoderState | None = state.get("coder_state")

    if coder_state is None:
        coder_state = CoderState(task_plan=state["task_plan"], current_step_idx=0)

    steps = coder_state.task_plan.implementation_steps

    if coder_state.current_step_idx >= len(steps):
        return {
            "coder_state": coder_state,
            "status": "CODE_DONE",
        }

    current_task = steps[coder_state.current_step_idx]

    print(
        f"\nCoding step {coder_state.current_step_idx + 1}/{len(steps)}: "
        f"{current_task.filepath}"
    )

    existing_content = read_file.invoke({"path": current_task.filepath})
    existing_files = list_files.invoke({"directory": "."})

    user_prompt = (
        f"{coder_system_prompt()}\n\n"
        f"Task:\n{current_task.task_description}\n\n"
        f"Target file:\n{current_task.filepath}\n\n"
        f"Existing project files:\n{existing_files}\n\n"
        f"Existing content:\n{existing_content}\n\n"
        "Return only a JSON object with exactly these keys:\n"
        "- path: the target file path\n"
        "- content: the complete final file content\n\n"
        "Do not call tools. Do not include markdown or code fences."
    )

    generated_file = _invoke_generated_file(user_prompt, current_task.filepath)

    if generated_file is None:
        raise ValueError("Coder did not return a valid file.")

    output_path = current_task.filepath
    if generated_file.path and generated_file.path != current_task.filepath:
        print(
            f"Coder returned path {generated_file.path}; "
            f"writing expected task path {current_task.filepath}."
        )

    write_file.invoke({"path": output_path, "content": generated_file.content})

    coder_state.current_step_idx += 1

    return {
        "coder_state": coder_state,
        "status": "CODING",
    }


def _is_external_reference(path: str) -> bool:
    return path.startswith(
        (
            "http://",
            "https://",
            "#",
            "mailto:",
            "tel:",
            "data:",
            "//",
        )
    )


def _check_html_references(root: Path) -> list[str]:
    issues = []

    for html_file in root.rglob("*.html"):
        try:
            text = html_file.read_text(encoding="utf-8", errors="ignore")
        except Exception as e:
            issues.append(f"Could not read {html_file}: {e}")
            continue

        refs = re.findall(r"""(?:src|href)=["']([^"']+)["']""", text)

        for ref in refs:
            if _is_external_reference(ref) or _is_optional_html_reference(ref):
                continue

            clean_ref = ref.split("?")[0].split("#")[0]

            if not clean_ref:
                continue

            target = (html_file.parent / clean_ref).resolve()

            try:
                target.relative_to(root)
            except ValueError:
                issues.append(f"{html_file.name} references file outside project: {ref}")
                continue

            if not target.exists():
                issues.append(f"{html_file.name} references missing file: {ref}")

    return issues


def _check_expected_plan_files(state: dict) -> list[str]:
    issues = []
    plan: Plan | None = state.get("plan")

    if not plan:
        return issues

    for file in plan.files:
        try:
            expected = safe_path_for_project(file.path)
        except Exception as e:
            issues.append(f"Invalid planned file path {file.path}: {e}")
            continue

        if not expected.exists():
            issues.append(f"Planned file was not created: {file.path}")

    return issues


def _check_python_syntax(root: Path) -> tuple[list[str], list[str], str, str]:
    commands_run = []
    issues = []
    stdout_all = ""
    stderr_all = ""

    py_files = [
        p
        for p in root.rglob("*.py")
        if ".venv" not in p.parts
        and "venv" not in p.parts
        and "__pycache__" not in p.parts
    ]

    if not py_files:
        return commands_run, issues, stdout_all, stderr_all

    cmd = f'"{sys.executable}" -m compileall -q .'
    commands_run.append(cmd)

    code, stdout, stderr = run_shell_command(cmd, timeout=60)

    stdout_all += stdout
    stderr_all += stderr

    if code != 0:
        issues.append("Python syntax validation failed. Check stderr.")

    return commands_run, issues, stdout_all, stderr_all


def _check_javascript_syntax(root: Path) -> tuple[list[str], list[str], str, str]:
    commands_run = []
    issues = []
    stdout_all = ""
    stderr_all = ""

    node_path = shutil.which("node")

    if not node_path:
        return commands_run, issues, stdout_all, stderr_all

    js_files = [
        p
        for p in root.rglob("*.js")
        if "node_modules" not in p.parts
    ]

    for js_file in js_files:
        relative_path = str(js_file.relative_to(root))
        cmd = f'node --check "{relative_path}"'
        commands_run.append(cmd)

        code, stdout, stderr = run_shell_command(cmd, timeout=30)

        stdout_all += stdout
        stderr_all += stderr

        if code != 0:
            issues.append(f"JavaScript syntax validation failed for {relative_path}")

    return commands_run, issues, stdout_all, stderr_all


def _check_package_json(root: Path) -> list[str]:
    issues = []
    package_json = root / "package.json"

    if not package_json.exists():
        return issues

    try:
        json.loads(package_json.read_text(encoding="utf-8"))
    except Exception as e:
        issues.append(f"package.json is invalid JSON: {e}")

    return issues


def _collect_project_context(max_chars: int = 40000) -> str:
    files = list_files.invoke({"directory": "."})
    if files == "No files found.":
        return files

    chunks = []
    total = 0

    for path in files.splitlines():
        content = read_file.invoke({"path": path})
        chunk = f"\n--- {path} ---\n{content}\n"

        if total + len(chunk) > max_chars:
            chunks.append("\n--- context truncated ---\n")
            break

        chunks.append(chunk)
        total += len(chunk)

    return "".join(chunks)


def tester_agent(state: dict) -> dict:
    """
    Deterministic basic tester.
    This does not install dependencies.
    It checks:
    - planned files exist
    - HTML references are valid
    - Python syntax
    - JavaScript syntax if Node.js is available
    - package.json validity
    """
    root = get_project_root()

    print(f"\nTesting generated project at: {root}")

    commands_run = []
    issues = []
    stdout_all = ""
    stderr_all = ""

    issues.extend(_check_expected_plan_files(state))
    issues.extend(_check_html_references(root))
    issues.extend(_check_package_json(root))

    py_commands, py_issues, py_stdout, py_stderr = _check_python_syntax(root)
    commands_run.extend(py_commands)
    issues.extend(py_issues)
    stdout_all += py_stdout
    stderr_all += py_stderr

    js_commands, js_issues, js_stdout, js_stderr = _check_javascript_syntax(root)
    commands_run.extend(js_commands)
    issues.extend(js_issues)
    stdout_all += js_stdout
    stderr_all += js_stderr

    passed = len(issues) == 0

    if passed:
        summary = "Basic validation passed."
        print(summary)
    else:
        summary = "Basic validation failed."
        print(summary)
        for issue in issues:
            print(f"- {issue}")

    return {
        "test_result": TestResult(
            passed=passed,
            summary=summary,
            commands_run=commands_run,
            issues=issues,
            stdout=stdout_all,
            stderr=stderr_all,
        ),
        "status": "TEST_PASSED" if passed else "TEST_FAILED",
    }


def debugger_agent(state: dict) -> dict:
    """
    Uses structured LLM output to fix failed validation.
    """
    attempts = state.get("debug_attempts", 0)
    max_attempts = _env_int("MAX_DEBUG_ATTEMPTS", 2)

    if attempts >= max_attempts:
        print("\nMax debug attempts reached. Stopping.")
        return {
            "status": "FAILED",
            "debug_attempts": attempts,
        }

    test_result: TestResult = state["test_result"]

    print(f"\nDebug attempt {attempts + 1}/{max_attempts}")

    project_context = _collect_project_context()

    user_prompt = (
        f"{debugger_system_prompt()}\n\n"
        "The generated project failed validation.\n\n"
        f"Project directory:\n{get_project_root()}\n\n"
        f"Project files and contents:\n{project_context}\n\n"
        f"Validation result:\n{test_result.model_dump_json(indent=2)}\n\n"
        "Return only a JSON object with a fixes array. Each fix must include path, content, and reason.\n\n"
        "Include only files that need changes. Each content value must be the complete corrected file."
    )

    try:
        debug_fixes = _invoke_json_model(debugger_llm, DebugFixes, user_prompt)
    except Exception as e:
        print(f"Debugger could not produce valid fixes: {e}")
        return {
            "status": "FAILED",
            "debug_attempts": attempts + 1,
            "test_result": test_result.model_copy(
                update={
                    "issues": [
                        *test_result.issues,
                        f"Debugger could not produce valid fixes: {e}",
                    ]
                }
            ),
        }

    if debug_fixes is None:
        raise ValueError("Debugger did not return valid fixes.")

    for fix in debug_fixes.fixes:
        print(f"Applying fix: {fix.path} - {fix.reason[:100]}")
        write_file.invoke({"path": fix.path, "content": fix.content})

    return {
        "status": "DEBUGGED",
        "debug_attempts": attempts + 1,
    }


def route_after_coder(state: dict) -> str:
    if state.get("status") == "CODE_DONE":
        return "tester"
    return "coder"


def route_after_tester(state: dict) -> str:
    if state.get("status") == "TEST_PASSED":
        return "end"
    return "debugger"


def route_after_debugger(state: dict) -> str:
    if state.get("status") == "FAILED":
        return "end"
    return "tester"


graph = StateGraph(dict)

graph.add_node("planner", planner_agent)
graph.add_node("architect", architect_agent)
graph.add_node("coder", coder_agent)
graph.add_node("tester", tester_agent)
graph.add_node("debugger", debugger_agent)

graph.set_entry_point("planner")

graph.add_edge("planner", "architect")
graph.add_edge("architect", "coder")

graph.add_conditional_edges(
    "coder",
    route_after_coder,
    {
        "coder": "coder",
        "tester": "tester",
    },
)

graph.add_conditional_edges(
    "tester",
    route_after_tester,
    {
        "debugger": "debugger",
        "end": END,
    },
)

graph.add_conditional_edges(
    "debugger",
    route_after_debugger,
    {
        "tester": "tester",
        "end": END,
    },
)

agent = graph.compile()


if __name__ == "__main__":
    result = agent.invoke(
        {"user_prompt": "Build a colourful modern todo app in HTML, CSS and JavaScript"},
        {"recursion_limit": 150},
    )
    print("Final State:", result)
