import os
import pathlib
import re
import subprocess
from typing import Optional, Tuple

from langchain_core.tools import tool


def _slugify(name: str) -> str:
    name = name.strip().lower()
    name = re.sub(r"[^a-z0-9]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    return name or "generated_project"


PROJECT_ROOT = pathlib.Path.cwd() / "generated_project"


def set_project_root(project_name: str) -> str:
    """
    Creates a separate output folder for each generated project.
    Example:
    generated_projects/calculator_app/
    generated_projects/todo_app/
    """
    global PROJECT_ROOT

    base_dir = pathlib.Path(os.getenv("PROJECTS_BASE_DIR", "generated_projects")).resolve()
    base_dir.mkdir(parents=True, exist_ok=True)

    project_dir = (base_dir / _slugify(project_name)).resolve()

    if project_dir != base_dir and base_dir not in project_dir.parents:
        raise ValueError("Invalid project directory")

    PROJECT_ROOT = project_dir
    PROJECT_ROOT.mkdir(parents=True, exist_ok=True)
    return str(PROJECT_ROOT)


def get_project_root() -> pathlib.Path:
    return PROJECT_ROOT


def safe_path_for_project(path: Optional[str] = ".") -> pathlib.Path:
    if path is None or path == "":
        path = "."

    root = PROJECT_ROOT.resolve()
    p = (root / path).resolve()

    if p != root and root not in p.parents:
        raise ValueError("Attempt to access outside project root")

    return p


@tool
def write_file(path: str, content: str) -> str:
    """Writes content to a file at the specified path within the project root."""
    p = safe_path_for_project(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    with open(p, "w", encoding="utf-8") as f:
        f.write(content)

    return f"WROTE: {p}"


@tool
def read_file(path: str) -> str:
    """Reads content from a file at the specified path within the project root."""
    p = safe_path_for_project(path)

    if not p.exists():
        return ""

    if p.is_dir():
        return f"ERROR: {p} is a directory"

    with open(p, "r", encoding="utf-8") as f:
        return f.read()


@tool
def get_current_directory() -> str:
    """Returns the current generated project directory."""
    return str(PROJECT_ROOT)


@tool
def list_files(directory: str = ".") -> str:
    """Lists all files in the specified directory within the project root."""
    p = safe_path_for_project(directory)

    if not p.exists():
        return "No files found."

    if not p.is_dir():
        return f"ERROR: {p} is not a directory"

    ignored_parts = {".git", ".venv", "venv", "__pycache__", "node_modules"}

    files = []
    for f in p.rglob("*"):
        if not f.is_file():
            continue

        if any(part in ignored_parts for part in f.parts):
            continue

        files.append(str(f.relative_to(PROJECT_ROOT)))

    return "\n".join(sorted(files)) if files else "No files found."


def run_shell_command(cmd: str, cwd: Optional[str] = None, timeout: int = 60) -> Tuple[int, str, str]:
    """
    Internal deterministic command runner used by the Tester node.
    We are not directly giving this to the coder agent as a free tool.
    """
    cwd_dir = safe_path_for_project(cwd or ".")

    result = subprocess.run(
        cmd,
        shell=True,
        cwd=str(cwd_dir),
        capture_output=True,
        text=True,
        timeout=timeout,
    )

    stdout = result.stdout[-5000:] if result.stdout else ""
    stderr = result.stderr[-5000:] if result.stderr else ""

    return result.returncode, stdout, stderr


@tool
def run_cmd(cmd: str, cwd: Optional[str] = None, timeout: int = 60) -> str:
    """
    Runs a shell command inside the generated project directory.
    Use carefully. This is available if you later decide to expose it to an agent.
    """
    code, stdout, stderr = run_shell_command(cmd, cwd=cwd, timeout=timeout)
    return f"EXIT_CODE: {code}\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}"


def init_project_root() -> str:
    PROJECT_ROOT.mkdir(parents=True, exist_ok=True)
    return str(PROJECT_ROOT)