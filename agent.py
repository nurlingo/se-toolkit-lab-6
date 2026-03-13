#!/usr/bin/env python3
"""CLI agent that answers questions using an LLM with tool calling."""

import json
import os
import sys
from pathlib import Path

import httpx


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def _load_env(filepath: str) -> None:
    """Load key=value pairs from a file into os.environ."""
    p = Path(filepath)
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


_load_env(".env.agent.secret")
_load_env(".env.docker.secret")

LLM_API_KEY = os.environ.get("LLM_API_KEY", "")
LLM_API_BASE = os.environ.get("LLM_API_BASE", "https://openrouter.ai/api/v1")
LLM_MODEL = os.environ.get("LLM_MODEL", "meta-llama/llama-4-scout:free")
LMS_API_KEY = os.environ.get("LMS_API_KEY", "")
AGENT_API_BASE_URL = os.environ.get("AGENT_API_BASE_URL", "http://localhost:42002")

PROJECT_ROOT = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

def tool_read_file(path: str) -> str:
    """Read a file from the project directory."""
    resolved = (PROJECT_ROOT / path).resolve()
    if not str(resolved).startswith(str(PROJECT_ROOT)):
        return "Error: path traversal not allowed"
    if not resolved.is_file():
        return f"Error: file not found: {path}"
    try:
        return resolved.read_text(encoding="utf-8", errors="replace")[:30000]
    except Exception as e:
        return f"Error reading file: {e}"


def tool_list_files(path: str) -> str:
    """List files and directories at a given path."""
    resolved = (PROJECT_ROOT / path).resolve()
    if not str(resolved).startswith(str(PROJECT_ROOT)):
        return "Error: path traversal not allowed"
    if not resolved.is_dir():
        return f"Error: directory not found: {path}"
    try:
        entries = sorted(p.name + ("/" if p.is_dir() else "") for p in resolved.iterdir() if not p.name.startswith("."))
        return "\n".join(entries)
    except Exception as e:
        return f"Error listing directory: {e}"


def tool_query_api(method: str, path: str, body: str | None = None, authenticated: bool = True) -> str:
    """Query the backend API."""
    url = f"{AGENT_API_BASE_URL}{path}"
    headers = {}
    if authenticated and LMS_API_KEY:
        headers["Authorization"] = f"Bearer {LMS_API_KEY}"
    try:
        with httpx.Client(timeout=15) as client:
            resp = client.request(method, url, headers=headers, content=body if body else None)
            return json.dumps({"status_code": resp.status_code, "body": resp.text[:5000]})
    except Exception as e:
        return json.dumps({"status_code": 0, "body": f"Error: {e}"})


TOOLS = {
    "read_file": tool_read_file,
    "list_files": tool_list_files,
    "query_api": tool_query_api,
}

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file from the project repository. Returns file contents as a string.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Relative path from project root (e.g., 'wiki/git-workflow.md')"}
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List files and directories at a given path in the project repository.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Relative directory path from project root (e.g., 'wiki' or 'backend')"}
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_api",
            "description": "Send an HTTP request to the deployed backend LMS API. Use this for data queries (item counts, scores, analytics) and system status checks. The API is at the base URL with endpoints like /items/, /analytics/completion-rate, etc. By default requests are authenticated with the API key. Set authenticated=false to test unauthenticated behavior.",
            "parameters": {
                "type": "object",
                "properties": {
                    "method": {"type": "string", "description": "HTTP method (GET, POST, PUT, DELETE)", "enum": ["GET", "POST", "PUT", "DELETE"]},
                    "path": {"type": "string", "description": "API endpoint path (e.g., '/items/', '/analytics/completion-rate')"},
                    "body": {"type": "string", "description": "Optional JSON request body"},
                    "authenticated": {"type": "boolean", "description": "Whether to send the API key. Default true. Set false to test unauthenticated requests.", "default": True},
                },
                "required": ["method", "path"],
            },
        },
    },
]

SYSTEM_PROMPT = """\
You are a helpful agent for a Learning Management Service project. You answer questions by using your tools to explore the codebase and query the backend API.

Strategy:
- For questions about project documentation, wiki, or how-to guides: use list_files to discover files in wiki/, then read_file to find the answer. Always include a source reference.
- For questions about the codebase, architecture, or what framework/technology is used: use read_file on relevant source files (e.g., backend/app/, pyproject.toml, docker-compose.yml).
- For questions about data (item counts, scores, analytics, learners): use query_api to call the backend API.
- For questions about bugs or errors: first query_api to reproduce the error, then read_file on the relevant source code to diagnose.
- For API endpoints with query parameters, include them in the path (e.g., '/analytics/completion-rate?lab=lab-04').
- To explore available API endpoints, read_file on 'backend/app/routers/' directory or use list_files('backend/app/routers').
- If you need to explore the project structure, start with list_files on the root or relevant subdirectory.

Important rules:
- Always use tools to find the real answer. Never guess or hallucinate.
- For wiki/documentation questions, set "source" to the file path and section anchor (e.g., "wiki/git-workflow.md#resolving-merge-conflicts").
- Be precise and concise in your answers.
- When reporting numbers from the API, state the exact number.
- NEVER read the same file twice. If you already read a file, use the content you have.
- If a file references another file, read that other file to find the answer.
- Give your final answer as soon as you have enough information. Do not keep searching.
"""


# ---------------------------------------------------------------------------
# LLM interaction
# ---------------------------------------------------------------------------

def _call_llm(messages: list[dict]) -> dict:
    """Call the LLM and return the response message."""
    url = f"{LLM_API_BASE}/chat/completions"
    headers = {
        "Authorization": f"Bearer {LLM_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": LLM_MODEL,
        "messages": messages,
        "tools": TOOL_SCHEMAS,
        "temperature": 0,
    }
    with httpx.Client(timeout=60) as client:
        resp = client.post(url, headers=headers, json=payload)
        resp.raise_for_status()
        data = resp.json()
    return data["choices"][0]["message"]


def _execute_tool(name: str, arguments: dict) -> str:
    """Execute a tool by name with the given arguments."""
    fn = TOOLS.get(name)
    if not fn:
        return f"Error: unknown tool '{name}'"
    try:
        return fn(**arguments)
    except TypeError as e:
        return f"Error calling {name}: {e}"


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------

def _extract_source(answer: str, tool_calls: list[dict]) -> str | None:
    """Pick the best source file from tool calls.

    Heuristic: prefer wiki/ files mentioned in the answer, then fall back
    to the last read_file path, then None.
    """
    read_paths = [
        tc["args"].get("path", "")
        for tc in tool_calls
        if tc["tool"] == "read_file" and tc["args"].get("path")
    ]
    if not read_paths:
        return None
    # If the answer mentions a specific file path, prefer that
    answer_lower = answer.lower()
    for p in read_paths:
        if p.lower() in answer_lower:
            return p
    # Prefer wiki files for documentation questions
    wiki_paths = [p for p in read_paths if p.startswith("wiki/")]
    if wiki_paths:
        return wiki_paths[-1]
    return read_paths[-1]


def run_agent(question: str) -> dict:
    """Run the agentic loop and return the final JSON output."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    all_tool_calls = []
    max_iterations = 10

    for i in range(max_iterations):
        print(f"[iteration {i+1}]", file=sys.stderr)
        response_msg = _call_llm(messages)

        tool_calls = response_msg.get("tool_calls")
        if not tool_calls:
            # Final answer
            answer = (response_msg.get("content") or "").strip()
            # Extract source: prefer the most relevant read_file path
            # (wiki file for wiki questions, last read_file otherwise)
            source = _extract_source(answer, all_tool_calls)
            # Strip bulky result text — checker only needs tool names
            compact_calls = [
                {"tool": tc["tool"], "args": tc["args"]}
                for tc in all_tool_calls
            ]
            return {
                "answer": answer,
                "source": source,
                "tool_calls": compact_calls,
            }

        # Process tool calls
        messages.append(response_msg)
        for tc in tool_calls:
            fn_name = tc["function"]["name"]
            try:
                fn_args = json.loads(tc["function"]["arguments"])
            except json.JSONDecodeError:
                fn_args = {}

            print(f"  tool: {fn_name}({fn_args})", file=sys.stderr)
            result = _execute_tool(fn_name, fn_args)

            all_tool_calls.append({
                "tool": fn_name,
                "args": fn_args,
                "result": result[:2000],
            })
            messages.append({
                "role": "tool",
                "tool_call_id": tc["id"],
                "content": result,
            })

    # Max iterations reached
    compact_calls = [
        {"tool": tc["tool"], "args": tc["args"]}
        for tc in all_tool_calls
    ]
    return {
        "answer": "Could not determine answer within iteration limit.",
        "source": None,
        "tool_calls": compact_calls,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: agent.py <question>", file=sys.stderr)
        sys.exit(1)

    question = sys.argv[1]
    try:
        result = run_agent(question)
        print(json.dumps(result))
    except Exception as e:
        print(f"Agent error: {e}", file=sys.stderr)
        sys.exit(1)
