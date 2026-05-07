"""
Co-pilot router — POST /api/copilot/chat

Powers the floating AI assistant panel in the Guardrails UI.
Uses the Anthropic SDK with tool use to query live system data (read-only).
"""

import json
import logging
import os
from typing import Any, Optional

import anthropic
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from app.backend import db
from app.backend.auth import UserContext, get_current_user

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/copilot", tags=["copilot"])

COPILOT_MODEL = "claude-opus-4-5"
MAX_TOOL_ITERATIONS = 5
MAX_HISTORY_MESSAGES = 10
MAX_SESSIONS_PER_TOOL_CALL = 20
MAX_AUDIT_LOGS_PER_TOOL_CALL = 50

# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class CopilotMessage(BaseModel):
    role: str
    content: str


class PageContext(BaseModel):
    page: str
    session_id: Optional[str] = None
    extra: Optional[dict] = None


class CopilotRequest(BaseModel):
    message: str
    history: list[CopilotMessage] = []
    context: PageContext


class ToolCallLog(BaseModel):
    tool_name: str
    input: dict
    output: Any


class CopilotResponse(BaseModel):
    reply: str
    tool_calls: list[ToolCallLog] = []
    model: str


# ---------------------------------------------------------------------------
# Tool definitions (sent to Claude)
# ---------------------------------------------------------------------------

TOOLS: list[dict] = [
    {
        "name": "list_sessions",
        "description": (
            "List recent pipeline sessions. Returns session IDs, queries, gateway decisions, "
            "MAD routing decisions, confidence scores, and status."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "description": "Max sessions to return (max 20)",
                    "default": 10,
                },
                "status": {
                    "type": "string",
                    "description": (
                        "Optional filter by pipeline status: GATEWAY_PASSED, GATEWAY_BLOCKED, "
                        "GATEWAY_ESCALATED, HUMAN_REVIEW_REQUIRED, COMPLETED, HARD_BLOCKED, FAILED"
                    ),
                },
            },
        },
    },
    {
        "name": "get_session",
        "description": (
            "Get full details for a specific session including gateway scores, "
            "LLM answer, MAD debate routing, CSE data, and status."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "session_id": {
                    "type": "string",
                    "description": "The session UUID (e.g. dbf93e06-69eb-4cee-bfd4-bd5ea2c3c7c8)",
                },
            },
            "required": ["session_id"],
        },
    },
    {
        "name": "get_analytics",
        "description": (
            "Get aggregate analytics: total sessions, block/escalate/pass counts, "
            "average pipeline latency, MAD routing distribution, human review queue size, "
            "and feedback count."
        ),
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "get_system_health",
        "description": (
            "Get current health status of pipeline components: database, gateway, "
            "LLM runtime, and CSE config. Returns status and latency per component."
        ),
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "get_review_queue",
        "description": (
            "Get all sessions currently awaiting human review (status=HUMAN_REVIEW_REQUIRED), "
            "with gateway scores and MAD routing decisions."
        ),
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "get_audit_logs",
        "description": (
            "Get recent audit log entries showing who did what and when. "
            "Can filter by session ID to see all actions taken on a specific session."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "session_id": {
                    "type": "string",
                    "description": "Optional session ID to filter logs to one session",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max entries to return (max 50)",
                    "default": 20,
                },
            },
        },
    },
]


# ---------------------------------------------------------------------------
# Tool resolvers (synchronous, read-only)
# ---------------------------------------------------------------------------

def _resolve_tool(name: str, tool_input: dict) -> Any:
    if name == "list_sessions":
        limit = min(int(tool_input.get("limit", 10)), MAX_SESSIONS_PER_TOOL_CALL)
        status = tool_input.get("status") or None
        rows = db.get_sessions(limit=limit, status=status)
        for r in rows:
            ans = r.get("llm_answer") or ""
            if len(ans) > 300:
                r["llm_answer"] = ans[:300] + "…"
            r.pop("mad_output", None)
            r.pop("gateway_payload", None)
        return rows

    if name == "get_session":
        sid = (tool_input.get("session_id") or "").strip()
        if not sid:
            return {"error": "session_id is required"}
        row = db.get_session(sid)
        if not row:
            return {"error": f"Session '{sid}' not found"}
        ans = row.get("llm_answer") or ""
        if len(ans) > 500:
            row["llm_answer"] = ans[:500] + "…"
        mad = row.get("mad_output")
        if isinstance(mad, dict):
            mad.pop("debate_transcript", None)
        return row

    if name == "get_analytics":
        return db.get_analytics()

    if name == "get_system_health":
        try:
            db.ping()
            db_status = "healthy"
        except Exception as exc:
            db_status = f"unhealthy: {exc}"
        gateway_url = os.environ.get("GATEWAY_URL", "http://localhost:8080")
        return {
            "database": db_status,
            "gateway_url": gateway_url,
            "note": "For full per-component health with latency, see GET /api/system/health",
        }

    if name == "get_review_queue":
        rows = db.get_sessions(status="HUMAN_REVIEW_REQUIRED", limit=MAX_SESSIONS_PER_TOOL_CALL)
        for r in rows:
            r.pop("mad_output", None)
            r.pop("gateway_payload", None)
        return rows

    if name == "get_audit_logs":
        sid = tool_input.get("session_id") or None
        limit = min(int(tool_input.get("limit", 20)), MAX_AUDIT_LOGS_PER_TOOL_CALL)
        return db.get_audit_logs(session_id=sid, limit=limit)

    return {"error": f"Unknown tool: {name}"}


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_BASE_SYSTEM_PROMPT = """You are a co-pilot assistant embedded in the Guardrails AI Safety Platform — an enterprise control plane that monitors, audits, and governs AI sessions flowing through a multi-stage safety pipeline.

## Platform Overview

Each user query is processed through this pipeline:
1. **Gateway** — Runs three models: PII detection, jailbreak detection, and prompt injection detection. Issues a PASS / BLOCK / ESCALATE decision with a composite gateway_score (0–1).
2. **LLM** — If the gateway did not BLOCK, the query is answered by the configured LLM (e.g. Qwen2.5-7B via Ollama).
3. **MAD (Multi-Agent Debate)** — Two agents (A: drafts claims with evidence from a RAG store, B: challenges claims) run debate cycles. The judge produces a final routing decision.
4. **CSE (Confidence Score Engine)** — Calculates a final_cse_score and routing_decision based on claim confidence.

## Key Vocabulary

- **gateway_decision**: PASS (safe), BLOCK (threat confirmed), ESCALATE (uncertain, proceed with flag)
- **gateway_score**: 0 = no threat, 1 = maximum threat
- **mad_routing**: DELIVER (answer is safe to deliver), RETRY (re-run with more evidence), HUMAN_REVIEW (send to human analyst queue), HARD_BLOCK (do not deliver under any circumstances), BLOCK
- **mad_confidence**: 0–1 confidence in the routing decision
- **final_cse_score**: 0–1 — how well the LLM answer is supported by verified evidence
- **HUMAN_REVIEW_REQUIRED** status: session is sitting in the human review queue awaiting analyst action
- **HARD_BLOCKED**: session blocked by both gateway and MAD — never delivered
- **session_events**: append-only log of every status transition in a session's lifecycle

## Your Capabilities

You have 6 read-only tools to query live system data:
- list_sessions — list recent sessions with optional status filter
- get_session — get full details for one session
- get_analytics — aggregate stats (counts, latency, routing distribution)
- get_system_health — per-component health status
- get_review_queue — sessions awaiting human review
- get_audit_logs — audit trail of all actions in the system

## Behaviour Guidelines

- Be concise. Operators are busy — lead with the answer, then provide supporting detail.
- Always include session IDs when referencing specific sessions (short UUIDs are fine: first 8 chars).
- When data is empty (no sessions, empty queue), say so clearly rather than speculating.
- You cannot take write actions (approving reviews, modifying config). If asked, explain this and point to the relevant UI page.
- If you are uncertain about data, use a tool to check — do not guess."""


def _build_system_prompt(ctx: PageContext, user: UserContext) -> str:
    page_ctx = f"\n\n## Current Operator Context\n**Page:** {ctx.page}"
    if ctx.session_id:
        page_ctx += f"\n**Viewing session:** {ctx.session_id}"
    page_ctx += f"\n**User role:** {user.role}"
    return _BASE_SYSTEM_PROMPT + page_ctx


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_text(content_blocks: list) -> str:
    parts = []
    for block in content_blocks:
        if hasattr(block, "text") and block.text:
            parts.append(block.text)
    return " ".join(parts).strip() or "I processed your request."


# ---------------------------------------------------------------------------
# Route handler
# ---------------------------------------------------------------------------

@router.post("/chat", response_model=CopilotResponse)
async def copilot_chat(
    body: CopilotRequest,
    request: Request,
    user: UserContext = Depends(get_current_user),
):
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=503,
            detail="Co-pilot is not configured (ANTHROPIC_API_KEY is missing)",
        )

    client = anthropic.AsyncAnthropic(api_key=api_key)
    system_prompt = _build_system_prompt(body.context, user)

    # Cap history to last N messages
    history = body.history[-MAX_HISTORY_MESSAGES:]
    messages: list[dict] = [{"role": m.role, "content": m.content} for m in history]
    messages.append({"role": "user", "content": body.message})

    tool_call_log: list[ToolCallLog] = []
    reply = "I was unable to complete your request."
    iterations = 0

    while iterations < MAX_TOOL_ITERATIONS:
        iterations += 1
        response = await client.messages.create(
            model=COPILOT_MODEL,
            max_tokens=1024,
            system=system_prompt,
            tools=TOOLS,
            messages=messages,
        )

        if response.stop_reason == "end_turn":
            reply = _extract_text(response.content)
            break

        if response.stop_reason == "tool_use":
            messages.append({"role": "assistant", "content": response.content})
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    try:
                        output = _resolve_tool(block.name, block.input)
                    except Exception as exc:
                        logger.warning("tool %s failed: %s", block.name, exc)
                        output = {"error": str(exc)}
                    tool_call_log.append(
                        ToolCallLog(
                            tool_name=block.name,
                            input=block.input,
                            output=output,
                        )
                    )
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": json.dumps(output, default=str),
                        }
                    )
            messages.append({"role": "user", "content": tool_results})
        else:
            reply = _extract_text(response.content)
            break
    else:
        # Hit the iteration cap — extract whatever text we have
        if response and response.content:
            reply = _extract_text(response.content)
            if not reply:
                reply = "I reached the maximum number of tool calls while processing your request."

    # Write audit log
    try:
        db.insert_audit_log(
            action="copilot_query",
            actor_id=user.user_id,
            actor_role=user.role,
            actor_type="user",
            resource_type="copilot",
            request_id=getattr(request.state, "request_id", None),
            tenant_id=user.tenant_id,
            after={
                "message_truncated": body.message[:200],
                "tool_calls_count": len(tool_call_log),
                "page": body.context.page,
            },
            ip_address=request.client.host if request.client else None,
        )
    except Exception as exc:
        logger.warning("audit log for copilot_query failed: %s", exc)

    return CopilotResponse(reply=reply, tool_calls=tool_call_log, model=COPILOT_MODEL)
