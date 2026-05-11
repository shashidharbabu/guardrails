"""
pipeline.py — Full end-to-end pipeline orchestrator.

Flow: Gateway → LLM → MAD (background asyncio task) → DB
All service URLs come from the settings module — no hardcoded localhost.
"""

import asyncio
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Optional

import httpx

from app.backend import db
from app.backend.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


def _sanitise_json_str(s: str) -> str:
    """Strip control characters that break JSON serialisation (e.g. raw tabs in LLM output)."""
    import re
    return re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', ' ', s)

SYSTEM_PROMPT = (
    "You are an enterprise compliance assistant. Answer questions about regulatory "
    "requirements accurately and concisely based on your knowledge of healthcare, "
    "banking, and legal regulations. Be concise — 2-4 sentences maximum."
)


async def run_pipeline(
    query: str,
    llm_model: Optional[str] = None,
    request_id: Optional[str] = None,
) -> dict:
    """
    Run the full pipeline for a query.
    Gateway + LLM run synchronously and are returned immediately.
    MAD runs as a background asyncio task and updates the session row when done.
    """
    session_id = str(uuid.uuid4())
    model = llm_model or settings.DEFAULT_LLM_MODEL
    t0 = time.time()

    # ── 1. GATEWAY ────────────────────────────────────────────────────────────
    db.insert_session_event(
        session_id=session_id,
        stage="gateway",
        to_status="GATEWAY_RUNNING",
        request_id=request_id,
    )
    gw_payload = await _call_gateway(query)
    gateway_decision = gw_payload["decision"]
    gateway_score = gw_payload.get("gateway_score", gw_payload.get("threat_score", 0.0))

    gateway_status = {
        "PASS": "GATEWAY_PASSED",
        "BLOCK": "GATEWAY_BLOCKED",
        "ESCALATE": "GATEWAY_ESCALATED",
    }.get(gateway_decision, "GATEWAY_PASSED")

    llm_answer = None

    # ── 2. LLM (skip if gateway BLOCK) ────────────────────────────────────────
    if gateway_decision != "BLOCK":
        db.insert_session_event(
            session_id=session_id,
            stage="llm",
            to_status="LLM_RUNNING",
            from_status=gateway_status,
            request_id=request_id,
        )
        try:
            llm_answer = await _call_llm(query, model)
            db.insert_session_event(
                session_id=session_id,
                stage="llm",
                to_status="LLM_COMPLETED",
                from_status="LLM_RUNNING",
                request_id=request_id,
            )
        except Exception as e:
            llm_answer = f"[LLM error: {e}]"
            db.insert_session_event(
                session_id=session_id,
                stage="llm",
                to_status="FAILED",
                from_status="LLM_RUNNING",
                error_message=str(e),
                request_id=request_id,
            )

    duration_ms = int((time.time() - t0) * 1000)
    initial_status = gateway_status if gateway_decision == "BLOCK" else "LLM_COMPLETED"

    # ── 3. PERSIST initial session ────────────────────────────────────────────
    session = {
        "id": session_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "query": query,
        "status": initial_status,
        "gateway_decision": gateway_decision,
        "gateway_score": gateway_score,
        "gateway_payload": json.dumps(gw_payload),
        "llm_answer": llm_answer,
        "llm_model": model,
        "mad_routing": None,
        "mad_confidence": None,
        "mad_output_json": None,
        "mad_query_id": None,
        "mad_rollout_id": None,
        "langfuse_trace_id": None,
        "pipeline_duration_ms": duration_ms,
        "cse_result_json": None,
        "final_route": gateway_status if gateway_decision == "BLOCK" else None,
    }
    db.insert_session(session)

    # ── 4. KICK OFF MAD in background ─────────────────────────────────────────
    if llm_answer and not llm_answer.startswith("[LLM error"):
        db.insert_session_event(
            session_id=session_id,
            stage="mad",
            to_status="MAD_QUEUED",
            from_status="LLM_COMPLETED",
            request_id=request_id,
        )
        asyncio.create_task(
            _run_mad_background(session_id, query, llm_answer, t0, request_id)
        )

    return db.get_session(session_id)


async def _run_mad_background(
    session_id: str,
    query: str,
    llm_answer: str,
    t0: float,
    request_id: Optional[str] = None,
):
    db.insert_session_event(
        session_id=session_id,
        stage="mad",
        to_status="MAD_RUNNING",
        from_status="MAD_QUEUED",
        request_id=request_id,
    )
    try:
        mad_out = await _run_mad_async(query, llm_answer)
        if mad_out:
            mad_routing = mad_out.routing_decision
            mad_confidence = mad_out.aggregate_confidence
            mad_output = mad_out.model_dump(mode="json")
            cse_result = getattr(mad_out, "cse_result", None)
            duration_ms = int((time.time() - t0) * 1000)
            db.update_session_mad(
                session_id=session_id,
                mad_routing=mad_routing,
                mad_confidence=mad_confidence,
                mad_output_json=_sanitise_json_str(json.dumps(mad_output)),
                mad_query_id=getattr(mad_out, "query_id", "") or "",
                mad_rollout_id=getattr(mad_out, "rollout_id", "") or "",
                pipeline_duration_ms=duration_ms,
                cse_result_json=json.dumps(cse_result) if isinstance(cse_result, dict) else None,
                langfuse_trace_id=getattr(mad_out, "langfuse_trace_id", None),
            )
            db.insert_session_event(
                session_id=session_id,
                stage="mad",
                to_status="MAD_COMPLETED",
                from_status="MAD_RUNNING",
                message=f"MAD routing: {mad_routing}",
                request_id=request_id,
            )
            logger.info(
                "mad_completed",
                extra={"session_id": session_id, "mad_routing": mad_routing},
            )
        else:
            db.insert_session_event(
                session_id=session_id,
                stage="mad",
                to_status="MAD_UNAVAILABLE",
                from_status="MAD_RUNNING",
                error_code="MAD_UNAVAILABLE",
                error_message="MAD service returned no result",
                request_id=request_id,
            )
            db.update_session_status(
                session_id,
                "MAD_UNAVAILABLE",
                error_message="MAD service returned no result",
            )
            logger.warning("mad_unavailable", extra={"session_id": session_id})
    except Exception as exc:
        logger.error("mad_background_failed", extra={"session_id": session_id, "error": str(exc)}, exc_info=True)
        db.insert_session_event(
            session_id=session_id,
            stage="mad",
            to_status="FAILED",
            from_status="MAD_RUNNING",
            error_message=str(exc),
            request_id=request_id,
        )
        db.update_session_status(session_id, "FAILED", error_message=str(exc))


async def _call_gateway(query: str) -> dict:
    url = f"{settings.GATEWAY_URL}/validate"
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            r = await client.post(url, json={"text": query})
            r.raise_for_status()
            return r.json()
    except httpx.ConnectError:
        logger.error("gateway_unreachable", extra={"url": url})
        # Fail open with ESCALATE so LLM still runs but is flagged
        return {
            "decision": "ESCALATE",
            "gateway_score": 0.5,
            "error": "gateway_unreachable",
        }


async def _call_llm(query: str, model: str) -> str:
    provider = getattr(settings, "LLM_PROVIDER_TYPE", "openai").lower()
    if provider == "sagemaker":
        return await _call_llm_sagemaker(query, model)
    if provider == "claude":
        return await _call_llm_claude(query, model)
    return await _call_llm_openai_compat(query, model)


async def _call_llm_sagemaker(query: str, model: str) -> str:
    """
    Call a DJL LMI endpoint on SageMaker (Qwen2.5-14B-Instruct).
    Uses Qwen2.5 chat template. model param = SageMaker endpoint name.
    """
    import json as _json
    import re
    import asyncio

    endpoint_name = model or getattr(settings, "LLM_SAGEMAKER_ENDPOINT", "spartanguard-agents")
    aws_region = getattr(settings, "AWS_REGION", "us-west-2")

    # Qwen2.5 chat template
    prompt = (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{query}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 512,
            "do_sample": True,
            "temperature": 0.7,
            "top_p": 0.9,
        },
    }

    _stop = re.compile(r"(<\|im_end\|>|<\|endoftext\|>|<\|eot_id\|>).*", re.DOTALL)

    def _invoke():
        import boto3
        client = boto3.client("sagemaker-runtime", region_name=aws_region)
        response = client.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType="application/json",
            Body=_json.dumps(payload),
        )
        result = _json.loads(response["Body"].read())
        if "generated_text" in result:
            raw = result["generated_text"]
        elif isinstance(result, list) and result:
            raw = result[0].get("generated_text", str(result[0]))
        else:
            raw = str(result)
        return _stop.sub("", raw).strip()

    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _invoke)


async def _call_llm_claude(query: str, model: str) -> str:
    import anthropic
    api_key = settings.ANTHROPIC_API_KEY
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set")
    client = anthropic.AsyncAnthropic(api_key=api_key)
    message = await client.messages.create(
        model=model,
        max_tokens=512,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": query}],
    )
    return message.content[0].text


async def _call_llm_openai_compat(query: str, model: str) -> str:
    base = settings.LLM_PROVIDER_URL.rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3]
    llm_url = f"{base}/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ],
        "stream": False,
        "max_tokens": 512,
    }
    async with httpx.AsyncClient(timeout=300.0) as client:
        r = await client.post(llm_url, json=payload)
        r.raise_for_status()
        data = r.json()
        return data["choices"][0]["message"]["content"]


async def _run_mad_async(query: str, llm_answer: str):
    """
    Call the mad-api HTTP service at MAD_API_URL (http://mad-api:8001/mad/verify).
    MAD_MODE=api   → HTTP POST to mad-api (production, always used if set)
    MAD_MODE=disabled → skip MAD entirely
    """
    if settings.MAD_MODE == "disabled":
        return None

    mad_url = getattr(settings, "MAD_API_URL", "http://mad-api:8001/mad/verify")
    timeout = float(getattr(settings, "MAD_TIMEOUT_SECONDS", "600"))

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.post(
                mad_url,
                json={"query": query, "llm_answer": llm_answer},
            )
            r.raise_for_status()
            data = r.json()

        class _MADResult:
            """Lightweight wrapper so the rest of pipeline.py works unchanged."""
            def __init__(self, d: dict):
                self.routing_decision = d.get("routing_decision", "HUMAN_REVIEW")
                self.aggregate_confidence = d.get("aggregate_confidence", 0.0)
                self.query_id = d.get("query_id", "")
                self.rollout_id = d.get("rollout_id", "")
                self.langfuse_trace_id = d.get("langfuse_trace_id")
                self.cse_result = None
                self._raw = d
            def model_dump(self, **kwargs):
                return self._raw

        return _MADResult(data)

    except Exception as exc:
        logger.warning("mad_http_call_failed", extra={"url": mad_url, "error": str(exc)})
        return None
