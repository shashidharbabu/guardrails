import os
import uuid
from typing import List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# --- GATEWAY / DATADOG IMPORTS ---
from gateway import telemetry as gw_telemetry
from gateway.gateway import GuardrailGateway

# --- MAD / LANGFUSE IMPORTS ---
from multi_agent_debate.multi_agent.claim_extractor import extract_claims
from multi_agent_debate.multi_agent.debate_engine import run_debate, build_transcript
from multi_agent_debate.multi_agent.judge import judge_claims
from multi_agent_debate.multi_agent.models import MADOutput, Claim, JudgeVerdict
from multi_agent_debate.multi_agent.config import CONFIDENCE_THRESHOLD_HIGH, CONFIDENCE_THRESHOLD_LOW, MAX_CYCLES

# Conditionally import Langfuse
try:
    from langfuse.decorators import observe, langfuse_context
    LANGFUSE_AVAILABLE = True
except ImportError:
    LANGFUSE_AVAILABLE = False
    # Mock observe decorator if Langfuse isn't installed
    def observe(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    class MockLangfuseContext:
        def update_current_trace(self, **kwargs): pass
        def update_current_observation(self, **kwargs): pass
    langfuse_context = MockLangfuseContext()

load_dotenv()

app = FastAPI(
    title="Unified Guardrails Gateway + MAD Pipeline",
    description="Combines Phase 1 (Input Gateway) and Phase 2 (Output MAD) with unified observability.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazy initialization for the Gateway
_gateway = None

def get_gateway():
    global _gateway
    if _gateway is None:
        _gateway = GuardrailGateway()
    return _gateway

# --- SCHEMAS ---

class UnifiedRequest(BaseModel):
    query: str = Field(..., description="The user's query")
    llm_answer: str = Field(..., description="The candidate LLM answer to evaluate")
    session_id: Optional[str] = None
    user_id: Optional[str] = None

class UnifiedResponse(BaseModel):
    # Phase 1
    gateway_decision: str
    gateway_blocked_reason: Optional[str]
    # Phase 2
    mad_routing_decision: Optional[str]
    final_confidence_score: Optional[float]
    correction_signal: Optional[str]
    # Detailed data
    debate_transcript: Optional[str]
    # Observability Links
    datadog_trace_url: Optional[str] = None
    langfuse_trace_url: Optional[str] = None

# --- CSE / ABLATION FORMULA ---

def get_deepeval_metrics(query: str, answer: str) -> tuple[float, float, float]:
    """
    Mock DeepEval Phase 2 Layer 1 metrics.
    In a real implementation, this would call DeepEval.
    Returns: (F_llm, H_llm, relevancy)
    """
    return 0.85, 0.1, 0.90 

def compute_full_confidence(judge_eval_score: float, query: str, answer: str) -> float:
    """
    Phase 4 CSE full formula:
    final_score = 0.30 × F_llm + 0.25 × (1 − H_llm) + 0.10 × relevancy + 0.35 × judge_eval_score
    """
    F_llm, H_llm, relevancy = get_deepeval_metrics(query, answer)
    score = (0.30 * F_llm) + (0.25 * (1 - H_llm)) + (0.10 * relevancy) + (0.35 * judge_eval_score)
    return round(score, 4)

# --- OBSERVABILITY WRAPPED MAD PIPELINE ---

@observe(name="mad_extraction")
def do_extract(query: str, llm_answer: str):
    return extract_claims(query, llm_answer)

@observe(name="mad_debate")
def do_debate(query: str, claims: list):
    return run_debate(query, claims, MAX_CYCLES)

@observe(name="mad_judge")
def do_judge(query: str, final_claims: list, evidence_pool: list):
    return judge_claims(query, final_claims, evidence_pool)

@observe(as_type="generation", name="unified_mad_pipeline")
def run_unified_mad(query: str, llm_answer: str, session_id: str = None, user_id: str = None) -> MADOutput:
    langfuse_context.update_current_trace(
        session_id=session_id,
        user_id=user_id,
        tags=["unified_pipeline"],
    )
    
    claims = do_extract(query, llm_answer)
    cycles, final_claims, evidence_pool = do_debate(query, claims)
    judge_verdicts, correction_signal = do_judge(query, final_claims, evidence_pool)
    
    # Calculate Judge score (v0.1 logic)
    material_jvs = [jv for jv in judge_verdicts if jv.is_material]
    if material_jvs:
        judge_eval_score = min(jv.score for jv in material_jvs)
    else:
        judge_eval_score = sum(jv.score for jv in judge_verdicts) / len(judge_verdicts) if judge_verdicts else 0.5
        
    # Apply Final Full Confidence Formula
    final_confidence = compute_full_confidence(judge_eval_score, query, llm_answer)
    langfuse_context.update_current_observation(
        metadata={"judge_eval_score": judge_eval_score, "final_confidence": final_confidence}
    )
    
    # Routing (updated to use final_confidence)
    routing = "HUMAN_REVIEW"
    for jv in judge_verdicts:
        claim = next((c for c in final_claims if c.claim_id == jv.claim_id), None)
        if claim and claim.is_material and jv.score == 0.0:
            routing = "HARD_BLOCK"
            break
    
    if routing != "HARD_BLOCK":
        if final_confidence >= CONFIDENCE_THRESHOLD_HIGH:
            routing = "DELIVER"
        elif final_confidence >= CONFIDENCE_THRESHOLD_LOW:
            routing = "RETRY"
        else:
            routing = "HUMAN_REVIEW"
            
    transcript = build_transcript(query, llm_answer, cycles, judge_verdicts, correction_signal or "")
    
    return MADOutput(
        query=query,
        llm_answer=llm_answer,
        claims=final_claims,
        debate_cycles=cycles,
        evidence_pool=evidence_pool,
        judge_verdicts=judge_verdicts,
        correction_signal=correction_signal,
        routing_decision=routing,
        aggregate_confidence=final_confidence,
        debate_transcript=transcript,
    )

# --- UNIFIED ENDPOINT ---

@app.post("/verify_unified", response_model=UnifiedResponse)
def verify_unified(request: UnifiedRequest):
    """
    Run the unified pipeline: Gateway (Datadog) -> MAD (Langfuse)
    """
    trace_hint = request.session_id or request.user_id or str(uuid.uuid4())
    
    datadog_trace_url = None
    langfuse_trace_url = None

    # 1. GATEWAY PHASE (Monitored by Datadog)
    gateway = get_gateway()
    with gw_telemetry.span(
        "gateway.unified.process", 
        trace_id=trace_hint, 
        session_id=request.session_id or "",
        user_id=request.user_id or "",
    ):
        gw_result = gateway.process(request.query, trace_id=trace_hint)
        
        # Datadog trace URL logic (assuming US5 site or passing standard DD format)
        if gw_telemetry.ddtrace_active():
            import ddtrace
            current_span = ddtrace.tracer.current_span()
            if current_span:
                trace_id = current_span.trace_id
                dd_site = os.getenv("DD_SITE", "us5.datadoghq.com")
                dd_env = os.getenv("DD_ENV", "local")
                datadog_trace_url = f"https://{dd_site}/apm/trace/{trace_id}?env={dd_env}"
    
    if gw_result.decision.value == "BLOCK":
        return UnifiedResponse(
            gateway_decision=gw_result.decision.value,
            gateway_blocked_reason=gw_result.blocked_reason,
            mad_routing_decision=None,
            final_confidence_score=None,
            correction_signal=None,
            debate_transcript=None,
            datadog_trace_url=datadog_trace_url,
            langfuse_trace_url=None
        )
        
    # 2. MAD PHASE (Monitored by Langfuse)
    # MAD is only executed if the input passes the Gateway
    mad_result = run_unified_mad(
        query=request.query, 
        llm_answer=request.llm_answer,
        session_id=request.session_id,
        user_id=request.user_id
    )
    
    if LANGFUSE_AVAILABLE and hasattr(langfuse_context, "get_current_trace_url"):
        langfuse_trace_url = langfuse_context.get_current_trace_url()
    
    return UnifiedResponse(
        gateway_decision=gw_result.decision.value,
        gateway_blocked_reason=gw_result.blocked_reason,
        mad_routing_decision=mad_result.routing_decision,
        final_confidence_score=mad_result.aggregate_confidence,
        correction_signal=mad_result.correction_signal,
        debate_transcript=mad_result.debate_transcript,
        datadog_trace_url=datadog_trace_url,
        langfuse_trace_url=langfuse_trace_url
    )

@app.get("/health")
def health():
    return {
        "status": "ok", 
        "langfuse_available": LANGFUSE_AVAILABLE,
        "datadog_available": gw_telemetry.ddtrace_active()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
