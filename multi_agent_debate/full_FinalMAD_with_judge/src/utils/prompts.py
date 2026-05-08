from src.schemas.schemas import AgentOutputStripped, Claim


DECOMPOSER_SYSTEM = """You are an atomic claim extractor for regulatory text verification.

Your job: take a regulatory answer and break it into a list of atomic claims.

RULES:
1. Each claim is ONE verifiable fact. Never merge two facts.
2. Tag each claim:
   - is_material: True if claim affects the correctness of the answer
   - is_critical: True if claim being wrong would cause real-world harm (legal liability, patient safety, financial loss)
3. Assign confidence_prior between 0.50 and 0.90:
   - 0.50: very uncertain, claim is hedged or vague in original
   - 0.90: very specific, falsifiable, central to the answer
   - Never 0.00 or 1.00 — leave room for agent updates

OUTPUT FORMAT: Valid JSON only, no markdown:
{
    "claims": [
        {
            "claim_text": "...",
            "claim_index": 0,
            "is_material": true,
            "is_critical": false,
            "confidence_prior": 0.75
        }
    ]
}"""


def decomposer_user(query: str, baseline_answer: str) -> str:
    return f"""ORIGINAL QUERY: {query}

BASELINE LLM ANSWER:
{baseline_answer}

Extract atomic claims from this answer."""


def format_chunks(chunks: list) -> str:
    formatted = []
    for chunk in chunks:
        chunk_id = chunk.get("chunk_id", "unknown")
        text = chunk.get("text", "")
        formatted.append(f"[{chunk_id}]\n{text}")
    return "\n\n".join(formatted)


AGENT_A_SYSTEM = """You are a strict regulatory compliance auditor in a debate.

YOUR ROLE: Verify whether the LLM's claim is supported by the retrieved evidence.
Be precise. Find errors, hallucinations, unsupported claims, or misleading statements.

RULES:
1. Read the claim carefully.
2. Analyze the retrieved evidence.
3. Determine if the claim is SUPPORTED, PARTIAL, NOT_SUPPORTED, or IDK.
4. Cite at most 2 specific chunk_ids.
5. If the other debater makes a valid point in Round 1, acknowledge it — your goal is TRUTH, not winning.
6. Confidence: assign your true confidence 0.0–1.0. This is a self-assessment.
7. Keep reasoning to at most 3 short sentences.
8. Each relevant_quote must be at most 240 characters.
9. Return valid JSON only. Do not use markdown fences, bullets, or text before/after JSON.

OUTPUT FORMAT: Valid JSON only, no markdown:
{
    "verdict": "SUPPORTED | PARTIAL | NOT_SUPPORTED | IDK",
    "reasoning": "At most 3 short sentences.",
    "evidence_cited": [{"chunk_id": "...", "relevant_quote": "..."}],
    "confidence_internal": 0.0-1.0
}"""


AGENT_B_SYSTEM = """You are an adversarial auditor in a debate.

YOUR ROLE: Challenge the LLM's claim. Look for what's WRONG, MISSING, or UNSUPPORTED.
You are NOT a blind attacker — you are a careful skeptic. Acknowledge when a claim is well-supported.

RULES:
1. Read the claim carefully.
2. Analyze the retrieved evidence.
3. Look for: scope errors, exceptions, outdated guidance, fabricated references, missing context.
4. Cite at most 2 specific chunk_ids.
5. If the claim is genuinely supported, say so — credibility depends on accuracy.
6. Confidence: assign your true confidence 0.0–1.0 in your verdict.
7. Keep reasoning to at most 3 short sentences.
8. Each relevant_quote must be at most 240 characters.
9. Return valid JSON only. Do not use markdown fences, bullets, or text before/after JSON.

OUTPUT FORMAT: Valid JSON only, no markdown:
{
    "verdict": "SUPPORTED | PARTIAL | NOT_SUPPORTED | IDK",
    "reasoning": "At most 3 short sentences.",
    "evidence_cited": [{"chunk_id": "...", "relevant_quote": "..."}],
    "confidence_internal": 0.0-1.0
}"""


def round0_user(claim: Claim, chunks: list, query: str) -> str:
    return f"""USER QUERY: {query}

CLAIM TO VERIFY:
{claim.claim_text}

CLAIM METADATA:
- is_material: {claim.is_material}
- is_critical: {claim.is_critical}

RETRIEVED EVIDENCE:
{format_chunks(chunks)}

Provide your independent verdict on this claim. Return compact valid JSON only."""


def round1_user(claim: Claim, chunks: list, query: str, peer: AgentOutputStripped) -> str:
    return f"""USER QUERY: {query}

CLAIM TO VERIFY:
{claim.claim_text}

RETRIEVED EVIDENCE:
{format_chunks(chunks)}

THE OTHER DEBATER'S POSITION ({peer.debater_label}):
Verdict: {peer.verdict}
Reasoning: {peer.reasoning}
Evidence cited: {peer.evidence_cited}

ROUND 1 TASK:
1. Review the other debater's Round 0 verdict, reasoning, and cited evidence.
2. Re-check the claim against the same retrieved evidence chunks above.
3. Identify whether the other debater cited valid evidence or made unsupported assumptions.
4. Change your verdict if the other debater is correct.
5. Keep your verdict if the other debater is wrong, weak, or unsupported.
6. Do not agree just to be cooperative.
7. Do not use the other debater's confidence; it is intentionally not shown.
8. Return compact valid JSON only."""


JUDGE_SYSTEM = """You are the final arbiter in a debate about an LLM claim's factual accuracy.

You witnessed a 2-round debate between two debaters about whether a claim is supported by evidence.

YOUR TASK:
1. Review the original claim.
2. Review both debaters' Round 0 and Round 1 positions (verdicts, reasoning, evidence).
3. Review the evidence pool.
4. Render a verdict:
   - 1.0: Fully supported by evidence
   - 0.5: Partially supported, ambiguous, or evidence is mixed
   - 0.0: Not supported, contradicted, or hallucinated

You are BLIND to:
- Which debater was Agent A vs Agent B
- Either debater's confidence scores

Judge based on the strength of evidence and reasoning alone.

OUTPUT FORMAT: Valid JSON only:
{
    "v_label": 1.0 | 0.5 | 0.0,
    "judge_confidence": 0.0-1.0,
    "judge_reasoning": "Your analysis of why this verdict, with citations...",
    "evidence_chunk_ids": ["chunk_id_1", "chunk_id_2"]
}"""


def judge_user(
    claim: Claim,
    chunks: list,
    query: str,
    d1_r0: AgentOutputStripped,
    d1_r1: AgentOutputStripped,
    d2_r0: AgentOutputStripped,
    d2_r1: AgentOutputStripped,
) -> str:
    return f"""USER QUERY: {query}

CLAIM:
{claim.claim_text}

EVIDENCE POOL:
{format_chunks(chunks)}

Debater 1, ROUND 0:
Verdict: {d1_r0.verdict}
Reasoning: {d1_r0.reasoning}
Evidence: {d1_r0.evidence_cited}

Debater 1, ROUND 1:
Verdict: {d1_r1.verdict}
Reasoning: {d1_r1.reasoning}
Evidence: {d1_r1.evidence_cited}

Debater 2, ROUND 0:
Verdict: {d2_r0.verdict}
Reasoning: {d2_r0.reasoning}
Evidence: {d2_r0.evidence_cited}

Debater 2, ROUND 1:
Verdict: {d2_r1.verdict}
Reasoning: {d2_r1.reasoning}
Evidence: {d2_r1.evidence_cited}

Render your final verdict."""
