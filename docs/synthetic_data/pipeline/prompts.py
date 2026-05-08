METADATA_PROMPT = """
You are a regulatory compliance expert analyzing document chunks for an enterprise
AI guardrails system. Given the chunk below, return ONLY a JSON object with these fields:

{{
  "threat_categories": [],   // list from: unauthorized_data_access, pii_exfiltration,
                             // prompt_injection, policy_bypass, sensitive_data_exposure,
                             // cross_border_data_transfer, model_extraction,
                             // benign_allowed, ambiguous_escalate
  "cross_references": [],    // list of doc_ids this chunk relates to (e.g. "t1__gdpr_2016_679")
  "applicable_scenarios": [], // 2-4 enterprise scenarios where this chunk is relevant
  "severity": "",            // "high", "medium", or "low" for guardrails importance
  "summary": ""             // one sentence max summarizing the chunk's key rule
}}

CHUNK:
{chunk_text}

SOURCE: {doc_id} | TIER: {tier}

Return ONLY valid JSON. No explanation, no markdown, no preamble.
"""


TRIPLET_GENERATION_PROMPT = """
You are generating training data for an enterprise AI guardrails system.

POLICY CHUNK (this is the CORRECT answer for the query):
{positive_text}

SOURCE: {doc_id} | THREAT CATEGORIES: {threat_categories}

Generate {n_queries} realistic enterprise user queries that this policy chunk
would be the correct retrieval result for.

Rules:
- Queries must sound like real employee requests to an AI assistant
- NOT compliance questions ("what does X law say about Y")
- Cover mix of: benign_sensitive, clear_violation, adversarial, ambiguous
- Each query 1-2 sentences max
- Vary phrasing, do not repeat same structure

Return ONLY a JSON array of strings. No explanation.
["query 1", "query 2", ...]
"""


HARD_NEGATIVE_PROMPT = """
You are selecting training data for an embedding model.

QUERY: {query}
CORRECT CHUNK (positive): {positive_summary}

From the candidates below, select the ONE that is most confusingly similar
to the positive chunk BUT would NOT correctly answer the query.
It must share the same general topic but differ in jurisdiction, obligation type,
or specific rule.

CANDIDATES:
{candidates}

Return ONLY the chunk_id of the best hard negative. Nothing else.
"""


AGENT_GENERATION_PROMPT = """
You are generating training data for an enterprise AI safety agent.

Generate a realistic enterprise user query and the correct safety agent response.

RETRIEVED POLICY CHUNKS (these are what the RAG system found):
{context_chunks}

TARGET DECISION: {decision}
TARGET THREAT CATEGORY: {threat_category}

Generate:
1. A realistic user_query an enterprise employee would type to an AI assistant
   (NOT a compliance question - a real action request)
2. A complete safety agent response with this exact JSON structure:

{{
  "decision": "{decision}",
  "confidence": <float 0.0-1.0>,
  "threat_category": "<category>",
  "policy_violations": [
    {{"policy": "<source + section>", "violation": "<specific issue>"}}
  ],
  "recommended_action": "<what to do>",
  "pii_detected": <true/false>,
  "escalate_to_human": <true/false>
}}

Return ONLY a JSON object with keys "user_query" and "response".
No explanation. No markdown. Valid JSON only.
"""

