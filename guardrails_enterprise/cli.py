"""
guardrails_enterprise/cli.py — Command-line interface.

Usage:
    guardrails-run "What does HIPAA require for PHI encryption?"
    guardrails-run "What does GDPR Article 17 require?" --skip-gateway
    guardrails-run "query" --skip-deepeval --model llama3.2:3b
    guardrails-run --health   # check all service health
"""
from __future__ import annotations

import argparse
import json
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="guardrails-run",
        description="Run a query through the Enterprise Guardrails pipeline.",
    )
    parser.add_argument("query", nargs="?", help="Query to process")
    parser.add_argument("--model",         default="qwen2.5:7b", help="Ollama model (default: qwen2.5:7b)")
    parser.add_argument("--ollama-url",    default="http://localhost:11434", help="Ollama base URL")
    parser.add_argument("--skip-gateway",  action="store_true", help="Skip PII/jailbreak validation")
    parser.add_argument("--skip-deepeval", action="store_true", help="Use judge-only CSE (no DeepEval)")
    parser.add_argument("--json",          action="store_true", dest="output_json", help="Output as JSON")
    parser.add_argument("--health",        action="store_true", help="Check service health and exit")
    args = parser.parse_args()

    # ── Health check mode ─────────────────────────────────────────────────────
    if args.health:
        from guardrails_enterprise.client import GuardrailsClient
        status = GuardrailsClient().health()
        for svc, s in status.items():
            icon = "✅" if s == "ok" else "❌"
            print(f"  {icon}  {svc:<10} {s}")
        ok = all(s == "ok" for s in status.values())
        sys.exit(0 if ok else 1)

    # ── Pipeline run mode ─────────────────────────────────────────────────────
    if not args.query:
        parser.print_help()
        sys.exit(1)

    from guardrails_enterprise.pipeline import run_pipeline

    try:
        result = run_pipeline(
            query         = args.query,
            ollama_url    = args.ollama_url,
            ollama_model  = args.model,
            skip_gateway  = args.skip_gateway,
            skip_deepeval = args.skip_deepeval,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    if args.output_json:
        out = {
            "query":      result.query,
            "routing":    result.routing,
            "blocked":    result.blocked,
            "confidence": result.confidence,
            "gateway": {
                "decision":      result.gateway.decision,
                "gateway_score": result.gateway.gateway_score,
                "pii_score":     result.gateway.pii_score,
                "jb_score":      result.gateway.jb_score,
                "pi_score":      result.gateway.pi_score,
            },
            "llm_answer": result.llm_answer,
            "mad": {
                "routing":    result.mad.routing_decision,
                "confidence": result.mad.aggregate_confidence,
                "claims":     result.mad.claim_count,
            } if result.mad else None,
            "cse": {
                "final_score": result.cse.final_score,
                "routing":     result.cse.routing_decision,
                "version":     result.cse.version,
                "components":  result.cse.components.__dict__,
            } if result.cse else None,
        }
        print(json.dumps(out, indent=2))
    else:
        _print_result(result)


def _print_result(result) -> None:
    SEP = "─" * 60
    print(f"\n{SEP}")
    print(f"  Query    : {result.query[:80]}")
    print(f"  Routing  : {result.routing}")
    print(f"  Blocked  : {result.blocked}")
    if result.confidence is not None:
        print(f"  Confidence: {result.confidence:.4f}")
    print(f"\n  Gateway  : {result.gateway.decision}  (score={result.gateway.gateway_score:.3f})")
    if result.gateway.threat_types:
        print(f"  Threats  : {', '.join(result.gateway.threat_types)}")

    if result.llm_answer:
        print(f"\n  LLM Answer:")
        for line in result.llm_answer.split("\n"):
            print(f"    {line}")

    if result.mad:
        m = result.mad
        print(f"\n  MAD      : {m.routing_decision}  (confidence={m.aggregate_confidence:.4f})")
        print(f"  Claims   : {m.claim_count} total ({m.material_claim_count} material)")
        if m.correction_signal:
            print(f"  Correction: {m.correction_signal[:80]}...")

    if result.cse:
        c = result.cse
        comp = c.components
        print(f"\n  CSE ({c.version}): {c.routing_decision}  final={c.final_score:.4f}")
        print(f"    F_llm={comp.f_llm:.3f}  H_llm={comp.h_llm:.3f}  "
              f"Relevancy={comp.relevancy:.3f}  Judge={comp.judge_eval:.3f}")
        if c.error:
            print(f"    ⚠ fallback: {c.error[:80]}")
    print(f"{SEP}\n")


if __name__ == "__main__":
    main()
