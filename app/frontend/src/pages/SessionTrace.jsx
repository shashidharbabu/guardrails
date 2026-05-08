import { useEffect, useState } from 'react'
import { useParams, Link, useNavigate } from 'react-router-dom'
import { motion, AnimatePresence, useReducedMotion } from 'framer-motion'
import { getSession, submitFeedback, getSessionCSE } from '../api/client'
import StatusBadge, { decisionCls, madCls, claimCls } from '../components/StatusBadge'
import ScoreBar from '../components/ScoreBar'
import LoadingSkeleton from '../components/LoadingSkeleton'

/* ─── local helpers ─────────────────────────────────────────── */
function verdictClass(d) {
  if (d === 'BLOCK' || d === 'HARD_BLOCK') return 'vd-block'
  if (d === 'ESCALATE' || d === 'RETRY') return 'vd-esc'
  return 'vd-pass'
}

/* ─── Stage accordion ────────────────────────────────────────── */
function Stage({ icon, iconClass, title, timeLabel, badge, badgeClass, children, defaultOpen = true, skipped = false }) {
  const [open, setOpen] = useState(defaultOpen)
  const shouldReduceMotion = useReducedMotion()

  return (
    <div className="stage">
      <button
        type="button"
        className={`stage-hdr${open ? ' open' : ''}${skipped ? ' skipped' : ''}`}
        onClick={() => setOpen(o => !o)}
        aria-expanded={open}
      >
        <div className={`si ${iconClass}`}>{icon}</div>
        <div className="sn">{title}</div>
        <div className="st-time">{timeLabel}</div>
        {badge && (
          <span className={`badge ${badgeClass || 'b-gray'}`} style={{ fontSize: '10px' }}>
            {badge}
          </span>
        )}
        <svg
          className={`stg${open ? ' open' : ''}`}
          viewBox="0 0 12 12"
          fill="none"
          aria-hidden="true"
        >
          <path d="M2 4l4 4 4-4" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round"/>
        </svg>
      </button>

      <AnimatePresence initial={false}>
        {open && (
          <motion.div
            className="stage-body"
            initial={shouldReduceMotion ? false : { height: 0, opacity: 0 }}
            animate={shouldReduceMotion ? {} : { height: 'auto', opacity: 1 }}
            exit={shouldReduceMotion ? {} : { height: 0, opacity: 0 }}
            transition={{ duration: 0.2, ease: [0.16, 1, 0.3, 1] }}
            style={{ overflow: 'hidden' }}
          >
            {children}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}

/* ─── Feedback panel ─────────────────────────────────────────── */
function FeedbackPanel({ sessionId }) {
  const [rating, setRating] = useState(0)
  const [label, setLabel] = useState('correct')
  const [comment, setComment] = useState('')
  const [submitted, setSubmitted] = useState(false)
  const [loading, setLoading] = useState(false)

  async function handleSubmit() {
    if (!rating) return
    setLoading(true)
    try {
      await submitFeedback(sessionId, { rating, label, comment })
      setSubmitted(true)
    } finally {
      setLoading(false)
    }
  }

  if (submitted) {
    return (
      <div className="ts-sec">
        <div className="ts-title">Analyst Feedback</div>
        <div className="inline-ok">Feedback submitted. Thank you.</div>
      </div>
    )
  }

  return (
    <div className="ts-sec">
      <div className="ts-title">Analyst Feedback</div>

      {/* Star rating */}
      <div style={{ display: 'flex', gap: '2px', marginBottom: '10px' }}>
        {[1,2,3,4,5].map(n => (
          <button
            key={n}
            type="button"
            onClick={() => setRating(n)}
            aria-label={`Rate ${n}`}
            style={{
              background: 'none',
              border: 'none',
              cursor: 'pointer',
              fontSize: '18px',
              color: n <= rating ? 'var(--amber)' : 'var(--border-hi)',
              padding: '0 2px',
              transition: 'color 0.1s',
            }}
          >
            ★
          </button>
        ))}
      </div>

      {/* Label chips */}
      <div style={{ display: 'flex', gap: '4px', marginBottom: '10px' }}>
        {['correct', 'partial', 'incorrect'].map(l => (
          <button
            key={l}
            type="button"
            onClick={() => setLabel(l)}
            className={`chip${label === l ? ' active' : ''}`}
            style={{ padding: '3px 9px', fontSize: '10px' }}
          >
            {l}
          </button>
        ))}
      </div>

      <textarea
        value={comment}
        onChange={e => setComment(e.target.value)}
        placeholder="Optional note…"
        style={{
          width: '100%',
          padding: '6px 8px',
          fontFamily: 'var(--font-mono)',
          fontSize: '11px',
          background: 'var(--bg-surface)',
          border: '1px solid var(--border-md)',
          borderRadius: 'var(--r-sm)',
          color: 'var(--text-primary)',
          resize: 'none',
          outline: 'none',
          marginBottom: '8px',
          lineHeight: 1.5,
        }}
        rows={2}
      />

      <button
        type="button"
        className="act-btn lf-btn"
        disabled={!rating || loading}
        onClick={handleSubmit}
        style={{ marginBottom: 0 }}
      >
        {loading ? 'Submitting…' : 'Submit Feedback'}
      </button>
    </div>
  )
}

/* ─── CSE panel ──────────────────────────────────────────────── */
function CSEPanel({ cseData }) {
  if (!cseData) return null
  const { cse } = cseData
  if (!cse) return null

  const score = cse.final_cse_score
  const route = cse.routing_decision
  const routeColor =
    route === 'DELIVER' ? 'var(--teal)' :
    route === 'HARD_BLOCK' ? 'var(--red)' :
    route === 'HUMAN_REVIEW' ? 'var(--violet)' : 'var(--amber)'

  return (
    <div className="ts-sec">
      <div className="ts-title">Confidence Engine (CSE)</div>
      <div style={{ marginBottom: 8 }}>
        <div style={{ fontSize: 10, color: 'var(--text-muted)', marginBottom: 2 }}>FINAL ROUTE</div>
        <div style={{ fontFamily: 'var(--font-ui)', fontWeight: 700, fontSize: 13, color: routeColor }}>
          {route || '—'}
        </div>
      </div>
      {score != null && (
        <div style={{ marginBottom: 8 }}>
          <div style={{ fontSize: 10, color: 'var(--text-muted)', marginBottom: 2 }}>SCORE</div>
          <div style={{ fontFamily: 'var(--font-mono)', fontSize: 12, color: 'var(--text-primary)' }}>
            {(score * 100).toFixed(1)}%
          </div>
        </div>
      )}
      {cse.scoring_mode && (
        <div style={{ marginBottom: 8 }}>
          <span style={{ fontSize: 9, padding: '1px 6px', borderRadius: 100, background: 'var(--bg-base)', border: '1px solid var(--border-md)', color: 'var(--text-muted)', fontFamily: 'var(--font-ui)' }}>
            {cse.scoring_mode} mode · v{cse.cse_version}
          </span>
        </div>
      )}
      {cse.explanation && (
        <div style={{ fontSize: 11, color: 'var(--text-sec)', lineHeight: 1.5, marginBottom: 8 }}>
          {cse.explanation}
        </div>
      )}
      {cse.triggered_flags?.length > 0 && (
        <div style={{ marginBottom: 8 }}>
          <div style={{ fontSize: 10, color: 'var(--text-muted)', marginBottom: 4 }}>FLAGS</div>
          <div style={{ display: 'flex', gap: 4, flexWrap: 'wrap' }}>
            {cse.triggered_flags.map((f, i) => (
              <span key={i} style={{ fontSize: 9, padding: '1px 6px', borderRadius: 100, background: 'rgba(239,68,68,0.1)', color: 'var(--red)', border: '1px solid rgba(239,68,68,0.2)', fontFamily: 'var(--font-ui)' }}>
                {f}
              </span>
            ))}
          </div>
        </div>
      )}
      {cse.top_failed_claims?.length > 0 && (
        <div>
          <div style={{ fontSize: 10, color: 'var(--text-muted)', marginBottom: 4 }}>FAILED CLAIMS</div>
          {cse.top_failed_claims.map((c, i) => (
            <div key={i} style={{ fontSize: 10, color: 'var(--red)', fontFamily: 'var(--font-mono)', marginBottom: 2 }}>
              · {typeof c === 'string' ? c : c.claim_text || JSON.stringify(c)}
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

/* ─── Main page ──────────────────────────────────────────────── */
export default function SessionTrace() {
  const { id } = useParams()
  const navigate = useNavigate()
  const [session, setSession] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [transcriptOpen, setTranscriptOpen] = useState(false)
  const [cseData, setCseData] = useState(null)

  useEffect(() => {
    getSession(id)
      .then(s => {
        setSession(s)
        // Try to load CSE if MAD is done
        if (s.mad_routing) {
          getSessionCSE(id).then(setCseData).catch(() => {})
        }
      })
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [id])

  useEffect(() => {
    if (!session) return
    if (session.mad_routing || !session.llm_answer || session.gateway_decision === 'BLOCK') return
    const interval = setInterval(() => {
      getSession(id)
        .then(s => {
          setSession(s)
          if (s.mad_routing) clearInterval(interval)
        })
        .catch(() => {})
    }, 10000)
    return () => clearInterval(interval)
  }, [session?.id, session?.mad_routing])

  if (loading) {
    return (
      <div>
        <div style={{ marginBottom: 12 }}>
          <div className="skeleton" style={{ width: 140, height: 12 }} />
        </div>
        <LoadingSkeleton type="cards" count={3} />
      </div>
    )
  }

  if (error || !session) {
    return (
      <div>
        <div className="inline-err" style={{ marginBottom: 12 }}>
          {error || 'Session not found.'}
        </div>
        <Link to="/" className="back-btn">← Back to conversations</Link>
      </div>
    )
  }

  const { gateway_payload: gw, mad_output: mad } = session
  const totalMs = session.pipeline_duration_ms
  const gatewayMs = gw?.duration_ms || Math.round((totalMs || 0) * 0.04)
  const piiScore = gw?.scores?.pii ?? gw?.pii_score ?? 0
  const jbScore  = gw?.scores?.jailbreak ?? gw?.jailbreak_score ?? 0
  const piScore  = gw?.scores?.prompt_injection ?? gw?.injection_score ?? 0
  const piiEntities = gw?.pii_entities || gw?.entities || []
  const llmMs = Math.round((totalMs || 0) * 0.12)
  const madMs = Math.round((totalMs || 0) * 0.7)
  const judgeMs = totalMs ? totalMs - gatewayMs - llmMs - madMs : 0
  const confidence = mad?.aggregate_confidence ?? mad?.confidence_score
  const confClass = confidence == null ? '' : confidence >= 0.75 ? 'conf-high' : confidence >= 0.45 ? 'conf-mid' : 'conf-low'

  return (
    <div>
      <Link to="/" className="back-btn">
        <svg width="12" height="12" viewBox="0 0 12 12" fill="none">
          <path d="M8 2L4 6l4 4" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round"/>
        </svg>
        Back to conversations
      </Link>

      {/* Page title */}
      <div style={{ marginBottom: '20px' }}>
        <div className="tq">{session.query}</div>
        <div className="tm">
          <StatusBadge type="decision" value={session.gateway_decision} />
          {session.mad_routing && (
            <StatusBadge type="mad" value={session.mad_routing} />
          )}
          <span className="badge b-gray">
            {new Date(session.created_at).toLocaleString('en-US', {
              month: 'short', day: 'numeric',
              hour: '2-digit', minute: '2-digit',
            })}
          </span>
          {totalMs && (
            <span className="badge b-gray">{(totalMs / 1000).toFixed(1)}s total</span>
          )}
          <span className="tid" style={{ marginLeft: 4 }}>{session.id}</span>
        </div>
      </div>

      {/* Latency ribbon */}
      {totalMs && (
        <div className="lat-strip" style={{ marginBottom: 20 }}>
          <div className="li">
            <div className="ls">Gateway</div>
            <div className="lv">{gatewayMs}ms</div>
          </div>
          <div className="li">
            <div className="ls">LLM</div>
            <div className={`lv${!session.llm_answer ? ' sk' : ''}`}>
              {session.llm_answer ? `${llmMs}ms` : '—'}
            </div>
          </div>
          <div className="li">
            <div className="ls">MAD</div>
            <div className={`lv${!mad ? ' sk' : ''}`}>{mad ? `${madMs}ms` : '—'}</div>
          </div>
          <div className="li">
            <div className="ls">Judge</div>
            <div className={`lv${!mad ? ' sk' : ''}`}>{mad ? `${judgeMs}ms` : '—'}</div>
          </div>
          <div className="li">
            <div className="ls">Total</div>
            <div className="lv" style={{ color: 'var(--teal)' }}>
              {(totalMs / 1000).toFixed(2)}s
            </div>
          </div>
        </div>
      )}

      <div className="trace-wrap">
        {/* Timeline */}
        <div className="trace-tl">

          {/* Gateway Stage */}
          <Stage
            icon="✓"
            iconClass="si-ok"
            title="Gateway Layer"
            timeLabel={`0 → ${gatewayMs}ms`}
            badge={`Composite ${session.gateway_score.toFixed(2)} · ${session.gateway_decision}`}
            badgeClass={
              session.gateway_decision === 'PASS' ? 'b-gray'
              : session.gateway_decision === 'BLOCK' ? 'b-block'
              : 'b-esc'
            }
          >
            <div className="span-row">
              <div className="sdot" style={{ background: 'var(--blue)' }}/>
              <div style={{ flex: 1 }}>
                <div className="slbl">
                  PII Detection
                  <span className="smdl">DeBERTa-base NER · 57 entity types</span>
                </div>
                <div className="ssub">
                  {piiEntities.length > 0
                    ? `Detected: ${piiEntities.map(e => `${e.entity_type || e.type} "${e.text}"`).join(', ')}`
                    : 'No PII detected in query'}
                </div>
              </div>
              <div className="sval">score {piiScore.toFixed(2)}</div>
            </div>
            <div className="conn"/>
            <div className="span-row">
              <div className="sdot" style={{ background: 'var(--pink)' }}/>
              <div style={{ flex: 1 }}>
                <div className="slbl">
                  Jailbreak Detection
                  <span className="smdl">RoBERTa-base finetuned</span>
                </div>
                <div className="ssub">
                  {jbScore > 0.5 ? 'Jailbreak pattern detected' : 'No jailbreak pattern detected'}
                </div>
              </div>
              <div className="sval">score {jbScore.toFixed(2)}</div>
            </div>
            <div className="conn"/>
            <div className="span-row">
              <div className="sdot" style={{ background: 'var(--amber)' }}/>
              <div style={{ flex: 1 }}>
                <div className="slbl">
                  Prompt Injection
                  <span className="smdl">Llama-PG-2-86M finetuned</span>
                </div>
                <div className="ssub">
                  {piScore > 0.5 ? 'Injection pattern detected' : 'No injection pattern detected'}
                </div>
              </div>
              <div className="sval">score {piScore.toFixed(2)}</div>
            </div>
            <div className="conn"/>
            <div
              className="span-row"
              style={{
                borderColor: session.gateway_decision === 'PASS'
                  ? 'var(--teal-mid)'
                  : 'var(--red-mid)',
              }}
            >
              <div
                className="sdot"
                style={{
                  background: session.gateway_decision === 'PASS'
                    ? 'var(--teal)'
                    : 'var(--red)',
                }}
              />
              <div style={{ flex: 1 }}>
                <div
                  className="slbl"
                  style={{
                    color: session.gateway_decision === 'PASS'
                      ? 'var(--teal)'
                      : 'var(--red)',
                  }}
                >
                  Decision Engine — {session.gateway_decision}
                </div>
                <div className="ssub">
                  Composite {session.gateway_score.toFixed(3)} · (PII×0.3)+(JB×0.4)+(PI×0.3)
                </div>
              </div>
              <div className="sval">0ms</div>
            </div>
          </Stage>

          {/* RAG Stage */}
          {mad && (
            <Stage
              icon="R"
              iconClass="si-i"
              title="RAG Pipeline"
              timeLabel={`${gatewayMs}ms → ${gatewayMs + llmMs}ms`}
              badge={`${mad.evidence_pool?.length || 0} chunks · Recall@1 94.9%`}
              badgeClass="b-bank"
            >
              <div className="span-row">
                <div className="sdot" style={{ background: 'var(--teal)' }}/>
                <div style={{ flex: 1 }}>
                  <div className="slbl">
                    Embedding query
                    <span className="smdl">Qwen3-4B finetuned · 2560-dim</span>
                  </div>
                  <div className="ssub">Instruction-aware prefix · cosine similarity search</div>
                </div>
              </div>
              <div className="conn"/>
              <div className="span-row">
                <div className="sdot" style={{ background: 'var(--blue)' }}/>
                <div style={{ flex: 1 }}>
                  <div className="slbl">
                    Qdrant retrieval + BM25 hybrid + cross-encoder reranking
                  </div>
                  <div className="ssub">
                    Top-{mad.evidence_pool?.length || 0} chunks · tier-aware authority scoring
                  </div>
                </div>
              </div>
              {mad.evidence_pool?.length > 0 && (
                <div className="chunk-list">
                  {mad.evidence_pool.map((chunk, i) => (
                    <div key={chunk.chunk_id || i} className="chunk-item">
                      <span className="tier-badge">T0</span>
                      <div className="chunk-text">{chunk.text}</div>
                      <div className="chunk-score">{chunk.relevance_score?.toFixed(2)}</div>
                    </div>
                  ))}
                </div>
              )}
              {session.llm_answer && (
                <>
                  <div className="conn"/>
                  <div className="span-row">
                    <div className="sdot" style={{ background: 'var(--violet)' }}/>
                    <div style={{ flex: 1 }}>
                      <div className="slbl">
                        LLM candidate answer
                        <span className="smdl">qwen2.5:7b</span>
                      </div>
                      <div className="ssub">"{session.llm_answer.slice(0, 140)}…"</div>
                    </div>
                    <div className="sval">{llmMs}ms</div>
                  </div>
                </>
              )}
            </Stage>
          )}

          {/* MAD running in background */}
          {!mad && session.llm_answer && session.gateway_decision !== 'BLOCK' && (
            <Stage
              icon="·"
              iconClass="si-w pulsing"
              title="MAD Pipeline"
              timeLabel="running…"
              badge="background task"
              badgeClass="b-gray"
            >
              <div className="span-row">
                <div className="sdot pulsing" style={{ background: 'var(--teal)' }}/>
                <div style={{ flex: 1 }}>
                  <div className="slbl">Multi-Agent Debate running in background</div>
                  <div className="ssub">
                    ~5–10 min · 2 cycles · Ollama qwen2.5:7b · auto-refreshing every 10s
                  </div>
                </div>
              </div>
            </Stage>
          )}

          {/* MAD Stage */}
          {mad && (
            <Stage
              icon="M"
              iconClass="si-w"
              title="MAD Pipeline"
              timeLabel={`${gatewayMs + llmMs}ms → ${gatewayMs + llmMs + madMs}ms`}
              badge={`${(mad.claims ?? mad.debate_cycles ?? []).length} claims · R0+R1 debate`}
              badgeClass={
                session.mad_routing === 'DELIVER' ? 'b-pass'
                : session.mad_routing === 'HARD_BLOCK' ? 'b-block'
                : 'b-esc'
              }
            >
              {/* V4MAD shape: claims + judge_verdicts */}
              {mad.claims?.length > 0 && (
                <div className="claim-list">
                  {mad.claims.map((claim, i) => {
                    const jv = mad.judge_verdicts?.find(v => v.claim_id === claim.claim_id)
                    const score = jv?.score
                    const scoreLabel = score === 1.0 ? 'SUPPORTED' : score === 0.5 ? 'PARTIAL' : score === 0.0 ? 'NOT_SUPPORTED' : claim.verdict
                    return (
                      <div key={claim.claim_id ?? i} className="claim-card">
                        <div className="claim-hdr">
                          <div className="claim-text">{claim.claim_text}</div>
                          <div style={{ display: 'flex', gap: 4, alignItems: 'center', flexShrink: 0 }}>
                            {claim.is_material && (
                              <span className="ev-tag" style={{ fontSize: '9px', background: 'var(--amber-dim)', color: 'var(--amber-hi)' }}>material</span>
                            )}
                            <span className={`badge ${claimCls(scoreLabel)}`} style={{ fontSize: '10px' }}>
                              {scoreLabel || 'pending'}
                            </span>
                          </div>
                        </div>
                        <div className="claim-ev">
                          Agent A R1 · conf {((claim.confidence || 0) * 100).toFixed(0)}%
                          {jv && ` · judge score ${(score * 100).toFixed(0)}%`}
                        </div>
                        {jv?.reasoning && (
                          <div className="claim-ev" style={{ marginTop: 2, color: 'var(--text-muted)', fontStyle: 'italic' }}>
                            {jv.reasoning.slice(0, 120)}{jv.reasoning.length > 120 ? '…' : ''}
                          </div>
                        )}
                      </div>
                    )
                  })}
                </div>
              )}

              {/* Legacy shape: debate_cycles (kept for old sessions) */}
              {!mad.claims && mad.debate_cycles?.map((cycle, ci) => (
                <div key={ci}>
                  <div className="cyc-lbl">
                    Cycle {cycle.cycle_number ?? (cycle.cycle_index ?? 0) + 1}
                  </div>
                  {cycle.agent_a_report?.length > 0 && (
                    <div className="claim-list">
                      {cycle.agent_a_report.map((claim, i) => (
                        <div key={i} className="claim-card">
                          <div className="claim-hdr">
                            <div className="claim-text">{claim.claim_text}</div>
                            <span className={`badge ${claimCls(claim.verdict)}`} style={{ fontSize: '10px' }}>
                              {claim.verdict || 'pending'}
                            </span>
                          </div>
                          <div className="claim-ev">
                            Agent A · confidence {((claim.confidence || 0) * 100).toFixed(0)}%
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                  {cycle.agent_b_challenges?.map((ch, i) => (
                    <div key={i} className="agent-box ab">
                      <div className="aw">
                        <div className="adot adot-b"/>
                        Agent B — Challenge ({ch.challenge_type})
                      </div>
                      <div className="at">"{ch.challenge_text}"</div>
                    </div>
                  ))}
                </div>
              ))}

              {/* Judge verdict summary — shown for both shapes */}
              <div className="judge-box">
                <div className="jt">Judge — Final Verdict</div>
                <div className="jtext">
                  MAD routing: {mad.routing_decision} · confidence{' '}
                  {(((mad.aggregate_confidence ?? mad.confidence_score) || 0) * 100).toFixed(0)}%
                  {mad.claims?.length > 0 && ` · ${mad.claims.length} claim${mad.claims.length !== 1 ? 's' : ''} evaluated`}
                </div>
                <div className="ev-tags" style={{ marginTop: '8px' }}>
                  <StatusBadge type="mad" value={mad.routing_decision} />
                </div>
              </div>

              {mad.debate_transcript && (
                <div>
                  <button
                    type="button"
                    onClick={() => setTranscriptOpen(o => !o)}
                    style={{
                      background: 'none',
                      border: 'none',
                      cursor: 'pointer',
                      fontFamily: 'var(--font-mono)',
                      fontSize: '10px',
                      color: 'var(--text-muted)',
                      padding: '8px 0',
                      width: '100%',
                      textAlign: 'left',
                    }}
                  >
                    {transcriptOpen ? '▲' : '▼'} raw transcript (debug)
                  </button>
                  <AnimatePresence>
                    {transcriptOpen && (
                      <motion.pre
                        initial={{ height: 0, opacity: 0 }}
                        animate={{ height: 'auto', opacity: 1 }}
                        exit={{ height: 0, opacity: 0 }}
                        transition={{ duration: 0.2 }}
                        style={{
                          fontFamily: 'var(--font-mono)',
                          fontSize: '10px',
                          color: 'var(--text-sec)',
                          whiteSpace: 'pre-wrap',
                          lineHeight: 1.6,
                          padding: '8px',
                          background: 'var(--bg-surface)',
                          border: '1px solid var(--border)',
                          borderRadius: '4px',
                          overflow: 'hidden',
                        }}
                      >
                        {mad.debate_transcript}
                      </motion.pre>
                    )}
                  </AnimatePresence>
                </div>
              )}
            </Stage>
          )}

          {/* Gateway blocked — skip remaining */}
          {!session.llm_answer && !mad && (
            <Stage
              icon="✕"
              iconClass="si-bl"
              title="LLM + MAD"
              timeLabel="skipped"
              badgeClass="b-block"
              badge="BLOCKED"
              defaultOpen={false}
              skipped
            >
              <div className="span-row">
                <div className="slbl" style={{ color: 'var(--text-muted)' }}>
                  Query was blocked at the gateway. No LLM or MAD processing performed.
                </div>
              </div>
            </Stage>
          )}
        </div>

        {/* Sidebar */}
        <div className="trace-sb">

          {/* Confidence — promoted to top */}
          <div className="ts-sec">
            <div className="ts-title">Confidence</div>
            <div className="conf-box">
              <div className="conf-lbl">MAD confidence score</div>
              <div className={`conf-v ${confClass}`}>
                {confidence != null
                  ? `${(confidence * 100).toFixed(0)}%`
                  : '—'}
              </div>
              {confidence == null && (
                <div className="conf-n">
                  {session.gateway_decision === 'BLOCK' ? 'blocked at gateway' : 'MAD pending…'}
                </div>
              )}
            </div>
          </div>

          {/* Verdict */}
          <div className={`vdict ${verdictClass(session.mad_routing || session.gateway_decision)}`}>
            {session.mad_routing
              ? `${session.mad_routing} · via MAD`
              : session.gateway_decision === 'BLOCK'
              ? 'BLOCK · gateway'
              : 'MAD pending…'}
          </div>

          {/* Threat scores */}
          <div className="ts-sec">
            <div className="ts-title">Threat Scores</div>
            <div className="score-block">
              <ScoreBar label="PII" value={piiScore} colorMode="category" category="pii" />
              <ScoreBar label="Jailbreak" value={jbScore} colorMode="category" category="jb" />
              <ScoreBar label="Prompt Inj." value={piScore} colorMode="category" category="pi" />
              <div style={{ height: 1, background: 'var(--border)', margin: '2px 0' }}/>
              <ScoreBar
                label="Composite"
                value={session.gateway_score}
                colorMode="fixed"
                color="var(--teal)"
              />
            </div>
          </div>

          {/* Latency breakdown */}
          <div className="ts-sec">
            <div className="ts-title">Latency</div>
            <div className="lat-tbl">
              <div className="lt-row">
                <span className="lt-s">Gateway</span>
                <span className="lt-v">{gatewayMs}ms</span>
              </div>
              <div className="lt-row">
                <span className="lt-s">LLM gen</span>
                <span className="lt-v" style={!session.llm_answer ? { color: 'var(--text-muted)' } : {}}>
                  {session.llm_answer ? `${llmMs}ms` : '—'}
                </span>
              </div>
              <div className="lt-row">
                <span className="lt-s">MAD</span>
                <span className="lt-v" style={!mad ? { color: 'var(--text-muted)' } : {}}>
                  {mad ? `${madMs}ms` : '—'}
                </span>
              </div>
              <div className="lt-row">
                <span className="lt-s">Judge</span>
                <span className="lt-v" style={!mad ? { color: 'var(--text-muted)' } : {}}>
                  {mad ? `${judgeMs}ms` : '—'}
                </span>
              </div>
              <div className="lt-row">
                <span className="lt-s">Total</span>
                <span className="lt-v" style={{ color: 'var(--teal)' }}>
                  {totalMs ? `${(totalMs / 1000).toFixed(2)}s` : '—'}
                </span>
              </div>
            </div>
          </div>

          {/* Actions */}
          <div className="ts-sec">
            <div className="ts-title">Actions</div>
            <button type="button" className="act-btn lf-btn">Open in Langfuse →</button>
            <button type="button" className="act-btn">Export trace as JSON</button>
          </div>

          <CSEPanel cseData={cseData} />

          <FeedbackPanel sessionId={session.id} />
        </div>
      </div>
    </div>
  )
}
