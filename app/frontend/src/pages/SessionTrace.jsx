import { useEffect, useState } from 'react'
import { useParams, Link, useNavigate } from 'react-router-dom'
import { getSession, submitFeedback } from '../api/client'

function decisionBadge(d) {
  if (d === 'BLOCK') return 'b-block'
  if (d === 'ESCALATE') return 'b-esc'
  return 'b-pass'
}
function madBadge(r) {
  if (!r) return 'b-gray'
  if (r === 'DELIVER') return 'b-deliver'
  if (r === 'RETRY') return 'b-retry'
  if (r === 'HARD_BLOCK') return 'b-hard-block'
  if (r === 'HUMAN_REVIEW') return 'b-human-review'
  return 'b-gray'
}
function verdictClass(d) {
  if (d === 'BLOCK' || d === 'HARD_BLOCK') return 'vd-block'
  if (d === 'ESCALATE' || d === 'RETRY') return 'vd-esc'
  return 'vd-pass'
}
function claimBadge(verdict) {
  if (!verdict) return 'b-gray'
  if (verdict === 'SUPPORTED') return 'b-pass'
  if (verdict === 'UNSUPPORTED' || verdict === 'NOT_SUPPORTED') return 'b-block'
  if (verdict === 'NEEDS_CAVEAT' || verdict === 'MISSING_CAVEAT') return 'b-esc'
  return 'b-gray'
}

function Stage({ icon, iconClass, title, timeLabel, badge, badgeClass, children, defaultOpen = true }) {
  const [open, setOpen] = useState(defaultOpen)
  return (
    <div className="stage">
      <div className="stage-hdr" onClick={() => setOpen(o => !o)}>
        <div className={`si ${iconClass}`}>{icon}</div>
        <div className="sn">{title}</div>
        <div className="st-time">{timeLabel}</div>
        {badge && <span className={`badge ${badgeClass || 'b-gray'}`} style={{ fontSize: '10px' }}>{badge}</span>}
        <span className={`stg${open ? ' open' : ''}`}>▲</span>
      </div>
      {open && <div className="stage-body">{children}</div>}
    </div>
  )
}

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
        <div className="ts-title">Feedback</div>
        <div style={{ padding: '10px', background: 'var(--green-dim)', border: '1px solid var(--green-mid)', borderRadius: '5px', fontFamily: 'var(--font-mono)', fontSize: '11px', color: 'var(--green)' }}>
          Submitted. Thank you.
        </div>
      </div>
    )
  }

  return (
    <div className="ts-sec">
      <div className="ts-title">Feedback</div>
      <div style={{ display: 'flex', gap: '2px', marginBottom: '10px' }}>
        {[1,2,3,4,5].map(n => (
          <button key={n} onClick={() => setRating(n)} style={{ background: 'none', border: 'none', cursor: 'pointer', fontSize: '18px', color: n <= rating ? 'var(--amber)' : 'var(--border-hi)', padding: '0 2px' }}>★</button>
        ))}
      </div>
      <div style={{ display: 'flex', gap: '4px', marginBottom: '10px' }}>
        {['correct','partial','incorrect'].map(l => (
          <button key={l} onClick={() => setLabel(l)}
            style={{ padding: '4px 10px', fontFamily: 'var(--font-mono)', fontSize: '10px', borderRadius: '3px', border: '1px solid', cursor: 'pointer', background: label === l ? 'var(--teal-dim)' : 'var(--bg-hover)', color: label === l ? 'var(--teal)' : 'var(--text-sec)', borderColor: label === l ? 'var(--teal-mid)' : 'var(--border-md)' }}>
            {l}
          </button>
        ))}
      </div>
      <textarea
        value={comment}
        onChange={e => setComment(e.target.value)}
        placeholder="Optional note…"
        style={{ width: '100%', padding: '6px 8px', fontFamily: 'var(--font-mono)', fontSize: '11px', background: 'var(--bg-surface)', border: '1px solid var(--border-md)', borderRadius: '3px', color: 'var(--text-primary)', resize: 'none', outline: 'none', marginBottom: '8px' }}
        rows={2}
      />
      <button className="act-btn lf-btn" disabled={!rating || loading} onClick={handleSubmit} style={{ marginBottom: 0 }}>
        {loading ? 'Submitting…' : 'Submit Feedback'}
      </button>
    </div>
  )
}

export default function SessionTrace() {
  const { id } = useParams()
  const navigate = useNavigate()
  const [session, setSession] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [transcriptOpen, setTranscriptOpen] = useState(false)

  useEffect(() => {
    getSession(id)
      .then(setSession)
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [id])

  // Poll for MAD completion if session has LLM answer but no MAD result yet
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

  if (loading) return (
    <div style={{ textAlign: 'center', padding: '60px', fontFamily: 'var(--font-mono)', fontSize: '12px', color: 'var(--text-muted)' }}>
      Loading trace…
    </div>
  )

  if (error || !session) return (
    <div>
      <div style={{ padding: '12px', background: 'var(--red-dim)', border: '1px solid var(--red-mid)', borderRadius: '6px', fontFamily: 'var(--font-mono)', fontSize: '12px', color: 'var(--red)', marginBottom: '12px' }}>
        {error || 'Session not found.'}
      </div>
      <button className="back-btn" onClick={() => navigate('/')}>← Back to conversations</button>
    </div>
  )

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

  return (
    <div>
      <Link to="/" className="back-btn">
        <svg width="12" height="12" viewBox="0 0 12 12" fill="none"><path d="M8 2L4 6l4 4" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/></svg>
        Back to conversations
      </Link>

      <div className="tq">{session.query}</div>

      <div className="tm">
        <span className={`badge ${decisionBadge(session.gateway_decision)}`}>{session.gateway_decision}</span>
        {session.mad_routing && <span className={`badge ${madBadge(session.mad_routing)}`}>MAD: {session.mad_routing}</span>}
        <span className="badge b-gray">{new Date(session.created_at).toLocaleString('en-US', { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' })}</span>
        {totalMs && <span className="badge b-gray">{(totalMs / 1000).toFixed(1)}s total</span>}
        <span className="tid">{session.id}</span>
      </div>

      {totalMs && (
        <div className="lat-strip">
          <div className="li"><div className="ls">Gateway</div><div className="lv">{gatewayMs}ms</div></div>
          <div className="ld"/>
          <div className="li"><div className="ls">LLM</div><div className={`lv${!session.llm_answer ? ' sk' : ''}`}>{session.llm_answer ? llmMs + 'ms' : '—'}</div></div>
          <div className="ld"/>
          <div className="li"><div className="ls">MAD</div><div className={`lv${!mad ? ' sk' : ''}`}>{mad ? madMs + 'ms' : '—'}</div></div>
          <div className="ld"/>
          <div className="li"><div className="ls">Judge</div><div className={`lv${!mad ? ' sk' : ''}`}>{mad ? judgeMs + 'ms' : '—'}</div></div>
          <div className="ld"/>
          <div className="li"><div className="ls">Total</div><div className="lv" style={{ color: 'var(--teal)' }}>{(totalMs / 1000).toFixed(2)}s</div></div>
        </div>
      )}

      <div className="trace-wrap">
        <div className="trace-tl">

          {/* Gateway Stage */}
          <Stage
            icon="✓" iconClass="si-ok" title="Gateway Layer"
            timeLabel={`0 → ${gatewayMs}ms`}
            badge={`Composite ${session.gateway_score.toFixed(2)} · ${session.gateway_decision}`}
            badgeClass={session.gateway_decision === 'PASS' ? 'b-gray' : session.gateway_decision === 'BLOCK' ? 'b-block' : 'b-esc'}
          >
            <div className="span-row">
              <div className="sdot" style={{ background: 'var(--blue)' }}/>
              <div style={{ flex: 1 }}>
                <div className="slbl">PII Detection <span className="smdl">DeBERTa-base NER · 57 entity types</span></div>
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
                <div className="slbl">Jailbreak Detection <span className="smdl">RoBERTa-base finetuned</span></div>
                <div className="ssub">{jbScore > 0.5 ? 'Jailbreak pattern detected' : 'No jailbreak pattern detected'}</div>
              </div>
              <div className="sval">score {jbScore.toFixed(2)}</div>
            </div>
            <div className="conn"/>
            <div className="span-row">
              <div className="sdot" style={{ background: 'var(--amber)' }}/>
              <div style={{ flex: 1 }}>
                <div className="slbl">Prompt Injection <span className="smdl">Llama-PG-2-86M finetuned</span></div>
                <div className="ssub">{piScore > 0.5 ? 'Injection pattern detected' : 'No injection pattern detected'}</div>
              </div>
              <div className="sval">score {piScore.toFixed(2)}</div>
            </div>
            <div className="conn"/>
            <div className="span-row" style={{ borderColor: session.gateway_decision === 'PASS' ? 'var(--teal-mid)' : 'var(--red-mid)' }}>
              <div className="sdot" style={{ background: session.gateway_decision === 'PASS' ? 'var(--teal)' : 'var(--red)' }}/>
              <div style={{ flex: 1 }}>
                <div className="slbl" style={{ color: session.gateway_decision === 'PASS' ? 'var(--teal)' : 'var(--red)' }}>
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
              icon="R" iconClass="si-i" title="RAG Pipeline"
              timeLabel={`${gatewayMs}ms → ${gatewayMs + llmMs}ms`}
              badge={`${mad.evidence_pool?.length || 0} chunks · Recall@1 94.9%`}
              badgeClass="b-bank"
            >
              <div className="span-row">
                <div className="sdot" style={{ background: 'var(--teal)' }}/>
                <div style={{ flex: 1 }}>
                  <div className="slbl">Embedding query <span className="smdl">Qwen3-4B finetuned · 2560-dim</span></div>
                  <div className="ssub">Instruction-aware prefix · cosine similarity search</div>
                </div>
              </div>
              <div className="conn"/>
              <div className="span-row">
                <div className="sdot" style={{ background: 'var(--blue)' }}/>
                <div style={{ flex: 1 }}>
                  <div className="slbl">Qdrant retrieval + BM25 hybrid + cross-encoder reranking</div>
                  <div className="ssub">Top-{mad.evidence_pool?.length || 0} chunks · tier-aware authority scoring</div>
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
                      <div className="slbl">LLM candidate answer <span className="smdl">qwen2.5:7b</span></div>
                      <div className="ssub">"{session.llm_answer.slice(0, 140)}…"</div>
                    </div>
                    <div className="sval">{llmMs}ms</div>
                  </div>
                </>
              )}
            </Stage>
          )}

          {/* MAD pending — running in background */}
          {!mad && session.llm_answer && session.gateway_decision !== 'BLOCK' && (
            <Stage icon="·" iconClass="si-w pulsing" title="MAD Pipeline"
              timeLabel="running…" badge="background task" badgeClass="b-gray">
              <div className="span-row">
                <div className="sdot pulsing" style={{ background: 'var(--teal)' }}/>
                <div style={{ flex: 1 }}>
                  <div className="slbl">Multi-Agent Debate running in background</div>
                  <div className="ssub">~5-10 min · 2 cycles · Ollama qwen2.5:7b · auto-refreshing every 10s</div>
                </div>
              </div>
            </Stage>
          )}

          {/* MAD Stage */}
          {mad && (
            <Stage
              icon="M" iconClass="si-w" title="MAD Pipeline"
              timeLabel={`${gatewayMs + llmMs}ms → ${gatewayMs + llmMs + madMs}ms`}
              badge={`${mad.debate_cycles?.length || 0} cycles · ${mad.debate_cycles?.reduce((a, c) => a + (c.agent_b_challenges?.length || 0), 0) || 0} challenges`}
              badgeClass={session.mad_routing === 'DELIVER' ? 'b-pass' : session.mad_routing === 'HARD_BLOCK' ? 'b-block' : 'b-esc'}
            >
              {mad.debate_cycles?.length === 0 && (
                <div className="span-row">
                  <div className="sdot" style={{ background: 'var(--text-muted)' }}/>
                  <div className="slbl" style={{ color: 'var(--text-muted)' }}>No debate cycles — routed to human review.</div>
                </div>
              )}

              {mad.debate_cycles?.map((cycle, ci) => (
                <div key={ci}>
                  <div className="cyc-lbl">Cycle {cycle.cycle_number ?? (cycle.cycle_index ?? 0) + 1}</div>

                  {cycle.agent_a_report?.length > 0 && (
                    <div>
                      <div className="claim-list">
                        {cycle.agent_a_report.map((claim, i) => (
                          <div key={i} className="claim-card">
                            <div className="claim-hdr">
                              <div className="claim-text">{claim.claim_text}</div>
                              <span className={`badge ${claimBadge(claim.verdict)}`} style={{ fontSize: '10px' }}>{claim.verdict || 'pending'}</span>
                            </div>
                            <div className="claim-ev">Agent A · confidence {((claim.confidence || 0) * 100).toFixed(0)}%</div>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {cycle.agent_b_challenges?.map((ch, i) => (
                    <div key={i} className="agent-box ab">
                      <div className="aw"><div className="adot adot-b"/>Agent B — Challenge ({ch.challenge_type})</div>
                      <div className="at">"{ch.challenge_text}"</div>
                      {ch.suggested_verdict && (
                        <div className="ev-tags"><span className="ev-tag">suggests: {ch.suggested_verdict}</span></div>
                      )}
                    </div>
                  ))}

                  {cycle.agent_a_revised?.map((claim, i) => (
                    <div key={i} className="agent-box aa">
                      <div className="aw"><div className="adot adot-a"/>Agent A — Revised</div>
                      <div className="at">"{claim.claim_text}"</div>
                      <div className="ev-tags">
                        <span className={`ev-tag badge ${claimBadge(claim.verdict)}`}>{claim.verdict || 'revised'}</span>
                        <span className="ev-tag">conf {((claim.confidence || 0) * 100).toFixed(0)}%</span>
                      </div>
                    </div>
                  ))}
                </div>
              ))}

              {mad.debate_cycles?.length > 0 && (
                <div className="judge-box">
                  <div className="jt">Judge — Final Verdict</div>
                  <div className="jtext">
                    MAD routing: {mad.routing_decision} · confidence {(((mad.aggregate_confidence ?? mad.confidence_score) || 0) * 100).toFixed(0)}%
                    {mad.debate_cycles?.length > 0 && ` · ${mad.debate_cycles.length} cycle${mad.debate_cycles.length !== 1 ? 's' : ''} completed`}
                  </div>
                  <div className="ev-tags" style={{ marginTop: '8px' }}>
                    <span className={`badge ${madBadge(mad.routing_decision)}`}>{mad.routing_decision}</span>
                  </div>
                </div>
              )}

              {/* Raw transcript collapsible */}
              {mad.debate_transcript && (
                <div>
                  <button onClick={() => setTranscriptOpen(o => !o)} style={{ background: 'none', border: 'none', cursor: 'pointer', fontFamily: 'var(--font-mono)', fontSize: '10px', color: 'var(--text-muted)', padding: '6px 0', width: '100%', textAlign: 'left' }}>
                    {transcriptOpen ? '▲' : '▼'} raw transcript (debug)
                  </button>
                  {transcriptOpen && (
                    <pre style={{ fontFamily: 'var(--font-mono)', fontSize: '10px', color: 'var(--text-sec)', whiteSpace: 'pre-wrap', lineHeight: 1.6, padding: '8px', background: 'var(--bg-surface)', border: '1px solid var(--border)', borderRadius: '4px' }}>
                      {mad.debate_transcript}
                    </pre>
                  )}
                </div>
              )}
            </Stage>
          )}

          {/* Gateway blocked — no further stages */}
          {!session.llm_answer && !mad && (
            <Stage icon="✕" iconClass="si-bl" title="LLM + MAD" timeLabel="skipped" badgeClass="b-block" badge="BLOCKED" defaultOpen={false}>
              <div className="span-row skipped">
                <div className="slbl" style={{ color: 'var(--text-muted)' }}>Query was blocked at the gateway. No LLM or MAD processing performed.</div>
              </div>
            </Stage>
          )}
        </div>

        {/* Sidebar */}
        <div className="trace-sb">
          <div className="ts-sec">
            <div className="ts-title">Threat scores</div>
            <div className="score-block">
              <div className="s-row">
                <div className="s-meta"><span className="s-key">PII</span><span className="s-val">{piiScore.toFixed(2)}</span></div>
                <div className="s-track"><div className="s-fill f-pii" style={{ width: `${piiScore * 100}%` }}/></div>
              </div>
              <div className="s-row">
                <div className="s-meta"><span className="s-key">Jailbreak</span><span className="s-val">{jbScore.toFixed(2)}</span></div>
                <div className="s-track"><div className="s-fill f-jb" style={{ width: `${jbScore * 100}%` }}/></div>
              </div>
              <div className="s-row">
                <div className="s-meta"><span className="s-key">Prompt inj.</span><span className="s-val">{piScore.toFixed(2)}</span></div>
                <div className="s-track"><div className="s-fill f-pi" style={{ width: `${piScore * 100}%` }}/></div>
              </div>
              <div style={{ height: '1px', background: 'var(--border)', margin: '4px 0' }}/>
              <div className="s-row">
                <div className="s-meta"><span className="s-key">Composite</span><span className="s-val" style={{ color: 'var(--teal)' }}>{session.gateway_score.toFixed(2)}</span></div>
                <div className="s-track"><div className="s-fill f-comp" style={{ width: `${session.gateway_score * 100}%` }}/></div>
              </div>
            </div>
          </div>

          <div className={`vdict ${verdictClass(session.mad_routing || session.gateway_decision)}`}>
            {session.mad_routing
              ? `${session.mad_routing} · via MAD`
              : session.gateway_decision === 'BLOCK'
              ? 'BLOCK'
              : 'MAD pending…'}
          </div>

          <div className="ts-sec">
            <div className="ts-title">Latency breakdown</div>
            <div className="lat-tbl">
              <div className="lt-row"><span className="lt-s">Gateway</span><span className="lt-v">{gatewayMs}ms</span></div>
              <div className="lt-row"><span className="lt-s">LLM gen</span><span className="lt-v" style={!session.llm_answer ? { color: 'var(--text-muted)' } : {}}>{session.llm_answer ? llmMs + 'ms' : '—'}</span></div>
              <div className="lt-row"><span className="lt-s">MAD</span><span className="lt-v" style={!mad ? { color: 'var(--text-muted)' } : {}}>{mad ? madMs + 'ms' : '—'}</span></div>
              <div className="lt-row"><span className="lt-s">Judge</span><span className="lt-v" style={!mad ? { color: 'var(--text-muted)' } : {}}>{mad ? judgeMs + 'ms' : '—'}</span></div>
              <div className="lt-row"><span className="lt-s">Total</span><span className="lt-v" style={{ color: 'var(--teal)' }}>{totalMs ? (totalMs / 1000).toFixed(2) + 's' : '—'}</span></div>
            </div>
          </div>

          <div className="ts-sec">
            <div className="ts-title">Confidence engine</div>
            <div className="conf-box">
              <div className="conf-lbl">Confidence score</div>
              <div className="conf-v">{(mad?.aggregate_confidence ?? mad?.confidence_score) != null ? (((mad?.aggregate_confidence ?? mad?.confidence_score)) * 100).toFixed(0) + '%' : '—'}</div>
              {(mad?.aggregate_confidence ?? mad?.confidence_score) == null && <div className="conf-n">MAD pending…</div>}
            </div>
          </div>

          <div className="ts-sec">
            <div className="ts-title">Actions</div>
            <button className="act-btn lf-btn">Open in Langfuse →</button>
            <button className="act-btn">Export trace as JSON</button>
          </div>

          <FeedbackPanel sessionId={session.id} />
        </div>
      </div>
    </div>
  )
}
