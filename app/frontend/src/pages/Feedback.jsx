import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { motion, useReducedMotion } from 'framer-motion'
import { getSessions, submitFeedback } from '../api/client'
import PageHeader from '../components/PageHeader'
import SectionHeader from '../components/SectionHeader'
import EmptyState, { InboxClearIcon, ErrorState } from '../components/EmptyState'
import LoadingSkeleton from '../components/LoadingSkeleton'
import StatusBadge from '../components/StatusBadge'

/* ─── Score helpers ──────────────────────────────────────────── */
function piiScore(s) { return s.gateway_payload?.scores?.pii ?? s.gateway_payload?.pii_score ?? 0 }
function jbScore(s)  { return s.gateway_payload?.scores?.jailbreak ?? s.gateway_payload?.jailbreak_score ?? 0 }
function piScore(s)  { return s.gateway_payload?.scores?.prompt_injection ?? s.gateway_payload?.injection_score ?? 0 }

function timeSince(iso) {
  const diff = Date.now() - new Date(iso).getTime()
  const mins = Math.floor(diff / 60000)
  if (mins < 1) return 'just now'
  if (mins < 60) return `${mins}m ago`
  const hrs = Math.floor(mins / 60)
  return hrs < 24 ? `${hrs}h ago` : `${Math.floor(hrs / 24)}d ago`
}

/* ─── Queue stat card ───────────────────────────────────────── */
function QueueStatCard({ title, children }) {
  return (
    <div className="panel" style={{ marginBottom: 12 }}>
      <div className="panel-hdr">
        <div className="panel-title">{title}</div>
      </div>
      <div className="panel-body">
        {children}
      </div>
    </div>
  )
}

function StatRow({ label, value, color }) {
  return (
    <div style={{
      display: 'flex',
      justifyContent: 'space-between',
      alignItems: 'center',
      padding: '7px 0',
      borderBottom: '1px solid var(--border)',
      fontSize: 13,
      fontFamily: 'var(--font-ui)',
    }}>
      <span style={{ color: 'var(--text-sec)' }}>{label}</span>
      <span style={{ color: color || 'var(--text-primary)', fontWeight: 600, fontFamily: 'var(--font-mono)' }}>
        {value}
      </span>
    </div>
  )
}

/* ─── Individual feedback card ───────────────────────────────── */
function FbItem({ item, onMark }) {
  const [marked, setMarked] = useState(null)
  const [note, setNote]     = useState('')
  const [saving, setSaving] = useState(false)
  const shouldReduceMotion  = useReducedMotion()

  const pii  = piiScore(item)
  const jb   = jbScore(item)
  const pi   = piScore(item)
  const comp = item.gateway_score

  const itemClass = item.gateway_decision === 'BLOCK' ? 'fb-item fb-block' : 'fb-item fb-esc'

  async function handleMark(label) {
    setMarked(label)
    setSaving(true)
    try {
      await submitFeedback(item.id, {
        rating:  label === 'c' ? 5 : 2,
        label:   label === 'c' ? 'correct' : 'incorrect',
        comment: note,
      })
      onMark?.(item.id, label)
    } finally {
      setSaving(false)
    }
  }

  if (marked) {
    return (
      <motion.div
        className={itemClass}
        initial={shouldReduceMotion ? false : { opacity: 1 }}
        exit={shouldReduceMotion ? {} : { opacity: 0, height: 0, marginBottom: 0, overflow: 'hidden' }}
        transition={{ duration: 0.25 }}
        style={{ borderLeftColor: marked === 'c' ? 'var(--green)' : 'var(--text-muted)', opacity: 0.5 }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, fontFamily: 'var(--font-ui)', fontSize: 12 }}>
          <span className={marked === 'c' ? 'badge b-pass' : 'badge b-gray'}>
            {marked === 'c' ? '✓ Confirmed correct' : '✕ Marked false positive'}
          </span>
          <span style={{ color: 'var(--text-muted)' }}>{item.query.slice(0, 60)}…</span>
        </div>
      </motion.div>
    )
  }

  return (
    <div className={itemClass}>
      <div className="fb-top">
        <div className="fb-q">{item.query}</div>
        <div className="fb-bd">
          <StatusBadge type="decision" value={item.gateway_decision} />
          {item.mad_routing && (
            <StatusBadge type="mad" value={item.mad_routing} />
          )}
          <span style={{ fontSize: 11, fontFamily: 'var(--font-ui)', color: 'var(--text-muted)', marginTop: 2 }}>
            {timeSince(item.created_at)}
          </span>
        </div>
      </div>

      <div className="fb-r">
        Composite score {comp.toFixed(2)} · {item.gateway_decision === 'BLOCK' ? 'block' : 'escalation'} zone
      </div>

      <div style={{ display: 'flex', gap: 4, marginBottom: 12, flexWrap: 'wrap' }}>
        <span className="sc">PII {pii.toFixed(2)}</span>
        <span className="sc">JB {jb.toFixed(2)}</span>
        <span className="sc">PI {pi.toFixed(2)}</span>
        <span className="sc" style={{ fontWeight: 600, color: 'var(--text-primary)' }}>∑ {comp.toFixed(2)}</span>
      </div>

      <div className="fb-acts">
        <button
          type="button"
          className={`fb-btn${marked === 'c' ? ' correct' : ''}`}
          onClick={() => handleMark('c')}
          disabled={!!marked || saving}
        >
          <svg width="11" height="11" viewBox="0 0 11 11" fill="none" style={{ marginRight: 4 }}>
            <path d="M1.5 5.5l3 3 5-5" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
          </svg>
          Confirmed correct
        </button>
        <button
          type="button"
          className={`fb-btn${marked === 'f' ? ' fp' : ''}`}
          onClick={() => handleMark('f')}
          disabled={!!marked || saving}
        >
          <svg width="11" height="11" viewBox="0 0 11 11" fill="none" style={{ marginRight: 4 }}>
            <path d="M2 2l7 7M9 2L2 9" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/>
          </svg>
          False positive
        </button>
        <input
          className="fb-note"
          placeholder="Add analyst note…"
          value={note}
          onChange={e => setNote(e.target.value)}
          disabled={!!marked}
        />
        <Link to={`/sessions/${item.id}`} className="fb-tlink">
          View trace →
        </Link>
      </div>
    </div>
  )
}

/* ─── Main page ──────────────────────────────────────────────── */
export default function Feedback() {
  const [sessions, setSessions]   = useState([])
  const [loading, setLoading]     = useState(true)
  const [error, setError]         = useState(null)
  const [resolved, setResolved]   = useState([])

  function load() {
    setLoading(true)
    setError(null)
    getSessions()
      .then(all => {
        const queue = all.filter(s =>
          s.gateway_decision === 'BLOCK' || s.gateway_decision === 'ESCALATE'
        )
        setSessions(queue)
      })
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }

  useEffect(() => { load() }, [])

  function handleMark(id, label) {
    setResolved(r => [...r, { id, label }])
    setSessions(prev => prev.filter(s => s.id !== id))
  }

  const confirmed = resolved.filter(r => r.label === 'c').length
  const falsePos  = resolved.filter(r => r.label === 'f').length
  const fpRate    = resolved.length ? ((falsePos / resolved.length) * 100).toFixed(0) : '—'
  const blocked   = sessions.filter(s => s.gateway_decision === 'BLOCK').length
  const escalated = sessions.filter(s => s.gateway_decision === 'ESCALATE').length

  return (
    <div>
      <PageHeader
        title="Review Queue"
        sub="Human analyst review — blocked and escalated sessions pending disposition"
        actions={
          sessions.length > 0 && !loading && (
            <span className="badge b-esc" style={{ fontSize: 11 }}>
              {sessions.length} pending
            </span>
          )
        }
      />

      <div className="fb-layout">
        {/* ── Queue ─────────────────────────────── */}
        <div>
          {loading && <LoadingSkeleton type="cards" count={3} />}

          {error && (
            <ErrorState
              title="Failed to load review queue"
              message={error}
              onRetry={load}
            />
          )}

          {!loading && !error && (
            <>
              <SectionHeader label="Pending Review" count={sessions.length} />

              {sessions.length === 0 ? (
                <div className="panel" style={{ marginTop: 4 }}>
                  <EmptyState
                    icon={<InboxClearIcon />}
                    title="Review queue is clear"
                    description="No sessions pending review. Submit queries from Test Query — blocked or escalated sessions will appear here."
                  />
                </div>
              ) : (
                <div style={{ display: 'flex', flexDirection: 'column' }}>
                  {sessions.map(item => (
                    <FbItem key={item.id} item={item} onMark={handleMark} />
                  ))}
                </div>
              )}

              {resolved.length > 0 && (
                <div style={{ marginTop: 24 }}>
                  <SectionHeader label="Resolved This Session" count={resolved.length} />
                  <div
                    className="panel"
                    style={{ display: 'flex', gap: 20, padding: '14px 18px', flexWrap: 'wrap' }}
                  >
                    <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                      <span className="badge b-pass">{confirmed} confirmed</span>
                    </div>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                      <span className="badge b-block">{falsePos} false positive{falsePos !== 1 ? 's' : ''}</span>
                    </div>
                    {resolved.length > 0 && (
                      <div style={{ marginLeft: 'auto', fontSize: 12, fontFamily: 'var(--font-ui)', color: 'var(--text-sec)' }}>
                        FP rate this session: <strong style={{ fontFamily: 'var(--font-mono)' }}>{fpRate}{resolved.length ? '%' : ''}</strong>
                      </div>
                    )}
                  </div>
                </div>
              )}
            </>
          )}
        </div>

        {/* ── Sidebar ───────────────────────────── */}
        <div>
          <QueueStatCard title="Queue Summary">
            <StatRow label="Pending total"  value={sessions.length} color={sessions.length > 0 ? 'var(--amber-hi)' : 'var(--green-hi)'} />
            <StatRow label="Blocked"        value={blocked}   color={blocked > 0 ? 'var(--red-hi)' : 'var(--text-primary)'} />
            <StatRow label="Escalated"      value={escalated} color={escalated > 0 ? 'var(--amber-hi)' : 'var(--text-primary)'} />
          </QueueStatCard>

          <QueueStatCard title="Session Progress">
            <StatRow label="Confirmed correct"  value={confirmed} color={confirmed > 0 ? 'var(--green-hi)' : 'var(--text-primary)'} />
            <StatRow label="False positives"    value={falsePos}  color={falsePos > 0 ? 'var(--red-hi)' : 'var(--text-primary)'} />
            <StatRow label="FP rate"            value={resolved.length ? `${fpRate}%` : '—'} />
          </QueueStatCard>

          <div className="panel">
            <div className="panel-hdr">
              <div className="panel-title">Feedback → Training</div>
            </div>
            <div className="panel-body">
              <div style={{ fontSize: 12, fontFamily: 'var(--font-ui)', color: 'var(--text-sec)', lineHeight: 1.7, marginBottom: 14 }}>
                Marking a decision as "False positive" logs it to the synthetic eval dataset for gateway retraining.
              </div>
              <div style={{
                padding: '10px 14px',
                background: 'var(--bg-surface)',
                border: '1px solid var(--border)',
                borderRadius: 'var(--r-md)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
              }}>
                <div style={{ fontSize: 12, fontFamily: 'var(--font-ui)', color: 'var(--text-sec)' }}>
                  FP cases ready
                </div>
                <div style={{
                  fontFamily: 'var(--font-mono)',
                  fontSize: 18,
                  fontWeight: 700,
                  color: falsePos > 0 ? 'var(--red-hi)' : 'var(--text-muted)',
                }}>
                  {falsePos}
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
