import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { getReviewQueue, submitReviewAction } from '../api/client'
import StatusBadge from '../components/StatusBadge'
import LoadingSkeleton from '../components/LoadingSkeleton'

const ACTIONS = [
  { value: 'approve_delivery', label: 'Approve Delivery', color: 'var(--teal)' },
  { value: 'reject_answer', label: 'Reject Answer', color: 'var(--red)' },
  { value: 'request_regeneration', label: 'Request Regeneration', color: 'var(--amber)' },
  { value: 'escalate_to_admin', label: 'Escalate to Admin', color: 'var(--violet)' },
  { value: 'mark_false_positive', label: 'Mark False Positive', color: 'var(--blue)' },
  { value: 'add_review_note', label: 'Add Note Only', color: 'var(--text-muted)' },
]

function ReviewCard({ session, onAction }) {
  const [expanded, setExpanded] = useState(false)
  const [selectedAction, setSelectedAction] = useState('')
  const [notes, setNotes] = useState('')
  const [submitting, setSubmitting] = useState(false)
  const [submitted, setSubmitted] = useState(false)

  const gw = session.gateway_payload || {}
  const mad = session.mad_output || {}

  async function handleSubmit() {
    if (!selectedAction) return
    setSubmitting(true)
    try {
      await onAction(session.id, selectedAction, notes)
      setSubmitted(true)
    } finally {
      setSubmitting(false)
    }
  }

  if (submitted) {
    return (
      <div
        style={{
          background: 'var(--bg-surface)',
          border: '1px solid var(--border)',
          borderRadius: 'var(--r)',
          padding: '14px 16px',
          display: 'flex',
          gap: 10,
          alignItems: 'center',
          opacity: 0.6,
        }}
      >
        <span style={{ fontSize: 11, color: 'var(--teal)' }}>✓ Review action submitted</span>
        <span style={{ fontSize: 10, color: 'var(--text-muted)' }}>{session.id}</span>
      </div>
    )
  }

  return (
    <div
      style={{
        background: 'var(--bg-surface)',
        border: '1px solid var(--border)',
        borderRadius: 'var(--r)',
        padding: '14px 16px',
        marginBottom: 10,
      }}
    >
      {/* Row header */}
      <div style={{ display: 'flex', alignItems: 'flex-start', gap: 10, marginBottom: 10 }}>
        <div style={{ flex: 1, minWidth: 0 }}>
          <div
            style={{
              fontFamily: 'var(--font-mono)',
              fontSize: 12,
              color: 'var(--text-primary)',
              marginBottom: 4,
              wordBreak: 'break-word',
            }}
          >
            {session.query}
          </div>
          <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
            <StatusBadge type="decision" value={session.gateway_decision} />
            {session.mad_routing && <StatusBadge type="mad" value={session.mad_routing} />}
            <span className="badge b-gray" style={{ fontSize: 10 }}>
              {new Date(session.created_at).toLocaleString()}
            </span>
            <span className="tid">{session.id}</span>
          </div>
        </div>
        <button
          type="button"
          onClick={() => setExpanded(e => !e)}
          style={{
            background: 'none',
            border: '1px solid var(--border-md)',
            borderRadius: 'var(--r-sm)',
            padding: '3px 10px',
            fontSize: 10,
            color: 'var(--text-sec)',
            cursor: 'pointer',
            whiteSpace: 'nowrap',
            fontFamily: 'var(--font-ui)',
          }}
        >
          {expanded ? 'Collapse' : 'Expand'}
        </button>
      </div>

      {/* Expanded details */}
      {expanded && (
        <div style={{ marginBottom: 12 }}>
          {session.llm_answer && (
            <div
              style={{
                background: 'var(--bg-base)',
                border: '1px solid var(--border)',
                borderRadius: 'var(--r-sm)',
                padding: '10px 12px',
                fontFamily: 'var(--font-mono)',
                fontSize: 11,
                color: 'var(--text-sec)',
                marginBottom: 8,
                lineHeight: 1.6,
              }}
            >
              <div style={{ fontSize: 10, color: 'var(--text-muted)', marginBottom: 4 }}>LLM ANSWER</div>
              {session.llm_answer}
            </div>
          )}
          {mad.routing_decision && (
            <div style={{ fontSize: 11, color: 'var(--text-sec)', marginBottom: 4 }}>
              MAD: {mad.routing_decision} · confidence {((mad.aggregate_confidence || mad.confidence_score || 0) * 100).toFixed(0)}%
            </div>
          )}
          <Link to={`/sessions/${session.id}`} style={{ fontSize: 11, color: 'var(--blue)' }}>
            View full trace →
          </Link>
        </div>
      )}

      {/* Action panel */}
      <div style={{ borderTop: '1px solid var(--border)', paddingTop: 10 }}>
        <div style={{ fontSize: 10, color: 'var(--text-muted)', marginBottom: 6, fontFamily: 'var(--font-ui)', letterSpacing: '0.05em' }}>
          REVIEW ACTION
        </div>
        <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap', marginBottom: 8 }}>
          {ACTIONS.map(a => (
            <button
              key={a.value}
              type="button"
              onClick={() => setSelectedAction(a.value)}
              style={{
                padding: '4px 10px',
                fontSize: 10,
                borderRadius: 100,
                border: `1px solid ${selectedAction === a.value ? a.color : 'var(--border-md)'}`,
                background: selectedAction === a.value ? `${a.color}15` : 'transparent',
                color: selectedAction === a.value ? a.color : 'var(--text-sec)',
                cursor: 'pointer',
                fontFamily: 'var(--font-ui)',
                fontWeight: 500,
                transition: 'all 0.1s',
              }}
            >
              {a.label}
            </button>
          ))}
        </div>
        <textarea
          value={notes}
          onChange={e => setNotes(e.target.value)}
          placeholder="Review notes (optional)…"
          rows={2}
          style={{
            width: '100%',
            padding: '6px 8px',
            fontFamily: 'var(--font-mono)',
            fontSize: 11,
            background: 'var(--bg-base)',
            border: '1px solid var(--border-md)',
            borderRadius: 'var(--r-sm)',
            color: 'var(--text-primary)',
            resize: 'none',
            outline: 'none',
            marginBottom: 8,
            lineHeight: 1.5,
          }}
        />
        <button
          type="button"
          className="act-btn lf-btn"
          disabled={!selectedAction || submitting}
          onClick={handleSubmit}
          style={{ marginBottom: 0 }}
        >
          {submitting ? 'Submitting…' : 'Submit Review Action'}
        </button>
      </div>
    </div>
  )
}

export default function HumanReview() {
  const [sessions, setSessions] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    getReviewQueue()
      .then(setSessions)
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  async function handleAction(sessionId, decision, notes) {
    await submitReviewAction(sessionId, decision, notes)
    // Refresh after action
    const updated = await getReviewQueue()
    setSessions(updated)
  }

  return (
    <div>
      {/* Header */}
      <div style={{ marginBottom: 20 }}>
        <div style={{ fontFamily: 'var(--font-ui)', fontWeight: 700, fontSize: 16, color: 'var(--text-primary)', letterSpacing: '0.02em' }}>
          Human Review Queue
        </div>
        <div style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 2 }}>
          Sessions routed to HUMAN_REVIEW_REQUIRED · {loading ? '…' : sessions.length} item{sessions.length !== 1 ? 's' : ''}
        </div>
      </div>

      {loading && <LoadingSkeleton type="cards" count={3} />}

      {error && <div className="inline-err">{error}</div>}

      {!loading && !error && sessions.length === 0 && (
        <div
          style={{
            background: 'var(--bg-surface)',
            border: '1px solid var(--border)',
            borderRadius: 'var(--r)',
            padding: '32px 20px',
            textAlign: 'center',
            color: 'var(--text-muted)',
            fontSize: 12,
          }}
        >
          No sessions awaiting review.
        </div>
      )}

      {sessions.map(s => (
        <ReviewCard key={s.id} session={s} onAction={handleAction} />
      ))}
    </div>
  )
}
