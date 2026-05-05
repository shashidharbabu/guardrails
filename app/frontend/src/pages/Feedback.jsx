import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { getSessions, submitFeedback } from '../api/client'

function piiScore(s) {
  return s.gateway_payload?.scores?.pii ?? s.gateway_payload?.pii_score ?? 0
}
function jbScore(s) {
  return s.gateway_payload?.scores?.jailbreak ?? s.gateway_payload?.jailbreak_score ?? 0
}
function piScore(s) {
  return s.gateway_payload?.scores?.prompt_injection ?? s.gateway_payload?.injection_score ?? 0
}

function timeSince(iso) {
  const diff = Date.now() - new Date(iso).getTime()
  const mins = Math.floor(diff / 60000)
  if (mins < 1) return 'just now'
  if (mins < 60) return `${mins}m ago`
  const hrs = Math.floor(mins / 60)
  return hrs < 24 ? `${hrs}h ago` : `${Math.floor(hrs / 24)}d ago`
}

function FbItem({ item, onMark }) {
  const [marked, setMarked] = useState(null)
  const [note, setNote]     = useState('')
  const [saving, setSaving] = useState(false)

  const pii  = piiScore(item)
  const jb   = jbScore(item)
  const pi   = piScore(item)
  const comp = item.gateway_score

  const itemClass = item.gateway_decision === 'BLOCK' ? 'fb-item fb-block' : 'fb-item fb-esc'
  const decBadge  = item.gateway_decision === 'BLOCK' ? 'b-block' : 'b-esc'

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

  return (
    <div className={itemClass}>
      <div className="fb-top">
        <div className="fb-q">{item.query}</div>
        <div className="fb-bd">
          <span className={`badge ${decBadge}`}>{item.gateway_decision}</span>
          {item.mad_routing && (
            <span className="badge b-gray">MAD: {item.mad_routing}</span>
          )}
          <span className="badge b-gray" style={{ color: 'var(--text-muted)' }}>{timeSince(item.created_at)}</span>
        </div>
      </div>
      <div className="fb-r">
        Composite {comp.toFixed(2)} · gateway score in {item.gateway_decision === 'BLOCK' ? 'block' : 'escalation'} zone
      </div>
      <div style={{ display: 'flex', gap: '4px', marginBottom: '10px' }}>
        <span className="sc">PII {pii.toFixed(2)}</span>
        <span className="sc">JB {jb.toFixed(2)}</span>
        <span className="sc">PI {pi.toFixed(2)}</span>
        <span className="sc">∑ {comp.toFixed(2)}</span>
      </div>
      <div className="fb-acts">
        <button
          className={`fb-btn${marked === 'c' ? ' correct' : ''}`}
          onClick={() => handleMark('c')}
          disabled={!!marked || saving}
        >
          ✓ Correct
        </button>
        <button
          className={`fb-btn${marked === 'f' ? ' fp' : ''}`}
          onClick={() => handleMark('f')}
          disabled={!!marked || saving}
        >
          ✕ False positive
        </button>
        <input
          className="fb-note"
          placeholder="Add analyst note..."
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

export default function Feedback() {
  const [sessions, setSessions] = useState([])
  const [loading, setLoading]   = useState(true)
  const [error, setError]       = useState(null)
  const [resolved, setResolved] = useState([])

  useEffect(() => {
    getSessions()
      .then(all => {
        const queue = all.filter(s =>
          s.gateway_decision === 'BLOCK' || s.gateway_decision === 'ESCALATE'
        )
        setSessions(queue)
      })
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  function handleMark(id, label) {
    setResolved(r => [...r, { id, label }])
    setSessions(prev => prev.filter(s => s.id !== id))
  }

  const confirmed   = resolved.filter(r => r.label === 'c').length
  const falsePos    = resolved.filter(r => r.label === 'f').length
  const fpRate      = resolved.length ? ((falsePos / resolved.length) * 100).toFixed(0) : '—'

  return (
    <div>
      <div className="ph">
        <div className="pt">Feedback</div>
        <div className="ps">Analyst review queue · escalated and blocked sessions for human review</div>
      </div>

      <div className="fb-layout">
        <div>
          {loading && (
            <div style={{ textAlign: 'center', padding: '40px', fontFamily: 'var(--font-mono)', fontSize: '12px', color: 'var(--text-muted)' }}>
              Loading review queue…
            </div>
          )}
          {error && (
            <div style={{ padding: '12px', background: 'var(--red-dim)', border: '1px solid var(--red-mid)', borderRadius: '6px', fontFamily: 'var(--font-mono)', fontSize: '12px', color: 'var(--red)' }}>
              {error}
            </div>
          )}
          {!loading && !error && (
            <>
              <div className="sec-lbl">Pending review · {sessions.length} items</div>
              {sessions.length === 0 ? (
                <div style={{ padding: '20px', textAlign: 'center', fontFamily: 'var(--font-mono)', fontSize: '12px', color: 'var(--text-muted)' }}>
                  No sessions pending review. Submit a query from New Query to populate this queue.
                </div>
              ) : (
                sessions.map(item => (
                  <FbItem key={item.id} item={item} onMark={handleMark} />
                ))
              )}

              {resolved.length > 0 && (
                <div style={{ marginTop: '20px' }}>
                  <div className="sec-lbl">Resolved this session · {resolved.length} items</div>
                  <div className="card-sm" style={{ fontSize: '12px', fontFamily: 'var(--font-mono)', color: 'var(--text-sec)', textAlign: 'center' }}>
                    {confirmed} confirmed correct &nbsp;·&nbsp; {falsePos} false positive
                  </div>
                </div>
              )}
            </>
          )}
        </div>

        <div>
          <div className="card" style={{ marginBottom: '12px' }}>
            <div className="ts-title">Review stats</div>
            <div className="lat-tbl">
              <div className="lt-row"><span className="lt-s">Pending</span><span className="lt-v" style={{ color: 'var(--amber)' }}>{sessions.length}</span></div>
              <div className="lt-row"><span className="lt-s">Confirmed</span><span className="lt-v" style={{ color: 'var(--green)' }}>{confirmed}</span></div>
              <div className="lt-row"><span className="lt-s">False positives</span><span className="lt-v" style={{ color: 'var(--red)' }}>{falsePos}</span></div>
              <div className="lt-row"><span className="lt-s">FP rate</span><span className="lt-v">{fpRate}{resolved.length ? '%' : ''}</span></div>
            </div>
          </div>

          <div className="card">
            <div className="ts-title">Feedback → training</div>
            <div style={{ fontSize: '11px', fontFamily: 'var(--font-mono)', color: 'var(--text-muted)', lineHeight: 1.7, marginBottom: '12px' }}>
              Marking a decision as "False positive" logs it to the synthetic eval dataset for gateway retraining.
            </div>
            <div className="ph-card" style={{ padding: '10px' }}>
              <div style={{ fontSize: '11px', fontFamily: 'var(--font-mono)', color: 'var(--text-sec)' }}>FP cases ready for export</div>
              <div style={{ fontSize: '10px', fontFamily: 'var(--font-mono)', color: 'var(--text-muted)', marginTop: '3px' }}>{falsePos} case{falsePos !== 1 ? 's' : ''} this session</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
