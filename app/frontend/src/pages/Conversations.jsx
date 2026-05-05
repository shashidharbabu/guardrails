import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { getSessions } from '../api/client'

function decisionBadge(d) {
  if (d === 'BLOCK') return 'b-block'
  if (d === 'ESCALATE') return 'b-esc'
  return 'b-pass'
}

function rowClass(d) {
  if (d === 'BLOCK') return 'conv-row is-block'
  if (d === 'ESCALATE') return 'conv-row is-esc'
  return 'conv-row is-pass'
}

function madBadge(r) {
  if (!r) return null
  if (r === 'DELIVER') return 'b-deliver'
  if (r === 'RETRY') return 'b-retry'
  if (r === 'HARD_BLOCK') return 'b-hard-block'
  if (r === 'HUMAN_REVIEW') return 'b-human-review'
  return 'b-gray'
}

function timeSince(iso) {
  const diff = Date.now() - new Date(iso).getTime()
  const mins = Math.floor(diff / 60000)
  if (mins < 1) return 'just now'
  if (mins < 60) return `${mins}m ago`
  const hrs = Math.floor(mins / 60)
  if (hrs < 24) return `${hrs}h ago`
  return `${Math.floor(hrs / 24)}d ago`
}

const FILTERS = ['ALL', 'BLOCK', 'ESCALATE', 'PASS']

export default function Conversations() {
  const [sessions, setSessions] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [filter, setFilter] = useState('ALL')
  const [search, setSearch] = useState('')

  useEffect(() => {
    getSessions()
      .then(setSessions)
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  const filtered = sessions.filter(s => {
    if (filter !== 'ALL' && s.gateway_decision !== filter) return false
    if (search && !s.query.toLowerCase().includes(search.toLowerCase())) return false
    return true
  })

  const total = sessions.length
  const blocked = sessions.filter(s => s.gateway_decision === 'BLOCK').length
  const escalated = sessions.filter(s => s.gateway_decision === 'ESCALATE').length
  const avgMs = total
    ? Math.round(sessions.reduce((a, s) => a + (s.pipeline_duration_ms || 0), 0) / total)
    : 0

  return (
    <div>
      <div className="ph">
        <div className="pt">Conversations</div>
        <div className="ps">All sessions · click any row to open the full trace</div>
      </div>

      <div className="stat-grid">
        <div className="stat-card">
          <div className="sl">Total today</div>
          <div className="sv">{total}</div>
          <div className="ss">sessions processed</div>
        </div>
        <div className="stat-card">
          <div className="sl">Blocked</div>
          <div className="sv" style={{ color: 'var(--red)' }}>{blocked}</div>
          <div className="ss">{total ? ((blocked / total) * 100).toFixed(1) : 0}% block rate</div>
        </div>
        <div className="stat-card">
          <div className="sl">Escalated</div>
          <div className="sv" style={{ color: 'var(--amber)' }}>{escalated}</div>
          <div className="ss">analyst review queue</div>
        </div>
        <div className="stat-card">
          <div className="sl">Avg latency</div>
          <div className="sv" style={{ color: 'var(--teal)' }}>{(avgMs / 1000).toFixed(1)}s</div>
          <div className="ss">gateway + MAD</div>
        </div>
      </div>

      <div className="filter-row">
        <input
          className="s-inp"
          placeholder="Search queries..."
          value={search}
          onChange={e => setSearch(e.target.value)}
        />
        {FILTERS.map(f => (
          <button key={f} className={`chip${filter === f ? ' active' : ''}`} onClick={() => setFilter(f)}>
            {f}
          </button>
        ))}
      </div>

      {loading && (
        <div style={{ textAlign: 'center', padding: '40px', fontFamily: 'var(--font-mono)', fontSize: '12px', color: 'var(--text-muted)' }}>
          Loading sessions…
        </div>
      )}

      {error && (
        <div style={{ padding: '12px', background: 'var(--red-dim)', border: '1px solid var(--red-mid)', borderRadius: '6px', fontFamily: 'var(--font-mono)', fontSize: '12px', color: 'var(--red)' }}>
          {error}
        </div>
      )}

      {!loading && !error && (
        <div className="conv-list">
          {filtered.length === 0 ? (
            <div style={{ textAlign: 'center', padding: '40px', fontFamily: 'var(--font-mono)', fontSize: '12px', color: 'var(--text-muted)' }}>
              No sessions match the current filter.
            </div>
          ) : filtered.map(s => (
            <Link key={s.id} to={`/sessions/${s.id}`} className={rowClass(s.gateway_decision) + ' fi'}>
              <div>
                <div className="cq">{s.query}</div>
                <div className="cm">
                  <span className={`badge ${decisionBadge(s.gateway_decision)}`}>{s.gateway_decision}</span>
                  {s.mad_routing && (
                    <span className={`badge ${madBadge(s.mad_routing)}`}>MAD: {s.mad_routing}</span>
                  )}
                  <span className="ct">
                    {timeSince(s.created_at)} · {s.pipeline_duration_ms != null ? `${(s.pipeline_duration_ms / 1000).toFixed(1)}s` : '—'}
                  </span>
                </div>
              </div>
              <div className="cs-wrap">
                <div className="sc-row">
                  <span className="sc">PII {(s.gateway_payload?.scores?.pii ?? s.gateway_payload?.pii_score ?? 0).toFixed(2)}</span>
                  <span className="sc">JB {(s.gateway_payload?.scores?.jailbreak ?? s.gateway_payload?.jailbreak_score ?? 0).toFixed(2)}</span>
                  <span className="sc">∑ {s.gateway_score.toFixed(2)}</span>
                </div>
                <span className="tlink">View trace →</span>
              </div>
            </Link>
          ))}
        </div>
      )}
    </div>
  )
}
