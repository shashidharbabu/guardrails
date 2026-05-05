import { useEffect, useState } from 'react'
import { getAnalytics } from '../api/client'

const DAYS = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Today']
const DAY_VOL = [38, 55, 72, 48, 65, 22, 90]
const DAY_MAX = Math.max(...DAY_VOL)

export default function Analytics() {
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    getAnalytics()
      .then(setData)
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  if (loading) return (
    <div style={{ textAlign: 'center', padding: '60px', fontFamily: 'var(--font-mono)', fontSize: '12px', color: 'var(--text-muted)' }}>
      Loading analytics…
    </div>
  )
  if (error) return (
    <div style={{ padding: '12px', background: 'var(--red-dim)', border: '1px solid var(--red-mid)', borderRadius: '6px', fontFamily: 'var(--font-mono)', fontSize: '12px', color: 'var(--red)' }}>
      {error}
    </div>
  )

  const blockRate = data.total_sessions ? ((data.blocked_count / data.total_sessions) * 100).toFixed(1) : '0.0'
  const escalateRate = data.total_sessions ? ((data.escalated_count / data.total_sessions) * 100).toFixed(1) : '0.0'
  const madDeliverCount = data.mad_routing.find(r => r.routing === 'DELIVER')?.count || 0
  const madEvaluated = data.mad_routing.reduce((a, r) => a + r.count, 0)
  const madPass = madEvaluated ? ((madDeliverCount / madEvaluated) * 100).toFixed(0) : '—'

  return (
    <div>
      <div className="ph">
        <div className="pt">Analytics</div>
        <div className="ps">System-wide metrics · all sessions</div>
      </div>

      <div className="stat-grid">
        <div className="stat-card">
          <div className="sl">Total requests</div>
          <div className="sv">{data.total_sessions.toLocaleString()}</div>
          <div className="ss">sessions processed</div>
        </div>
        <div className="stat-card">
          <div className="sl">Block rate</div>
          <div className="sv" style={{ color: 'var(--red)' }}>{blockRate}%</div>
          <div className="ss">{data.blocked_count} blocked</div>
        </div>
        <div className="stat-card">
          <div className="sl">Escalation rate</div>
          <div className="sv" style={{ color: 'var(--amber)' }}>{escalateRate}%</div>
          <div className="ss">{data.escalated_count} escalated</div>
        </div>
        <div className="stat-card">
          <div className="sl">MAD pass rate</div>
          <div className="sv" style={{ color: 'var(--teal)' }}>{madPass}%</div>
          <div className="ss">of MAD-eligible queries</div>
        </div>
      </div>

      <div className="three-col" style={{ marginBottom: '14px' }}>
        <div className="card">
          <div style={{ fontSize: '10px', fontFamily: 'var(--font-mono)', color: 'var(--text-muted)', letterSpacing: '0.1em', textTransform: 'uppercase', marginBottom: '12px' }}>
            Gateway decisions
          </div>
          {data.decisions.map(d => {
            const color = d.decision === 'PASS' ? 'var(--teal)' : d.decision === 'BLOCK' ? 'var(--red)' : 'var(--amber)'
            const max = Math.max(...data.decisions.map(x => x.count)) || 1
            return (
              <div key={d.decision} className="mb-row">
                <div className="mb-lbl">{d.decision}</div>
                <div className="mb-track"><div className="mb-fill" style={{ width: `${(d.count / max) * 100}%`, background: color }}/></div>
                <div className="mb-val">{d.count}</div>
              </div>
            )
          })}
        </div>

        <div className="card">
          <div style={{ fontSize: '10px', fontFamily: 'var(--font-mono)', color: 'var(--text-muted)', letterSpacing: '0.1em', textTransform: 'uppercase', marginBottom: '12px' }}>
            MAD routing outcomes
          </div>
          {data.mad_routing.map(d => {
            const color = d.routing === 'DELIVER' ? 'var(--green)' : d.routing === 'HARD_BLOCK' || d.routing === 'BLOCKED' ? 'var(--red)' : d.routing === 'HUMAN_REVIEW' ? '#f97316' : 'var(--amber)'
            const max = Math.max(...data.mad_routing.map(x => x.count)) || 1
            return (
              <div key={d.routing} className="mb-row">
                <div className="mb-lbl">{d.routing}</div>
                <div className="mb-track"><div className="mb-fill" style={{ width: `${(d.count / max) * 100}%`, background: color }}/></div>
                <div className="mb-val">{d.count}</div>
              </div>
            )
          })}
        </div>

        <div className="card">
          <div style={{ fontSize: '10px', fontFamily: 'var(--font-mono)', color: 'var(--text-muted)', letterSpacing: '0.1em', textTransform: 'uppercase', marginBottom: '12px' }}>
            Block reasons
          </div>
          {[
            { lbl: 'PII leak', pct: 72, color: 'var(--blue)' },
            { lbl: 'Jailbreak', pct: 18, color: 'var(--pink)' },
            { lbl: 'Prompt inj.', pct: 10, color: 'var(--amber)' },
          ].map(r => (
            <div key={r.lbl} className="mb-row">
              <div className="mb-lbl">{r.lbl}</div>
              <div className="mb-track"><div className="mb-fill" style={{ width: `${r.pct}%`, background: r.color }}/></div>
              <div className="mb-val">{r.pct}%</div>
            </div>
          ))}
        </div>
      </div>

      <div className="two-col">
        <div className="card">
          <div style={{ fontSize: '10px', fontFamily: 'var(--font-mono)', color: 'var(--text-muted)', letterSpacing: '0.1em', textTransform: 'uppercase', marginBottom: '12px' }}>
            Request volume — last 7 days
          </div>
          <div className="vol-bars">
            {DAYS.map((day, i) => (
              <div key={day} className="vol-col">
                <div className="vol-bar" style={{ height: `${(DAY_VOL[i] / DAY_MAX) * 80}px`, background: 'var(--teal)', opacity: i === DAYS.length - 1 ? 1 : 0.3 + (i / DAYS.length) * 0.5 }}/>
                <div className="vol-lbl">{day}</div>
              </div>
            ))}
          </div>
        </div>

        <div className="card">
          <div style={{ fontSize: '10px', fontFamily: 'var(--font-mono)', color: 'var(--text-muted)', letterSpacing: '0.1em', textTransform: 'uppercase', marginBottom: '12px' }}>
            Pipeline latency — avg per stage
          </div>
          {[
            { lbl: 'Gateway models', val: '120ms', pct: 15 },
            { lbl: 'Embedding (Qwen3)', val: '210ms', pct: 26 },
            { lbl: 'Qdrant retrieval', val: '180ms', pct: 22 },
            { lbl: 'LLM gen (7B)', val: '410ms', pct: 51 },
            { lbl: 'MAD cycle 1', val: '640ms', pct: 80 },
            { lbl: 'MAD cycle 2', val: '590ms', pct: 74 },
            { lbl: 'Judge', val: '320ms', pct: 40 },
          ].map(r => (
            <div key={r.lbl} className="lh-row">
              <div className="lh-lbl">{r.lbl}</div>
              <div className="lh-track"><div className="lh-fill" style={{ width: `${r.pct}%` }}/></div>
              <div className="lh-val">{r.val}</div>
            </div>
          ))}
          <div className="lh-row tot">
            <div className="lh-lbl">Total</div>
            <div className="lh-track"><div className="lh-fill" style={{ width: '100%' }}/></div>
            <div className="lh-val">{(data.avg_pipeline_ms / 1000).toFixed(1)}s</div>
          </div>
        </div>
      </div>

      <div className="card" style={{ marginTop: '14px', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <div className="st-dot" style={{ background: 'var(--teal)', width: 8, height: 8, borderRadius: '50%' }}/>
          <span style={{ fontSize: '12px', fontFamily: 'var(--font-mono)', color: 'var(--text-sec)' }}>
            {data.total_sessions} sessions · {madEvaluated} MAD-evaluated · {madDeliverCount} delivered
          </span>
        </div>
        <span style={{ fontSize: '11px', fontFamily: 'var(--font-mono)', color: 'var(--text-muted)' }}>
          SQLite · app_sessions.db
        </span>
      </div>
    </div>
  )
}
