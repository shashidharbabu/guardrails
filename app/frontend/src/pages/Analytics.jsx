import { useEffect, useState } from 'react'
import { getAnalytics } from '../api/client'
import PageHeader from '../components/PageHeader'
import MetricCard from '../components/MetricCard'
import SectionHeader from '../components/SectionHeader'
import LoadingSkeleton from '../components/LoadingSkeleton'
import { ErrorState } from '../components/EmptyState'

const DAYS = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Today']
const DAY_VOL = [38, 55, 72, 48, 65, 22, 90]
const DAY_MAX = Math.max(...DAY_VOL)

const LATENCY_ROWS = [
  { lbl: 'Gateway models',    val: '120ms', pct: 15, color: 'var(--blue)' },
  { lbl: 'Embedding (Qwen3)', val: '210ms', pct: 26, color: 'var(--violet)' },
  { lbl: 'Qdrant retrieval',  val: '180ms', pct: 22, color: 'var(--teal)' },
  { lbl: 'LLM gen (7B)',      val: '410ms', pct: 51, color: 'var(--amber)' },
  { lbl: 'MAD cycle 1',       val: '640ms', pct: 80, color: 'var(--orange)' },
  { lbl: 'MAD cycle 2',       val: '590ms', pct: 74, color: 'var(--orange)' },
  { lbl: 'Judge',             val: '320ms', pct: 40, color: 'var(--pink)' },
]

const BLOCK_REASONS = [
  { lbl: 'PII leak',     pct: 72, color: 'var(--blue)' },
  { lbl: 'Jailbreak',   pct: 18, color: 'var(--pink)' },
  { lbl: 'Prompt inj.', pct: 10, color: 'var(--amber)' },
]

/* ─── KPI Icons ─────────────────────────────────────────────── */
function RequestsIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.4">
      <rect x="2" y="3" width="12" height="10" rx="1.5"/>
      <path d="M5 7h6M5 10h4" strokeLinecap="round"/>
    </svg>
  )
}
function BlockIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.4">
      <circle cx="8" cy="8" r="6"/>
      <path d="M4.3 4.3l7.4 7.4" strokeLinecap="round"/>
    </svg>
  )
}
function EscIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.4">
      <path d="M8 2l1.8 3.5L14 6.3l-3 2.9.7 4.1L8 11.5l-3.7 1.8.7-4.1L2 6.3l4.2-.8z"/>
    </svg>
  )
}
function PassIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.4">
      <circle cx="8" cy="8" r="6"/>
      <path d="M5.5 8l2 2 3-3" strokeLinecap="round" strokeLinejoin="round"/>
    </svg>
  )
}

function decisionColor(decision) {
  if (decision === 'PASS') return 'var(--teal)'
  if (decision === 'BLOCK') return 'var(--red)'
  return 'var(--amber)'
}

function madColor(routing) {
  if (routing === 'DELIVER')      return 'var(--blue)'
  if (routing === 'HARD_BLOCK' || routing === 'BLOCKED') return 'var(--red)'
  if (routing === 'HUMAN_REVIEW') return 'var(--orange)'
  return 'var(--amber)'
}

/* ─── Reusable chart panel ──────────────────────────────────── */
function ChartPanel({ title, children, footer }) {
  return (
    <div className="panel">
      <div className="panel-hdr">
        <div className="panel-title">{title}</div>
      </div>
      <div className="panel-body">
        {children}
      </div>
      {footer && (
        <div style={{
          padding: '8px 20px',
          borderTop: '1px solid var(--border)',
          fontFamily: 'var(--font-ui)',
          fontSize: '10px',
          color: 'var(--text-muted)',
        }}>
          {footer}
        </div>
      )}
    </div>
  )
}

/* ─── Analytics skeleton ────────────────────────────────────── */
function AnalyticsSkeleton() {
  return (
    <div>
      <LoadingSkeleton type="stat-grid" count={4} />
      <div className="three-col" style={{ marginBottom: 20 }}>
        <LoadingSkeleton type="cards" count={1} />
        <LoadingSkeleton type="cards" count={1} />
        <LoadingSkeleton type="cards" count={1} />
      </div>
      <div className="two-col">
        <LoadingSkeleton type="cards" count={1} />
        <LoadingSkeleton type="cards" count={1} />
      </div>
    </div>
  )
}

export default function Analytics() {
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  function load() {
    setLoading(true)
    setError(null)
    getAnalytics()
      .then(setData)
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }

  useEffect(() => { load() }, [])

  return (
    <div>
      <PageHeader
        title="Analytics"
        sub="System-wide metrics · safety signal distribution · pipeline performance"
        actions={
          !loading && !error && (
            <button type="button" className="btn" onClick={load}>
              <svg width="12" height="12" viewBox="0 0 12 12" fill="none" style={{ marginRight: 5 }}>
                <path d="M10 6A4 4 0 112 6" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round"/>
                <path d="M10 3v3H7" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round" strokeLinejoin="round"/>
              </svg>
              Refresh
            </button>
          )
        }
      />

      {loading && <AnalyticsSkeleton />}

      {error && (
        <ErrorState
          title="Failed to load analytics"
          message={error}
          onRetry={load}
        />
      )}

      {!loading && !error && data && (
        <>
          {/* ── KPI Row ─────────────────────────────────── */}
          {(() => {
            const blockRate = data.total_sessions
              ? ((data.blocked_count / data.total_sessions) * 100).toFixed(1)
              : '0.0'
            const escalateRate = data.total_sessions
              ? ((data.escalated_count / data.total_sessions) * 100).toFixed(1)
              : '0.0'
            const madDeliverCount = data.mad_routing.find(r => r.routing === 'DELIVER')?.count || 0
            const madEvaluated = data.mad_routing.reduce((a, r) => a + r.count, 0)
            const madPass = madEvaluated
              ? ((madDeliverCount / madEvaluated) * 100).toFixed(0)
              : '—'

            return (
              <div className="stat-grid">
                <MetricCard
                  label="Total Requests"
                  value={data.total_sessions.toLocaleString()}
                  sub="sessions processed"
                  accent="blue"
                  icon={<RequestsIcon />}
                />
                <MetricCard
                  label="Block Rate"
                  value={`${blockRate}%`}
                  sub={`${data.blocked_count.toLocaleString()} blocked`}
                  accent="red"
                  valueColor="var(--red-hi)"
                  icon={<BlockIcon />}
                />
                <MetricCard
                  label="Escalation Rate"
                  value={`${escalateRate}%`}
                  sub={`${data.escalated_count.toLocaleString()} escalated`}
                  accent="amber"
                  valueColor="var(--amber-hi)"
                  icon={<EscIcon />}
                />
                <MetricCard
                  label="MAD Pass Rate"
                  value={`${madPass}%`}
                  sub="of MAD-eligible queries"
                  accent="teal"
                  valueColor="var(--teal-hi)"
                  icon={<PassIcon />}
                />
              </div>
            )
          })()}

          {/* ── Distribution Charts ──────────────────────── */}
          <SectionHeader label="Decision Distribution" />
          {(() => {
            const decisionMax = Math.max(...data.decisions.map(x => x.count), 1)
            const madMax = Math.max(...data.mad_routing.map(x => x.count), 1)

            return (
              <div className="three-col" style={{ marginBottom: 24 }}>
                <ChartPanel title="Gateway Decisions">
                  <div className="bc-chart">
                    {data.decisions.map(d => (
                      <div key={d.decision} className="bc-row">
                        <div className="bc-lbl" style={{ width: 80 }}>{d.decision}</div>
                        <div className="bc-track">
                          <div
                            className="bc-fill"
                            style={{
                              width: `${(d.count / decisionMax) * 100}%`,
                              background: decisionColor(d.decision),
                            }}
                          />
                        </div>
                        <div className="bc-val">{d.count}</div>
                      </div>
                    ))}
                  </div>
                </ChartPanel>

                <ChartPanel title="MAD Routing Outcomes">
                  <div className="bc-chart">
                    {data.mad_routing.map(d => (
                      <div key={d.routing} className="bc-row">
                        <div className="bc-lbl" style={{ width: 110, fontSize: 10 }}>{d.routing}</div>
                        <div className="bc-track">
                          <div
                            className="bc-fill"
                            style={{
                              width: `${(d.count / madMax) * 100}%`,
                              background: madColor(d.routing),
                            }}
                          />
                        </div>
                        <div className="bc-val">{d.count}</div>
                      </div>
                    ))}
                  </div>
                </ChartPanel>

                <ChartPanel title="Block Reasons" footer="* indicative breakdown based on primary trigger">
                  <div className="bc-chart">
                    {BLOCK_REASONS.map(r => (
                      <div key={r.lbl} className="bc-row">
                        <div className="bc-lbl" style={{ width: 90 }}>{r.lbl}</div>
                        <div className="bc-track">
                          <div
                            className="bc-fill"
                            style={{ width: `${r.pct}%`, background: r.color }}
                          />
                        </div>
                        <div className="bc-val">{r.pct}%</div>
                      </div>
                    ))}
                  </div>
                </ChartPanel>
              </div>
            )
          })()}

          {/* ── Volume + Latency ─────────────────────────── */}
          <SectionHeader label="Volume & Performance" />
          <div className="two-col" style={{ marginBottom: 24 }}>
            <ChartPanel title="Request Volume — Last 7 Days">
              <div className="vol-bars">
                {DAYS.map((day, i) => (
                  <div key={day} className="vol-col">
                    <div
                      className="vol-bar"
                      style={{
                        height: `${(DAY_VOL[i] / DAY_MAX) * 80}px`,
                        background: i === DAYS.length - 1
                          ? 'var(--blue)'
                          : 'var(--blue)',
                        opacity: i === DAYS.length - 1
                          ? 1
                          : 0.22 + (i / DAYS.length) * 0.55,
                        boxShadow: i === DAYS.length - 1
                          ? '0 0 8px rgba(59,130,246,0.3)'
                          : 'none',
                      }}
                    />
                    <div className="vol-lbl">{day}</div>
                  </div>
                ))}
              </div>
              <div style={{
                marginTop: 12,
                display: 'flex',
                justifyContent: 'space-between',
                fontFamily: 'var(--font-ui)',
                fontSize: '11px',
                color: 'var(--text-muted)',
              }}>
                <span>Peak: {DAY_MAX} req/day</span>
                <span>Today: <strong style={{ color: 'var(--blue-hi)' }}>{DAY_VOL[DAY_VOL.length - 1]}</strong> requests</span>
              </div>
            </ChartPanel>

            <ChartPanel title="Pipeline Latency — Avg per Stage">
              {LATENCY_ROWS.map(r => (
                <div key={r.lbl} className="lh-row">
                  <div className="lh-lbl">{r.lbl}</div>
                  <div className="lh-track">
                    <div className="lh-fill" style={{ width: `${r.pct}%`, background: r.color }} />
                  </div>
                  <div className="lh-val">{r.val}</div>
                </div>
              ))}
              <div className="lh-row tot" style={{ marginTop: 8, paddingTop: 8, borderTop: '1px solid var(--border)' }}>
                <div className="lh-lbl">Total E2E</div>
                <div className="lh-track">
                  <div className="lh-fill" style={{ width: '100%', background: 'var(--teal)' }} />
                </div>
                <div className="lh-val">{(data.avg_pipeline_ms / 1000).toFixed(1)}s</div>
              </div>
            </ChartPanel>
          </div>

          {/* ── Summary footer ───────────────────────────── */}
          {(() => {
            const madDeliverCount = data.mad_routing.find(r => r.routing === 'DELIVER')?.count || 0
            const madEvaluated = data.mad_routing.reduce((a, r) => a + r.count, 0)
            return (
              <div
                className="panel"
                style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '14px 20px' }}
              >
                <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                  <div style={{
                    width: 8,
                    height: 8,
                    borderRadius: '50%',
                    background: 'var(--teal)',
                    boxShadow: '0 0 6px rgba(20,184,166,0.4)',
                    flexShrink: 0,
                  }} />
                  <span style={{ fontSize: '12px', fontFamily: 'var(--font-ui)', color: 'var(--text-sec)' }}>
                    {data.total_sessions.toLocaleString()} sessions ·{' '}
                    {madEvaluated} MAD-evaluated ·{' '}
                    {madDeliverCount} delivered
                  </span>
                </div>
                <span style={{ fontSize: '11px', fontFamily: 'var(--font-mono)', color: 'var(--text-muted)' }}>
                  SQLite · app_sessions.db
                </span>
              </div>
            )
          })()}
        </>
      )}
    </div>
  )
}
