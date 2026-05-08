import { useEffect, useState } from 'react'
import { Link, useSearchParams } from 'react-router-dom'
import { motion, useReducedMotion } from 'framer-motion'
import { getSessions } from '../api/client'
import MetricCard from '../components/MetricCard'
import FilterBar from '../components/FilterBar'
import PageHeader from '../components/PageHeader'
import SectionHeader from '../components/SectionHeader'
import EmptyState, { NoSessionsIcon, NoResultsIcon, ErrorState } from '../components/EmptyState'
import LoadingSkeleton from '../components/LoadingSkeleton'
import StatusBadge, { decisionCls, madCls } from '../components/StatusBadge'

function timeSince(iso) {
  const diff = Date.now() - new Date(iso).getTime()
  const mins = Math.floor(diff / 60000)
  if (mins < 1) return 'just now'
  if (mins < 60) return `${mins}m ago`
  const hrs = Math.floor(mins / 60)
  if (hrs < 24) return `${hrs}h ago`
  return `${Math.floor(hrs / 24)}d ago`
}

function rowClass(d) {
  if (d === 'BLOCK') return 'conv-row is-block'
  if (d === 'ESCALATE') return 'conv-row is-esc'
  return 'conv-row is-pass'
}

const FILTERS = ['ALL', 'BLOCK', 'ESCALATE', 'PASS']

/* ─── KPI Icons ─────────────────────────────────────────────── */
function TotalIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.4">
      <rect x="2" y="3" width="12" height="10" rx="1.5"/>
      <path d="M5 7h6M5 10h4" strokeLinecap="round"/>
    </svg>
  )
}
function BlockedIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.4">
      <circle cx="8" cy="8" r="6"/>
      <path d="M4.3 4.3l7.4 7.4" strokeLinecap="round"/>
    </svg>
  )
}
function EscalatedIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.4">
      <path d="M8 2l1.8 3.5L14 6.3l-3 2.9.7 4.1L8 11.5l-3.7 1.8.7-4.1L2 6.3l4.2-.8z"/>
    </svg>
  )
}
function LatencyIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.4">
      <circle cx="8" cy="8" r="6"/>
      <path d="M8 5v3.5l2.5 1.5" strokeLinecap="round" strokeLinejoin="round"/>
    </svg>
  )
}

export default function Conversations() {
  const [searchParams, setSearchParams] = useSearchParams()
  const [sessions, setSessions] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [filter, setFilter] = useState('ALL')
  const [search, setSearch] = useState(searchParams.get('q') || '')
  const shouldReduceMotion = useReducedMotion()

  function load() {
    setLoading(true)
    setError(null)
    getSessions()
      .then(setSessions)
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }

  useEffect(() => { load() }, [])

  useEffect(() => {
    setSearch(searchParams.get('q') || '')
  }, [searchParams])

  function handleSearch(value) {
    setSearch(value)
    const next = new URLSearchParams(searchParams)
    if (value.trim()) {
      next.set('q', value)
    } else {
      next.delete('q')
    }
    setSearchParams(next, { replace: true })
  }

  const filtered = sessions.filter(s => {
    if (filter !== 'ALL' && s.gateway_decision !== filter) return false
    if (search && !s.query.toLowerCase().includes(search.toLowerCase())) return false
    return true
  })

  const total     = sessions.length
  const blocked   = sessions.filter(s => s.gateway_decision === 'BLOCK').length
  const escalated = sessions.filter(s => s.gateway_decision === 'ESCALATE').length
  const avgMs     = total
    ? Math.round(sessions.reduce((a, s) => a + (s.pipeline_duration_ms || 0), 0) / total)
    : 0
  const blockRate = total ? ((blocked / total) * 100).toFixed(1) : '0.0'

  const listVariants = {
    hidden: {},
    show: { transition: { staggerChildren: 0.035 } },
  }
  const itemVariants = shouldReduceMotion
    ? {}
    : {
        hidden: { opacity: 0, y: 6 },
        show:   { opacity: 1, y: 0, transition: { duration: 0.16, ease: [0.16, 1, 0.3, 1] } },
      }

  return (
    <div>
      <PageHeader
        title="Conversations"
        sub="All sessions processed through the guardrails pipeline — click any row to inspect the full trace"
        actions={
          <button type="button" className="btn" onClick={load} disabled={loading}>
            <svg width="12" height="12" viewBox="0 0 12 12" fill="none" style={{ marginRight: 5 }}>
              <path d="M10 6A4 4 0 112 6" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round"/>
              <path d="M10 3v3H7" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
            Refresh
          </button>
        }
      />

      {/* ── KPI Row ─────────────────────────────────── */}
      {loading ? (
        <LoadingSkeleton type="stat-grid" count={4} />
      ) : (
        <div className="stat-grid">
          <MetricCard
            label="Total Today"
            value={total}
            sub="sessions processed"
            accent="blue"
            icon={<TotalIcon />}
          />
          <MetricCard
            label="Blocked"
            value={blocked}
            sub={`${blockRate}% block rate`}
            accent="red"
            valueColor="var(--red-hi)"
            icon={<BlockedIcon />}
          />
          <MetricCard
            label="Escalated"
            value={escalated}
            sub="pending analyst review"
            accent="amber"
            valueColor="var(--amber-hi)"
            icon={<EscalatedIcon />}
          />
          <MetricCard
            label="Avg Latency"
            value={`${(avgMs / 1000).toFixed(1)}s`}
            sub="gateway + MAD pipeline"
            accent="teal"
            valueColor="var(--teal-hi)"
            icon={<LatencyIcon />}
          />
        </div>
      )}

      {/* ── Filter Bar ──────────────────────────────── */}
      <FilterBar
        search={search}
        onSearch={handleSearch}
        placeholder="Search sessions…"
        filters={FILTERS}
        activeFilter={filter}
        onFilter={setFilter}
        count={!loading && !error ? filtered.length : undefined}
      />

      {/* ── Content ─────────────────────────────────── */}
      {loading && <LoadingSkeleton type="rows" count={5} />}

      {error && (
        <ErrorState
          title="Failed to load sessions"
          message={error}
          onRetry={load}
        />
      )}

      {!loading && !error && (
        <>
          <SectionHeader
            label="Sessions"
            count={filtered.length}
          />

          {filtered.length === 0 ? (
            <div className="panel" style={{ marginTop: 4 }}>
              <EmptyState
                icon={search || filter !== 'ALL' ? <NoResultsIcon /> : <NoSessionsIcon />}
                title={
                  search || filter !== 'ALL'
                    ? 'No matching sessions'
                    : 'No sessions yet'
                }
                description={
                  search || filter !== 'ALL'
                    ? 'Try adjusting your search terms or filter criteria.'
                    : 'Run a query via the Test Query page to see sessions appear here.'
                }
                action={
                  (search || filter !== 'ALL') && (
                    <button
                      type="button"
                      className="btn"
                      onClick={() => { handleSearch(''); setFilter('ALL') }}
                    >
                      Clear filters
                    </button>
                  )
                }
              />
            </div>
          ) : (
            <motion.div
              className="conv-list"
              variants={listVariants}
              initial="hidden"
              animate="show"
            >
              {filtered.map(s => (
                <motion.div key={s.id} variants={itemVariants}>
                  <Link to={`/sessions/${s.id}`} className={rowClass(s.gateway_decision)}>
                    <div style={{ minWidth: 0 }}>
                      <div className="cq">{s.query}</div>
                      <div className="cm">
                        <StatusBadge type="decision" value={s.gateway_decision} />
                        {s.mad_routing && (
                          <StatusBadge type="mad" value={s.mad_routing} />
                        )}
                        <span className="ct">
                          {timeSince(s.created_at)}
                          {s.pipeline_duration_ms != null && (
                            <> · {(s.pipeline_duration_ms / 1000).toFixed(1)}s</>
                          )}
                        </span>
                      </div>
                    </div>
                    <div className="cs-wrap">
                      <div className="sc-row">
                        <span className="sc">
                          PII {(s.gateway_payload?.scores?.pii ?? s.gateway_payload?.pii_score ?? 0).toFixed(2)}
                        </span>
                        <span className="sc">
                          JB {(s.gateway_payload?.scores?.jailbreak ?? s.gateway_payload?.jailbreak_score ?? 0).toFixed(2)}
                        </span>
                        <span className="sc">∑ {s.gateway_score.toFixed(2)}</span>
                      </div>
                      <span className="tlink">View trace →</span>
                    </div>
                  </Link>
                </motion.div>
              ))}
            </motion.div>
          )}
        </>
      )}
    </div>
  )
}
