import { useEffect, useState } from 'react'
import { getSystemHealth } from '../api/client'

function StatusDot({ status }) {
  const color =
    status === 'healthy' ? 'var(--teal)' :
    status === 'degraded' ? 'var(--amber)' : 'var(--red)'
  return (
    <span
      style={{
        display: 'inline-block',
        width: 8,
        height: 8,
        borderRadius: '50%',
        background: color,
        marginRight: 6,
        flexShrink: 0,
      }}
    />
  )
}

function ComponentCard({ name, component }) {
  const isHealthy = component.status === 'healthy'
  return (
    <div
      style={{
        background: 'var(--bg-surface)',
        border: `1px solid ${isHealthy ? 'var(--border)' : 'var(--red-mid)'}`,
        borderRadius: 'var(--r)',
        padding: '14px 16px',
        display: 'flex',
        alignItems: 'flex-start',
        gap: 12,
      }}
    >
      <StatusDot status={component.status} />
      <div style={{ flex: 1, minWidth: 0 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 4 }}>
          <span style={{ fontFamily: 'var(--font-ui)', fontWeight: 600, fontSize: 12, textTransform: 'uppercase', letterSpacing: '0.05em', color: 'var(--text-primary)' }}>
            {name.replace(/_/g, ' ')}
          </span>
          <span
            style={{
              fontSize: 10,
              padding: '1px 7px',
              borderRadius: 100,
              background: isHealthy ? 'rgba(20,184,166,0.1)' : 'rgba(239,68,68,0.1)',
              color: isHealthy ? 'var(--teal)' : 'var(--red)',
              fontFamily: 'var(--font-ui)',
              fontWeight: 600,
              letterSpacing: '0.04em',
            }}
          >
            {component.status.toUpperCase()}
          </span>
        </div>
        {component.latency_ms != null && (
          <div style={{ fontSize: 11, color: 'var(--text-sec)', fontFamily: 'var(--font-mono)' }}>
            {component.latency_ms}ms latency
          </div>
        )}
        {component.error && (
          <div style={{ fontSize: 11, color: 'var(--red)', fontFamily: 'var(--font-mono)', marginTop: 4, wordBreak: 'break-all' }}>
            {component.error}
          </div>
        )}
        {component.version && (
          <div style={{ fontSize: 10, color: 'var(--text-muted)', marginTop: 2 }}>
            v{component.version}
          </div>
        )}
        {component.checked_at && (
          <div style={{ fontSize: 10, color: 'var(--text-muted)', marginTop: 2 }}>
            Checked {new Date(component.checked_at).toLocaleTimeString()}
          </div>
        )}
      </div>
    </div>
  )
}

export default function SystemHealth() {
  const [health, setHealth] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [lastRefreshed, setLastRefreshed] = useState(null)

  async function fetchHealth() {
    try {
      const data = await getSystemHealth()
      setHealth(data)
      setLastRefreshed(new Date())
      setError(null)
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchHealth()
    const id = setInterval(fetchHealth, 30_000)
    return () => clearInterval(id)
  }, [])

  const overallColor =
    health?.status === 'healthy' ? 'var(--teal)' :
    health?.status === 'degraded' ? 'var(--amber)' : 'var(--red)'

  return (
    <div>
      {/* Header */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 20 }}>
        <div>
          <div style={{ fontFamily: 'var(--font-ui)', fontWeight: 700, fontSize: 16, color: 'var(--text-primary)', letterSpacing: '0.02em' }}>
            System Health
          </div>
          <div style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 2 }}>
            Auto-refreshes every 30s
            {lastRefreshed && ` · Last: ${lastRefreshed.toLocaleTimeString()}`}
          </div>
        </div>
        <button
          type="button"
          onClick={fetchHealth}
          style={{
            background: 'var(--bg-surface)',
            border: '1px solid var(--border-md)',
            borderRadius: 'var(--r-sm)',
            padding: '5px 12px',
            fontSize: 11,
            color: 'var(--text-sec)',
            cursor: 'pointer',
            fontFamily: 'var(--font-ui)',
          }}
        >
          Refresh
        </button>
      </div>

      {/* Overall status banner */}
      {health && (
        <div
          style={{
            border: `1px solid ${overallColor}`,
            borderRadius: 'var(--r)',
            padding: '12px 16px',
            marginBottom: 20,
            background: `${overallColor}08`,
            display: 'flex',
            alignItems: 'center',
            gap: 10,
          }}
        >
          <StatusDot status={health.status} />
          <div>
            <span style={{ fontFamily: 'var(--font-ui)', fontWeight: 700, fontSize: 13, color: overallColor, textTransform: 'uppercase' }}>
              {health.status}
            </span>
            <span style={{ marginLeft: 12, fontSize: 11, color: 'var(--text-sec)' }}>
              v{health.version} · {health.environment}
            </span>
          </div>
        </div>
      )}

      {loading && (
        <div style={{ color: 'var(--text-muted)', fontSize: 12, padding: '20px 0' }}>Checking components…</div>
      )}

      {error && !health && (
        <div className="inline-err">{error}</div>
      )}

      {/* Component grid */}
      {health?.components && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))', gap: 12 }}>
          {Object.entries(health.components).map(([name, comp]) => (
            <ComponentCard key={name} name={name} component={comp} />
          ))}
        </div>
      )}

      {/* Metadata */}
      {health && (
        <div style={{ marginTop: 24, fontSize: 10, color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
          Checked at {new Date(health.timestamp).toISOString()}
        </div>
      )}
    </div>
  )
}
