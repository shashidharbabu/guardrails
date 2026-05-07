import { useEffect, useState } from 'react'
import { getAuditLogs } from '../api/client'
import LoadingSkeleton from '../components/LoadingSkeleton'

export default function AuditLogs() {
  const [logs, setLogs] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    getAuditLogs({ limit: 200 })
      .then(setLogs)
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  return (
    <div>
      <div style={{ marginBottom: 20 }}>
        <div style={{ fontFamily: 'var(--font-ui)', fontWeight: 700, fontSize: 16, color: 'var(--text-primary)', letterSpacing: '0.02em' }}>
          Audit Logs
        </div>
        <div style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 2 }}>
          Append-only record of all system actions · {loading ? '…' : logs.length} entries
        </div>
      </div>

      {loading && <LoadingSkeleton type="table" count={8} />}
      {error && <div className="inline-err">{error}</div>}

      {!loading && !error && logs.length === 0 && (
        <div style={{ background: 'var(--bg-surface)', border: '1px solid var(--border)', borderRadius: 'var(--r)', padding: '32px 20px', textAlign: 'center', color: 'var(--text-muted)', fontSize: 12 }}>
          No audit log entries yet.
        </div>
      )}

      {logs.length > 0 && (
        <div style={{ background: 'var(--bg-surface)', border: '1px solid var(--border)', borderRadius: 'var(--r)', overflow: 'hidden' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse' }}>
            <thead>
              <tr style={{ borderBottom: '1px solid var(--border)' }}>
                {['Timestamp', 'Action', 'Actor', 'Resource', 'Session'].map(h => (
                  <th key={h} style={{ padding: '8px 12px', textAlign: 'left', fontSize: 10, fontFamily: 'var(--font-ui)', fontWeight: 600, color: 'var(--text-muted)', letterSpacing: '0.06em', textTransform: 'uppercase' }}>
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {logs.map(log => (
                <tr key={log.id} style={{ borderBottom: '1px solid var(--border)' }}>
                  <td style={{ padding: '8px 12px', fontSize: 10, fontFamily: 'var(--font-mono)', color: 'var(--text-muted)', whiteSpace: 'nowrap' }}>
                    {new Date(log.timestamp).toLocaleString()}
                  </td>
                  <td style={{ padding: '8px 12px', fontSize: 11, fontFamily: 'var(--font-mono)', color: 'var(--text-primary)' }}>
                    {log.action}
                  </td>
                  <td style={{ padding: '8px 12px', fontSize: 11, fontFamily: 'var(--font-mono)', color: 'var(--text-sec)' }}>
                    {log.actor_id || '—'}
                    {log.actor_role && <span style={{ marginLeft: 4, fontSize: 9, color: 'var(--text-muted)' }}>({log.actor_role})</span>}
                  </td>
                  <td style={{ padding: '8px 12px', fontSize: 11, fontFamily: 'var(--font-mono)', color: 'var(--text-sec)' }}>
                    {log.resource_type || '—'}
                    {log.resource_id && <span style={{ marginLeft: 4, fontSize: 9, color: 'var(--text-muted)' }}>{log.resource_id.slice(0, 8)}…</span>}
                  </td>
                  <td style={{ padding: '8px 12px', fontSize: 10, fontFamily: 'var(--font-mono)', color: 'var(--text-muted)' }}>
                    {log.session_id ? log.session_id.slice(0, 12) + '…' : '—'}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}
