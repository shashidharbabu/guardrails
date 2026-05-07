export default function RAGSpans({ evidencePool = [] }) {
  if (!evidencePool.length) {
    return (
      <div style={{
        border: '1px solid var(--border)',
        borderRadius: 'var(--r-lg)',
        padding: 16,
        background: 'var(--bg-card)',
        fontFamily: 'var(--font-ui)',
        fontSize: 12,
        color: 'var(--text-muted)',
      }}>
        No evidence retrieved.
      </div>
    )
  }

  return (
    <div className="panel">
      <div className="panel-hdr">
        <div className="panel-title">Evidence Pool</div>
        <span style={{ fontFamily: 'var(--font-mono)', fontSize: 10, color: 'var(--text-muted)' }}>
          {evidencePool.length} chunk{evidencePool.length !== 1 ? 's' : ''}
        </span>
      </div>
      <div className="panel-body" style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
        {evidencePool.map((chunk, i) => (
          <div key={chunk.chunk_id ?? i} style={{
            border: '1px solid var(--border)',
            borderRadius: 'var(--r-md)',
            padding: '10px 12px',
            background: 'var(--bg-base)',
          }}>
            <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', gap: 8, marginBottom: 6 }}>
              <p style={{ fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--blue-hi)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', flex: 1 }}>
                {chunk.source}
              </p>
              {chunk.relevance_score != null && (
                <span style={{ fontFamily: 'var(--font-mono)', fontSize: 10, color: 'var(--text-muted)', flexShrink: 0 }}>
                  {(chunk.relevance_score * 100).toFixed(0)}% rel
                </span>
              )}
            </div>
            <p style={{ fontFamily: 'var(--font-ui)', fontSize: 12, color: 'var(--text-sec)', lineHeight: 1.6 }}>
              {chunk.text}
            </p>
          </div>
        ))}
      </div>
    </div>
  )
}
