/**
 * MetricCard — enterprise KPI card with label, value, sub-text,
 * optional top accent strip, optional icon, and optional delta.
 *
 * accent: 'blue' | 'teal' | 'green' | 'red' | 'amber'
 * delta: { value: '+12%', dir: 'up' | 'down' | 'neutral' }
 * icon: JSX element (SVG, 16x16 recommended)
 */
export default function MetricCard({
  label,
  value,
  sub,
  accent,
  delta,
  valueColor,
  icon,
  className = '',
}) {
  return (
    <div className={`stat-card${accent ? ` accent-${accent}` : ''} ${className}`}>
      {icon && (
        <div className={`sc-icon ${accent || 'blue'}`}>
          {icon}
        </div>
      )}
      <div className="sl">{label}</div>
      <div className="sv" style={valueColor ? { color: valueColor } : {}}>
        {value}
      </div>
      {sub && <div className="ss">{sub}</div>}
      {delta && (
        <div className={`stat-delta ${delta.dir || 'neutral'}`}>
          {delta.dir === 'up' && (
            <svg width="9" height="9" viewBox="0 0 9 9" fill="none">
              <path d="M4.5 8V1M1.5 4l3-3 3 3" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
          )}
          {delta.dir === 'down' && (
            <svg width="9" height="9" viewBox="0 0 9 9" fill="none">
              <path d="M4.5 1v7M1.5 5l3 3 3-3" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
          )}
          {delta.value}
        </div>
      )}
    </div>
  )
}
