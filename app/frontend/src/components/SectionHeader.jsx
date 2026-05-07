/**
 * SectionHeader — premium section label with optional count badge and action.
 */
export default function SectionHeader({ label, count, action, className = '' }) {
  return (
    <div className={`sec-lbl ${className}`}>
      <span>{label}</span>
      <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
        {count != null && (
          <span className="sec-count">{count}</span>
        )}
        {action}
      </div>
    </div>
  )
}
