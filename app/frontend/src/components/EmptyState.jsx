/**
 * EmptyState — premium empty state component with icon, title,
 * description, and optional CTA.
 */
export default function EmptyState({
  icon,
  title,
  description,
  action,
  className = '',
}) {
  return (
    <div className={`empty-state ${className}`}>
      {icon && (
        <div
          className="empty-icon"
          style={{
            width: 48,
            height: 48,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            background: 'var(--bg-hover)',
            border: '1px solid var(--border-md)',
            borderRadius: 'var(--r-lg)',
            marginBottom: 6,
          }}
        >
          {icon}
        </div>
      )}
      <div className="empty-title">{title}</div>
      {description && <div className="empty-desc">{description}</div>}
      {action && (
        <div style={{ marginTop: '16px' }}>{action}</div>
      )}
    </div>
  )
}

/* Pre-built icon shortcuts */
export function NoDataIcon() {
  return (
    <svg viewBox="0 0 24 24" fill="none" width="22" height="22" stroke="currentColor" strokeWidth="1.4">
      <rect x="4" y="5" width="16" height="14" rx="2"/>
      <path d="M8 9h8M8 13h5" strokeLinecap="round"/>
    </svg>
  )
}

export function NoSessionsIcon() {
  return (
    <svg viewBox="0 0 24 24" fill="none" width="22" height="22" stroke="currentColor" strokeWidth="1.4">
      <circle cx="12" cy="12" r="9"/>
      <path d="M12 8v4M12 16h.01" strokeLinecap="round"/>
    </svg>
  )
}

export function NoResultsIcon() {
  return (
    <svg viewBox="0 0 24 24" fill="none" width="22" height="22" stroke="currentColor" strokeWidth="1.4">
      <circle cx="11" cy="11" r="7"/>
      <path d="M16 16l4 4" strokeLinecap="round"/>
      <path d="M9 9l4 4M13 9l-4 4" strokeLinecap="round"/>
    </svg>
  )
}

export function InboxClearIcon() {
  return (
    <svg viewBox="0 0 24 24" fill="none" width="22" height="22" stroke="currentColor" strokeWidth="1.4">
      <path d="M3 12l3-7h12l3 7" strokeLinecap="round" strokeLinejoin="round"/>
      <rect x="3" y="12" width="18" height="8" rx="1.5"/>
      <path d="M9 16h6" strokeLinecap="round"/>
    </svg>
  )
}

export function ErrorIcon() {
  return (
    <svg viewBox="0 0 24 24" fill="none" width="22" height="22" stroke="currentColor" strokeWidth="1.4">
      <path d="M12 2L2 20h20L12 2z" strokeLinejoin="round"/>
      <path d="M12 9v5M12 16.5h.01" strokeLinecap="round"/>
    </svg>
  )
}

/* Error state — themed banner for API/fetch errors */
export function ErrorState({ title = 'Failed to load data', message, onRetry }) {
  return (
    <div
      style={{
        padding: '28px 24px',
        background: 'var(--red-dim)',
        border: '1px solid var(--red-mid)',
        borderRadius: 'var(--r-lg)',
        display: 'flex',
        alignItems: 'flex-start',
        gap: '16px',
      }}
    >
      <div
        style={{
          width: 36,
          height: 36,
          borderRadius: 'var(--r-md)',
          background: 'rgba(239,68,68,0.12)',
          border: '1px solid var(--red-mid)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          flexShrink: 0,
          color: 'var(--red-hi)',
        }}
      >
        <svg viewBox="0 0 20 20" fill="none" width="18" height="18" stroke="currentColor" strokeWidth="1.5">
          <path d="M10 2L2 17h16L10 2z" strokeLinejoin="round"/>
          <path d="M10 8v4M10 14.5h.01" strokeLinecap="round"/>
        </svg>
      </div>
      <div style={{ flex: 1 }}>
        <div style={{
          fontFamily: 'var(--font-ui)',
          fontWeight: 600,
          fontSize: '14px',
          color: 'var(--red-hi)',
          marginBottom: '5px',
        }}>
          {title}
        </div>
        {message && (
          <div style={{
            fontFamily: 'var(--font-ui)',
            fontSize: '12px',
            color: 'rgba(248,113,113,0.75)',
            lineHeight: 1.6,
          }}>
            {message}
          </div>
        )}
        {onRetry && (
          <button
            onClick={onRetry}
            style={{
              marginTop: 12,
              padding: '5px 14px',
              fontFamily: 'var(--font-ui)',
              fontSize: '11px',
              fontWeight: 600,
              background: 'rgba(239,68,68,0.12)',
              border: '1px solid var(--red-mid)',
              borderRadius: 'var(--r-md)',
              color: 'var(--red-hi)',
              cursor: 'pointer',
              transition: 'all var(--motion-base)',
            }}
          >
            Retry
          </button>
        )}
      </div>
    </div>
  )
}
