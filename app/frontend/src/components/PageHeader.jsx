/**
 * PageHeader — consistent enterprise page header.
 * Includes title, subtitle, optional back link, optional badge,
 * and optional right-side actions.
 */
import { Link } from 'react-router-dom'

export default function PageHeader({
  title,
  sub,
  back,
  backLabel = 'Back',
  actions,
  badge,
  pills,
  className = '',
}) {
  return (
    <div className={`ph ${className}`}>
      {back && (
        <Link to={back} className="back-btn">
          <svg width="12" height="12" viewBox="0 0 12 12" fill="none">
            <path d="M8 2L4 6l4 4" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
          </svg>
          {backLabel}
        </Link>
      )}
      <div className="ph-row">
        <div className="ph-left">
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px', flexWrap: 'wrap' }}>
            <h1 className="pt">{title}</h1>
            {badge}
          </div>
          {sub && <div className="ps">{sub}</div>}
          {pills && (
            <div style={{ display: 'flex', gap: '6px', marginTop: '10px', flexWrap: 'wrap' }}>
              {pills}
            </div>
          )}
        </div>
        {actions && <div className="ph-actions">{actions}</div>}
      </div>
    </div>
  )
}
