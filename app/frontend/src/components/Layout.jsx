import { useEffect, useState } from 'react'
import { NavLink, Outlet, useLocation } from 'react-router-dom'
import { motion, AnimatePresence, useReducedMotion } from 'framer-motion'
import CopilotPanel from './CopilotPanel'

const NAV = [
  {
    section: 'Observe',
    items: [
      {
        to: '/', label: 'Conversations', end: true,
        icon: (
          <svg className="nav-icon" viewBox="0 0 16 16" fill="none">
            <path d="M2 4h12M2 8h8M2 12h10" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round"/>
          </svg>
        ),
      },
      {
        to: '/gateway', label: 'Gateway',
        icon: (
          <svg className="nav-icon" viewBox="0 0 16 16" fill="none">
            <path d="M3 5h10M3 8h10M3 11h10" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round"/>
            <circle cx="6" cy="5" r="1.5" fill="currentColor"/>
            <circle cx="10" cy="8" r="1.5" fill="currentColor"/>
            <circle cx="7" cy="11" r="1.5" fill="currentColor"/>
          </svg>
        ),
      },
      {
        to: '/analytics', label: 'Analytics',
        icon: (
          <svg className="nav-icon" viewBox="0 0 16 16" fill="none">
            <path d="M2 12L5 8l3 2.5 3-5 3 2.5" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round"/>
          </svg>
        ),
      },
    ],
  },
  {
    section: 'Review',
    items: [
      {
        to: '/human-review', label: 'Human Review',
        icon: (
          <svg className="nav-icon" viewBox="0 0 16 16" fill="none">
            <circle cx="8" cy="5" r="2.5" stroke="currentColor" strokeWidth="1.2"/>
            <path d="M3 13c0-2.76 2.24-5 5-5s5 2.24 5 5" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round"/>
            <path d="M11 9l1.5 1.5L14 9" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round" strokeLinejoin="round"/>
          </svg>
        ),
      },
      {
        to: '/feedback', label: 'Feedback',
        icon: (
          <svg className="nav-icon" viewBox="0 0 16 16" fill="none">
            <path d="M3 7a5 5 0 0110 0v4.5a1.5 1.5 0 01-1.5 1.5h-7A1.5 1.5 0 013 11.5V7z" stroke="currentColor" strokeWidth="1.2"/>
            <path d="M6 4.5V4a2 2 0 014 0v.5" stroke="currentColor" strokeWidth="1.2"/>
          </svg>
        ),
      },
      {
        to: '/evaluation', label: 'Evaluation',
        icon: (
          <svg className="nav-icon" viewBox="0 0 16 16" fill="none">
            <path d="M8 1.5l1.7 3.4L14 5.6l-3 2.9.7 4L8 10.6l-3.7 1.9.7-4L2 5.6l4.3-.7z" stroke="currentColor" strokeWidth="1.2"/>
          </svg>
        ),
      },
    ],
  },
  {
    section: 'Operate',
    items: [
      {
        to: '/new', label: 'Test Query',
        icon: (
          <svg className="nav-icon" viewBox="0 0 16 16" fill="none">
            <circle cx="8" cy="8" r="6" stroke="currentColor" strokeWidth="1.2"/>
            <path d="M6 8h4M8 6v4" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round"/>
          </svg>
        ),
      },
      {
        to: '/system-health', label: 'System Health',
        icon: (
          <svg className="nav-icon" viewBox="0 0 16 16" fill="none">
            <path d="M2 8h2l2-4 2 8 2-4 2 4 2-4" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round"/>
          </svg>
        ),
      },
      {
        to: '/audit', label: 'Audit Logs',
        icon: (
          <svg className="nav-icon" viewBox="0 0 16 16" fill="none">
            <rect x="3" y="2" width="10" height="12" rx="1.5" stroke="currentColor" strokeWidth="1.2"/>
            <path d="M5.5 5h5M5.5 8h5M5.5 11h3" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round"/>
          </svg>
        ),
      },
      {
        to: '/settings', label: 'Settings',
        icon: (
          <svg className="nav-icon" viewBox="0 0 16 16" fill="none">
            <circle cx="8" cy="8" r="2.2" stroke="currentColor" strokeWidth="1.2"/>
            <path d="M8 2v1.5M8 12.5V14M2 8h1.5M12.5 8H14M3.5 3.5l1.1 1.1M11.4 11.4l1.1 1.1M3.5 12.5l1.1-1.1M11.4 4.6l1.1-1.1" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round"/>
          </svg>
        ),
      },
    ],
  },
]

function derivePageName(pathname) {
  if (pathname === '/') return 'conversations'
  if (pathname.startsWith('/sessions/')) return 'session_trace'
  if (pathname === '/analytics') return 'analytics'
  if (pathname === '/human-review') return 'human_review'
  if (pathname === '/audit') return 'audit_logs'
  if (pathname === '/system-health') return 'system_health'
  if (pathname === '/gateway') return 'gateway'
  if (pathname === '/feedback') return 'feedback'
  if (pathname === '/evaluation') return 'evaluation'
  if (pathname === '/new') return 'test_query'
  if (pathname === '/settings') return 'settings'
  return 'unknown'
}

export default function Layout() {
  const [clock, setClock] = useState('')
  const [copilotOpen, setCopilotOpen] = useState(false)
  const location = useLocation()
  const shouldReduceMotion = useReducedMotion()

  const sessionIdMatch = location.pathname.match(/^\/sessions\/([^/]+)$/)
  const pageContext = {
    page: derivePageName(location.pathname),
    session_id: sessionIdMatch ? sessionIdMatch[1] : undefined,
  }

  useEffect(() => {
    const tick = () =>
      setClock(
        new Date().toLocaleTimeString('en-US', {
          hour12: false,
          hour: '2-digit',
          minute: '2-digit',
          second: '2-digit',
        })
      )
    tick()
    const id = setInterval(tick, 1000)
    return () => clearInterval(id)
  }, [])

  const pageVariants = shouldReduceMotion
    ? {}
    : {
        initial:  { opacity: 0, y: 5 },
        animate:  { opacity: 1, y: 0, transition: { duration: 0.16, ease: [0.16, 1, 0.3, 1] } },
        exit:     { opacity: 0, transition: { duration: 0.1 } },
      }

  return (
    <div className="shell">
      {/* ─── Topbar ─────────────────────────────────────────── */}
      <header className="topbar" role="banner">
        <div className="logo-wrap">
          <div className="logo-icon" aria-hidden="true">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
              <path d="M12 2L21 7V17L12 22L3 17V7L12 2Z" stroke="var(--blue)" strokeWidth="1.5" fill="rgba(59,130,246,0.06)"/>
              <path d="M12 6L17.5 9V15L12 18L6.5 15V9L12 6Z" fill="rgba(59,130,246,0.1)" stroke="var(--blue)" strokeWidth="1"/>
              <circle cx="12" cy="12" r="2.5" fill="var(--blue)"/>
            </svg>
          </div>
          <div>
            <div className="logo-name">GUARDRAILS</div>
            <div className="logo-sub">AI SAFETY PLATFORM</div>
          </div>
        </div>

        <div className="tb-div" />

        <div className="env-pill">
          <span className="env-dot" />
          {import.meta.env.VITE_APP_ENV || 'Development'}
        </div>

        <div className="tb-right">
          <div className="st-row">
            <div className="st-dot g" />
            <span className="st-lbl">API</span>
          </div>
          <div className="tb-div" />
          <div className="st-row">
            <div className="st-dot a pulsing" />
            <span className="st-lbl">{import.meta.env.VITE_DEFAULT_MODEL || 'qwen2.5:7b'}</span>
          </div>
          <div className="tb-div" />
          <div className="clock" aria-label="Current time">{clock}</div>
        </div>
      </header>

      {/* ─── Sidebar ─────────────────────────────────────────── */}
      <nav className="sidebar" aria-label="Main navigation">
        {NAV.map(group => (
          <div key={group.section}>
            <div className="nav-sec" aria-hidden="true">{group.section}</div>
            {group.items.map(item => (
              <NavLink
                key={item.to}
                to={item.to}
                end={item.end}
                className={({ isActive }) => `nav-item${isActive ? ' active' : ''}`}
                aria-current={item.to === location.pathname ? 'page' : undefined}
              >
                {item.icon}
                {item.label}
                {item.badge && (
                  <span className="nav-badge" aria-label={`${item.badge} items`}>{item.badge}</span>
                )}
              </NavLink>
            ))}
            <div className="nav-div" aria-hidden="true" />
          </div>
        ))}

        <div style={{ marginTop: 'auto', padding: '14px 20px', fontSize: '9px', fontFamily: 'var(--font-ui)', fontWeight: 500, color: 'var(--text-muted)', letterSpacing: '0.06em', textTransform: 'uppercase' }}>
          v{import.meta.env.VITE_APP_VERSION || '1.0.0'} · Enterprise
        </div>
      </nav>

      {/* ─── Main ─────────────────────────────────────────────── */}
      <main className="main" id="main-content">
        <AnimatePresence mode="wait">
          <motion.div
            key={location.pathname}
            className="page-body"
            {...(shouldReduceMotion ? {} : pageVariants)}
            initial="initial"
            animate="animate"
            exit="exit"
          >
            <Outlet />
          </motion.div>
        </AnimatePresence>
      </main>

      {/* ─── Co-pilot FAB ────────────────────────────────────── */}
      <button
        type="button"
        className={`copilot-fab${copilotOpen ? ' copilot-fab-open' : ''}`}
        onClick={() => setCopilotOpen(o => !o)}
        aria-label={copilotOpen ? 'Close AI co-pilot' : 'Open AI co-pilot'}
        aria-expanded={copilotOpen}
      >
        {copilotOpen ? (
          <svg width="20" height="20" viewBox="0 0 20 20" fill="none" aria-hidden="true">
            <path d="M5 5l10 10M15 5L5 15" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round"/>
          </svg>
        ) : (
          <svg width="20" height="20" viewBox="0 0 20 20" fill="none" aria-hidden="true">
            <path d="M10 2.5l2 4 4.5 1.5L13 11l.8 4.5-3.8-2-3.8 2L7 11l-3.5-3L8 6.5z"
              stroke="currentColor" strokeWidth="1.5" fill="rgba(255,255,255,0.08)" strokeLinejoin="round"/>
          </svg>
        )}
      </button>

      {/* ─── Co-pilot Panel ──────────────────────────────────── */}
      <AnimatePresence>
        {copilotOpen && (
          <CopilotPanel
            onClose={() => setCopilotOpen(false)}
            pageContext={pageContext}
          />
        )}
      </AnimatePresence>
    </div>
  )
}
