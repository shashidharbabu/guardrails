import { useEffect, useState } from 'react'
import { NavLink, Outlet } from 'react-router-dom'

const NAV = [
  {
    section: 'Main',
    items: [
      {
        to: '/', label: 'Conversations', end: true,
        icon: <svg className="nav-icon" viewBox="0 0 14 14" fill="none"><rect x="1" y="2" width="12" height="2" rx="1" fill="currentColor"/><rect x="1" y="6" width="8" height="2" rx="1" fill="currentColor"/><rect x="1" y="10" width="10" height="2" rx="1" fill="currentColor"/></svg>,
      },
      {
        to: '/new', label: 'New Query',
        icon: <svg className="nav-icon" viewBox="0 0 14 14" fill="none"><path d="M7 1v12M1 7h12" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/></svg>,
      },
    ],
  },
  {
    section: 'Observability',
    items: [
      {
        to: '/gateway', label: 'Gateway',
        icon: <svg className="nav-icon" viewBox="0 0 14 14" fill="none"><path d="M2 4h10M2 7h10M2 10h10" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round"/><circle cx="5" cy="4" r="1.2" fill="currentColor"/><circle cx="9" cy="7" r="1.2" fill="currentColor"/><circle cx="6" cy="10" r="1.2" fill="currentColor"/></svg>,
      },
      {
        to: '/analytics', label: 'Analytics',
        icon: <svg className="nav-icon" viewBox="0 0 14 14" fill="none"><path d="M1 11L4 7l3 2 3-4 3 2" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round"/></svg>,
      },
    ],
  },
  {
    section: 'Quality',
    items: [
      {
        to: '/evaluation', label: 'Evaluation',
        icon: <svg className="nav-icon" viewBox="0 0 14 14" fill="none"><path d="M7 1l1.5 3L12 4.5l-2.5 2.5.5 3.5L7 9l-3 1.5.5-3.5L2 4.5 5.5 4z" stroke="currentColor" strokeWidth="1.1"/></svg>,
      },
      {
        to: '/feedback', label: 'Feedback',
        icon: <svg className="nav-icon" viewBox="0 0 14 14" fill="none"><rect x="1" y="4" width="12" height="8" rx="1.5" stroke="currentColor" strokeWidth="1.1"/><path d="M4 4V3a3 3 0 016 0v1" stroke="currentColor" strokeWidth="1.1"/></svg>,
        badge: '3',
      },
    ],
  },
  {
    section: 'System',
    items: [
      {
        to: '/settings', label: 'Settings',
        icon: <svg className="nav-icon" viewBox="0 0 14 14" fill="none"><circle cx="7" cy="7" r="2" stroke="currentColor" strokeWidth="1.1"/><path d="M7 1v2M7 11v2M1 7h2M11 7h2M2.9 2.9l1.4 1.4M9.7 9.7l1.4 1.4M2.9 11.1l1.4-1.4M9.7 4.3l1.4-1.4" stroke="currentColor" strokeWidth="1.1"/></svg>,
      },
    ],
  },
]

export default function Layout() {
  const [clock, setClock] = useState('')

  useEffect(() => {
    const tick = () => setClock(new Date().toLocaleTimeString('en-US', { hour12: false }))
    tick()
    const id = setInterval(tick, 1000)
    return () => clearInterval(id)
  }, [])

  return (
    <div className="shell">
      {/* Topbar */}
      <div className="topbar">
        <div className="logo-wrap">
          <svg width="28" height="28" viewBox="0 0 28 28" fill="none">
            <path d="M14 2L25 8V20L14 26L3 20V8L14 2Z" stroke="#14b8a6" strokeWidth="1.5" fill="rgba(20,184,166,0.08)"/>
            <path d="M14 7L21 11V17L14 21L7 17V11L14 7Z" fill="rgba(20,184,166,0.2)" stroke="#14b8a6" strokeWidth="1"/>
            <circle cx="14" cy="14" r="3" fill="#14b8a6"/>
          </svg>
          <div>
            <div className="logo-name">ENTERPRISE GUARDRAILS</div>
            <div className="logo-sub">AI Safety · CS298B</div>
          </div>
        </div>

        <div className="tb-right">
          <div className="st-row"><div className="st-dot g"/><span className="st-lbl">FastAPI :8000</span></div>
          <div className="tb-div"/>
          <div className="st-row"><div className="st-dot a pulsing"/><span className="st-lbl">Ollama qwen2.5:7b</span></div>
          <div className="tb-div"/>
          <div className="clock">{clock}</div>
        </div>
      </div>

      {/* Sidebar */}
      <div className="sidebar">
        {NAV.map(group => (
          <div key={group.section}>
            <div className="nav-sec">{group.section}</div>
            {group.items.map(item => (
              <NavLink
                key={item.to}
                to={item.to}
                end={item.end}
                className={({ isActive }) => `nav-item${isActive ? ' active' : ''}`}
              >
                {item.icon}
                {item.label}
                {item.badge && <span className="nav-badge">{item.badge}</span>}
              </NavLink>
            ))}
            <div className="nav-div"/>
          </div>
        ))}
      </div>

      {/* Main */}
      <main className="main">
        <Outlet />
      </main>
    </div>
  )
}
