import { useState } from 'react'
import { Link, Navigate, useLocation, useNavigate } from 'react-router-dom'
import { motion, useReducedMotion } from 'framer-motion'
import { useAuth } from '../../context/AuthContext'

function AuthShell({ mode }) {
  const { isAuthenticated, signIn } = useAuth()
  const navigate = useNavigate()
  const location = useLocation()
  const reduceMotion = useReducedMotion()
  const from = location.state?.from?.pathname || '/conversations'
  const isRequest = mode === 'request'
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)

  if (isAuthenticated) return <Navigate to={from} replace />

  async function submit(e) {
    e.preventDefault()
    setError('')
    setLoading(true)
    try {
      const form = new FormData(e.currentTarget)
      await signIn(
        form.get('username') || 'admin',
        form.get('password') || '',
      )
      navigate(from, { replace: true })
    } catch (err) {
      setError(err.message || 'Login failed')
    } finally {
      setLoading(false)
    }
  }

  return (
    <main className="auth-page">
      <Link to="/" className="brand-lockup auth-brand">
        <span className="brand-mark" aria-hidden="true" />
        <span>
          <strong>Guardrails</strong>
          <small>Enterprise AI Safety</small>
        </span>
      </Link>

      <motion.section
        className="auth-card"
        initial={reduceMotion ? false : { opacity: 0, y: 16, scale: 0.98 }}
        animate={reduceMotion ? {} : { opacity: 1, y: 0, scale: 1 }}
        transition={{ duration: 0.35, ease: [0.16, 1, 0.3, 1] }}
      >
        <div className="auth-card-head">
          <span className="eyebrow">{isRequest ? 'Enterprise onboarding' : 'Secure console access'}</span>
          <h1>{isRequest ? 'Request access to Guardrails.' : 'Sign in to Guardrails.'}</h1>
          <p>
            {isRequest
              ? 'Create a workspace-ready access profile for evaluation, gateway operations, and human review.'
              : 'Enter your credentials to access the protected AI safety console.'}
          </p>
        </div>

        {error && (
          <div style={{
            background: 'rgba(239,68,68,0.12)',
            border: '1px solid rgba(239,68,68,0.3)',
            borderRadius: 8,
            padding: '10px 14px',
            color: '#f87171',
            fontSize: 13,
            marginBottom: 12,
          }}>
            {error}
          </div>
        )}

        <form onSubmit={submit} className="auth-form">
          <label>
            Username
            <input
              name="username"
              autoComplete="username"
              placeholder="admin"
              required
              defaultValue="admin"
            />
          </label>
          <label>
            Password
            <input
              name="password"
              type="password"
              autoComplete="current-password"
              placeholder="••••••••••••"
              required
            />
          </label>
          <button className="btn btn-primary btn-xl" type="submit" disabled={loading}>
            {loading ? 'Signing in…' : 'Enter console'}
          </button>
        </form>

        <p className="auth-switch">
          Need access?{' '}
          <Link to="/request-access">Request access</Link>
        </p>
      </motion.section>

      <aside className="auth-assurance" aria-label="Platform assurances">
        <div><b>SSO-ready</b><span>SAML/OIDC compatible surface</span></div>
        <div><b>Audit-first</b><span>Traceable decisions and reviews</span></div>
        <div><b>Policy aware</b><span>Gateway thresholds and model controls</span></div>
      </aside>
    </main>
  )
}

export function SignIn() {
  return <AuthShell mode="sign-in" />
}

export function RequestAccess() {
  return <AuthShell mode="request" />
}
