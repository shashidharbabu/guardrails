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

  if (isAuthenticated) return <Navigate to={from} replace />

  function submit(e) {
    e.preventDefault()
    const form = new FormData(e.currentTarget)
    signIn({
      email: form.get('email') || 'avery@northstar.ai',
      name: form.get('name') || 'Avery Stone',
      org: form.get('org') || 'Northstar AI Governance',
    })
    navigate(from, { replace: true })
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
              : 'Use your enterprise identity or a workspace email to open the protected AI safety console.'}
          </p>
        </div>

        <button type="button" className="sso-button" onClick={() => signIn()}>
          <span className="sso-icon">S</span>
          Continue with enterprise SSO
        </button>

        <div className="auth-divider"><span>or continue with email</span></div>

        <form onSubmit={submit} className="auth-form">
          {isRequest && (
            <>
              <label>
                Full name
                <input name="name" autoComplete="name" placeholder="Avery Stone" />
              </label>
              <label>
                Organization
                <input name="org" autoComplete="organization" placeholder="Northstar AI Governance" />
              </label>
            </>
          )}
          <label>
            Work email
            <input name="email" type="email" autoComplete="email" placeholder="you@company.com" required />
          </label>
          {!isRequest && (
            <label>
              Password
              <input name="password" type="password" autoComplete="current-password" placeholder="••••••••••••" />
            </label>
          )}
          <div className="auth-row">
            <label className="check-row">
              <input type="checkbox" defaultChecked />
              <span>{isRequest ? 'Send onboarding checklist' : 'Remember this device'}</span>
            </label>
            {!isRequest && <a href="#forgot">Forgot password?</a>}
          </div>
          <button className="btn btn-primary btn-xl" type="submit">
            {isRequest ? 'Create secure workspace' : 'Enter console'}
          </button>
        </form>

        <p className="auth-switch">
          {isRequest ? 'Already have access?' : 'Need a workspace?'}{' '}
          <Link to={isRequest ? '/sign-in' : '/request-access'}>
            {isRequest ? 'Sign in' : 'Request access'}
          </Link>
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
