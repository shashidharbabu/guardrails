import { Link } from 'react-router-dom'
import { motion, useReducedMotion } from 'framer-motion'

const MODULES = [
  ['Gateway', 'Preflight policy enforcement for PII, jailbreak, and prompt-injection risk before model execution.'],
  ['Evaluation', 'Benchmark guardrail quality, retrieval confidence, and MAD outcomes with executive-ready evidence.'],
  ['Review', 'Human-in-the-loop queues for escalations, false-positive triage, and analyst feedback capture.'],
  ['Analytics', 'Operational safety metrics, decision trends, latency distribution, and model health in one place.'],
]

const SIGNALS = [
  ['99.9%', 'policy uptime target'],
  ['4.7k', 'governance evidence chunks'],
  ['3-layer', 'gateway, RAG, MAD verification'],
  ['SOC-ready', 'audit and review workflow'],
]

export default function Landing() {
  const reduceMotion = useReducedMotion()
  const reveal = reduceMotion ? {} : {
    initial: { opacity: 0, y: 18 },
    animate: { opacity: 1, y: 0 },
    transition: { duration: 0.55, ease: [0.16, 1, 0.3, 1] },
  }

  return (
    <main className="public-page">
      <nav className="public-nav" aria-label="Public navigation">
        <Link to="/" className="brand-lockup">
          <span className="brand-mark" aria-hidden="true" />
          <span>
            <strong>Guardrails</strong>
            <small>Enterprise AI Safety</small>
          </span>
        </Link>
        <div className="public-nav-links">
          <a href="#platform">Platform</a>
          <a href="#workflow">Workflow</a>
          <a href="#trust">Trust</a>
        </div>
        <div className="public-nav-actions">
          <Link to="/sign-in" className="btn btn-ghost">Sign in</Link>
          <Link to="/request-access" className="btn btn-primary">Request access</Link>
        </div>
      </nav>

      <section className="landing-hero">
        <motion.div className="hero-copy" {...reveal}>
          <div className="eyebrow">AI governance, evaluation, and enforcement</div>
          <h1>Operate production AI with guardrails your risk team can trust.</h1>
          <p>
            Guardrails unifies live gateway enforcement, evaluation, monitoring, and human review into a premium control plane for enterprise AI safety.
          </p>
          <div className="hero-actions">
            <Link to="/request-access" className="btn btn-primary btn-xl">Request enterprise access</Link>
            <Link to="/sign-in" className="btn btn-glass btn-xl">Open console</Link>
          </div>
          <div className="hero-proof">
            {SIGNALS.map(([value, label]) => (
              <div key={label}>
                <strong>{value}</strong>
                <span>{label}</span>
              </div>
            ))}
          </div>
        </motion.div>

        <motion.div
          className="hero-product-card"
          initial={reduceMotion ? false : { opacity: 0, scale: 0.96, y: 20 }}
          animate={reduceMotion ? {} : { opacity: 1, scale: 1, y: 0 }}
          transition={{ duration: 0.7, delay: 0.08, ease: [0.16, 1, 0.3, 1] }}
          aria-label="Product preview"
        >
          <div className="preview-top">
            <span className="brand-mark mini" />
            <span>Live Safety Control</span>
            <b>Production</b>
          </div>
          <div className="preview-grid">
            <div className="preview-main">
              <div className="preview-title">Gateway decision stream</div>
              {['PASS', 'ESCALATE', 'BLOCK', 'PASS'].map((status, i) => (
                <div key={`${status}-${i}`} className={`preview-row ${status.toLowerCase()}`}>
                  <span>{status}</span>
                  <p>{['Policy-safe compliance answer', 'Clinical caveat requires analyst', 'PII exfiltration attempt', 'EU AI Act retrieval query'][i]}</p>
                  <em>{[0.08, 0.62, 0.91, 0.14][i].toFixed(2)}</em>
                </div>
              ))}
            </div>
            <div className="preview-side">
              <div className="orbit-score">94<span>%</span></div>
              <p>Evaluation confidence</p>
              <div className="mini-bars">
                <i style={{ '--w': '78%' }} />
                <i style={{ '--w': '52%' }} />
                <i style={{ '--w': '88%' }} />
              </div>
            </div>
          </div>
        </motion.div>
      </section>

      <section id="platform" className="landing-section">
        <div className="section-kicker">One platform</div>
        <h2>Policy enforcement, observability, and review designed as one operating system.</h2>
        <div className="module-grid">
          {MODULES.map(([title, copy]) => (
            <article key={title} className="module-card">
              <span>{title.slice(0, 2)}</span>
              <h3>{title}</h3>
              <p>{copy}</p>
            </article>
          ))}
        </div>
      </section>

      <section id="workflow" className="workflow-band">
        <div>
          <div className="section-kicker">Operational clarity</div>
          <h2>Every decision carries the evidence, trace, and review path your teams need.</h2>
          <p>
            Route risky requests before they reach models, inspect full session traces, compare evaluation performance, and close the loop with analyst feedback.
          </p>
        </div>
        <div className="workflow-steps">
          {['Validate', 'Retrieve', 'Debate', 'Decide', 'Review'].map((step, i) => (
            <div key={step}>
              <b>{String(i + 1).padStart(2, '0')}</b>
              <span>{step}</span>
            </div>
          ))}
        </div>
      </section>

      <section id="trust" className="landing-cta">
        <div>
          <div className="section-kicker">Built for serious AI teams</div>
          <h2>Give safety, product, and compliance teams the same source of truth.</h2>
        </div>
        <Link to="/request-access" className="btn btn-primary btn-xl">Start with Guardrails</Link>
      </section>

      <footer className="public-footer">
        <span>Guardrails</span>
        <span>Enterprise AI safety platform</span>
        <Link to="/sign-in">Sign in</Link>
      </footer>
    </main>
  )
}
