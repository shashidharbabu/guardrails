import { useEffect, useState } from 'react'
import PageHeader from '../components/PageHeader'
import SectionHeader from '../components/SectionHeader'

/* ─── Service health hook ────────────────────────────────────── */
function useServiceHealth(url) {
  const [status, setStatus] = useState('checking')
  useEffect(() => {
    fetch(url, { signal: AbortSignal.timeout(4000) })
      .then(r => setStatus(r.ok ? 'ok' : 'error'))
      .catch(() => setStatus('error'))
  }, [url])
  return status
}

/* ─── Threshold slider ───────────────────────────────────────── */
function ThresholdRow({ label, defaultValue, color, hint }) {
  const [val, setVal] = useState(defaultValue)
  return (
    <div className="slider-wrap">
      <div className="slider-label">
        <span className="slider-name">{label}</span>
        <span className="slider-value" style={{ color }}>{val.toFixed(2)}</span>
      </div>
      <input
        type="range"
        min="0" max="1" step="0.05"
        value={val}
        onChange={e => setVal(parseFloat(e.target.value))}
        className="slider"
        style={{ accentColor: color.includes('red') ? 'var(--red)' : color.includes('green') ? 'var(--green)' : 'var(--blue)' }}
      />
      {hint && <div className="slider-hint">{hint}</div>}
    </div>
  )
}

/* ─── KV detail row ──────────────────────────────────────────── */
function KvRow({ label, value }) {
  return (
    <div className="lt-row">
      <span className="lt-s">{label}</span>
      <span className="lt-v" style={{ fontSize: 11, color: 'var(--teal-hi)', fontFamily: 'var(--font-mono)', textAlign: 'right', maxWidth: '55%', wordBreak: 'break-all' }}>
        {value}
      </span>
    </div>
  )
}

/* ─── Service integration row ────────────────────────────────── */
function IntItem({ name, status, note }) {
  const statusConfig = {
    ok:       { dot: 'var(--green)',       label: 'Running',         cls: 'b-pass',   glow: '0 0 5px rgba(34,197,94,0.4)' },
    error:    { dot: 'var(--red)',         label: 'Unreachable',     cls: 'b-block',  glow: 'none' },
    checking: { dot: 'var(--amber)',       label: 'Checking…',       cls: 'b-esc',    glow: 'none' },
    manual:   { dot: 'var(--text-muted)', label: note || 'Manual',  cls: 'b-gray',   glow: 'none' },
  }
  const cfg = statusConfig[status] || statusConfig.manual

  return (
    <div className="int-item">
      <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
        <div style={{
          width: 7,
          height: 7,
          borderRadius: '50%',
          background: cfg.dot,
          flexShrink: 0,
          boxShadow: cfg.glow,
        }} />
        <div className="int-name">{name}</div>
      </div>
      <div className="int-status">
        <span className={`badge ${cfg.cls}`} style={{ fontSize: 10 }}>
          {status === 'checking' ? (
            <span className="pulsing">{cfg.label}</span>
          ) : cfg.label}
        </span>
      </div>
    </div>
  )
}

/* ─── Config panel ───────────────────────────────────────────── */
function ConfigPanel({ title, children, footer }) {
  return (
    <div className="panel">
      <div className="panel-hdr">
        <div className="panel-title">{title}</div>
      </div>
      <div className="panel-body">
        {children}
      </div>
      {footer && (
        <div style={{
          padding: '10px 20px',
          borderTop: '1px solid var(--border)',
          fontSize: 11,
          fontFamily: 'var(--font-ui)',
          color: 'var(--text-muted)',
          lineHeight: 1.7,
        }}>
          {footer}
        </div>
      )}
    </div>
  )
}

/* ─── Main page ──────────────────────────────────────────────── */
export default function Settings() {
  const backendHealth = useServiceHealth('/healthz')
  const gatewayHealth = useServiceHealth('/api/gateway/health')
  const [gatewayConfig, setGatewayConfig] = useState(null)

  useEffect(() => {
    fetch('/api/gateway/config')
      .then(r => r.ok ? r.json() : null)
      .then(data => setGatewayConfig(data))
      .catch(() => {})
  }, [])

  return (
    <div>
      <PageHeader
        title="Settings"
        sub="Routing thresholds · model configuration · service integrations · system overview"
      />

      {/* ─── Gateway Configuration ──────────────── */}
      <SectionHeader label="Gateway Configuration" />
      <div className="settings-grid" style={{ marginBottom: 28 }}>
        <ConfigPanel
          title="Routing Thresholds"
          footer={
            <>
              Composite score = PII×{gatewayConfig?.pii_weight ?? 0.3} + JB×{gatewayConfig?.jailbreak_weight ?? 0.4} + PI×{gatewayConfig?.injection_weight ?? 0.3}.
              {' '}For per-validator control, use{' '}
              <a href="/gateway" style={{ color: 'var(--blue-hi)', textDecoration: 'none' }}>
                Gateway → Model Config
              </a>.
            </>
          }
        >
          <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
            <ThresholdRow
              label="Pass threshold"
              defaultValue={gatewayConfig?.thresholds?.pass_threshold ?? 0.30}
              color="var(--green-hi)"
              hint="Composite score below this → PASS"
            />
            <ThresholdRow
              label="Block threshold"
              defaultValue={gatewayConfig?.thresholds?.block_threshold ?? 0.70}
              color="var(--red-hi)"
              hint="Composite score above this → BLOCK"
            />
          </div>
        </ConfigPanel>

        <ConfigPanel title="Service Integrations">
          <IntItem name="App Backend (:8000)"           status={backendHealth} />
          <IntItem name="Gateway FastAPI (:8080)"       status={gatewayHealth} />
          <IntItem name="Ollama (local :11434)"         status="manual" note="Not checked" />
          <IntItem name="Qdrant Cloud (GCP us-east4)"   status="manual" note="Key required" />
          <IntItem name="Langfuse"                      status="manual" note="Not configured" />
        </ConfigPanel>
      </div>

      {/* ─── System Overview ────────────────────── */}
      <SectionHeader label="System Overview" />
      <div className="settings-grid">
        <ConfigPanel title="Active Models">
          <div className="lat-tbl">
            <KvRow label="PII detection"    value="shashidharbabu/deberta-pii-guardrails" />
            <KvRow label="Jailbreak"        value="shashidharbabu/roberta-jailbreak-guardrails" />
            <KvRow label="Prompt injection" value="shashidharbabu/llama-prompt-guard-guardrails" />
            <KvRow label="Embedder"         value="nvidia/llama-embed-nemotron-8b" />
            <KvRow label="LLM"             value="qwen2.5:7b (Ollama)" />
            <KvRow label="MAD agents"       value="qwen2.5:7b (Ollama)" />
          </div>
        </ConfigPanel>

        <ConfigPanel title="Infrastructure">
          <div className="lat-tbl">
            <KvRow label="Database"       value="SQLite (app_sessions.db)" />
            <KvRow label="Vector store"   value="Qdrant Cloud · GCP us-east4" />
            <KvRow label="Corpus chunks"  value="4,664" />
            <KvRow label="Collection"     value="ai_governance_chunks_nemotron8b" />
            <KvRow label="Embed dim"      value="4096 (Nemotron-8B)" />
            <KvRow label="Similarity"     value="cosine · HNSW" />
          </div>
        </ConfigPanel>
      </div>

      {/* ─── Pipeline Config ────────────────────── */}
      <SectionHeader label="Pipeline Configuration" />
      <div className="settings-grid">
        <ConfigPanel title="MAD Debate Settings">
          <div className="lat-tbl">
            <KvRow label="Cycles"         value="2 (configurable)" />
            <KvRow label="Agent A"        value="qwen2.5:7b" />
            <KvRow label="Agent B"        value="qwen2.5:7b" />
            <KvRow label="Judge model"    value="qwen2.5:7b" />
            <KvRow label="Routing"        value="DELIVER · RETRY · HARD_BLOCK · HUMAN_REVIEW" />
          </div>
        </ConfigPanel>

        <ConfigPanel title="RAG Settings">
          <div className="lat-tbl">
            <KvRow label="Top-K"          value="5 chunks" />
            <KvRow label="Score threshold" value="0.35 cosine" />
            <KvRow label="Domain"         value="AI governance · compliance" />
            <KvRow label="Corpus tiers"   value="T0 (security) · T1-T3 (regulation)" />
            <KvRow label="Index"          value="HNSW" />
          </div>
        </ConfigPanel>
      </div>
    </div>
  )
}
