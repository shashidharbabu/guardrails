import { useEffect, useState } from 'react'

function useServiceHealth(url) {
  const [status, setStatus] = useState('checking')
  useEffect(() => {
    fetch(url, { signal: AbortSignal.timeout(4000) })
      .then(r => setStatus(r.ok ? 'ok' : 'error'))
      .catch(() => setStatus('error'))
  }, [url])
  return status
}

function StatusDot({ status }) {
  const color =
    status === 'ok'       ? 'var(--green)'  :
    status === 'error'    ? 'var(--red)'    :
    status === 'checking' ? 'var(--amber)'  : 'var(--text-muted)'
  return (
    <span style={{
      display: 'inline-block', width: 7, height: 7, borderRadius: '50%',
      background: color, marginRight: 6, flexShrink: 0,
    }} />
  )
}

function StatusLabel({ status }) {
  const label =
    status === 'ok'       ? 'running' :
    status === 'error'    ? 'unreachable' :
    status === 'checking' ? 'checking…'   : 'unknown'
  const color =
    status === 'ok'    ? 'var(--green)' :
    status === 'error' ? 'var(--red)'   : 'var(--amber)'
  return <span style={{ color, fontSize: '10px' }}>{label}</span>
}

function ThresholdRow({ label, defaultValue, color }) {
  const [val, setVal] = useState(defaultValue)
  return (
    <div>
      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '11px', fontFamily: 'var(--font-mono)', marginBottom: '6px' }}>
        <span style={{ color: 'var(--text-sec)' }}>{label}</span>
        <span style={{ color }}>{val.toFixed(2)}</span>
      </div>
      <input
        type="range" min="0" max="1" step="0.05"
        value={val}
        onChange={e => setVal(parseFloat(e.target.value))}
        style={{ width: '100%', accentColor: color }}
      />
    </div>
  )
}

export default function Settings() {
  const backendHealth = useServiceHealth('/api/health')
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
      <div className="ph">
        <div className="pt">Settings</div>
        <div className="ps">Thresholds · models · integrations</div>
      </div>

      <div className="two-col">
        <div className="card">
          <div className="sec-lbl">Routing thresholds</div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '14px' }}>
            <ThresholdRow
              label="Pass threshold"
              defaultValue={gatewayConfig?.pass_threshold ?? 0.30}
              color="var(--green)"
            />
            <ThresholdRow
              label="Block threshold"
              defaultValue={gatewayConfig?.block_threshold ?? 0.70}
              color="var(--red)"
            />
            <div style={{ fontSize: '11px', fontFamily: 'var(--font-mono)', color: 'var(--text-muted)', padding: '8px', background: 'var(--bg-surface)', borderRadius: '4px', border: '1px solid var(--border)' }}>
              Composite: PII×{gatewayConfig?.pii_weight ?? 0.3} + JB×{gatewayConfig?.jailbreak_weight ?? 0.4} + PI×{gatewayConfig?.injection_weight ?? 0.3}
            </div>
          </div>
        </div>

        <div className="card">
          <div className="sec-lbl">Integrations</div>
          <div className="lat-tbl">
            <div className="lt-row">
              <span className="lt-s" style={{ display: 'flex', alignItems: 'center' }}>
                <StatusDot status={gatewayHealth} />Gateway FastAPI (:8080)
              </span>
              <StatusLabel status={gatewayHealth} />
            </div>
            <div className="lt-row">
              <span className="lt-s" style={{ display: 'flex', alignItems: 'center' }}>
                <StatusDot status={backendHealth} />App Backend (:8000)
              </span>
              <StatusLabel status={backendHealth} />
            </div>
            <div className="lt-row">
              <span className="lt-s">Ollama (local :11434)</span>
              <span style={{ color: 'var(--text-muted)', fontSize: '10px' }}>not checked</span>
            </div>
            <div className="lt-row">
              <span className="lt-s">Qdrant Cloud (GCP)</span>
              <span style={{ color: 'var(--text-muted)', fontSize: '10px' }}>key required</span>
            </div>
            <div className="lt-row">
              <span className="lt-s">Langfuse</span>
              <span style={{ color: 'var(--text-muted)', fontSize: '10px' }}>not configured</span>
            </div>
          </div>
        </div>
      </div>

      <div className="two-col" style={{ marginTop: '14px' }}>
        <div className="card">
          <div className="sec-lbl">Active models</div>
          <div className="lat-tbl">
            <div className="lt-row"><span className="lt-s">PII detection</span><span className="lt-v" style={{ fontSize: '10px' }}>shashidharbabu/deberta-pii-guardrails</span></div>
            <div className="lt-row"><span className="lt-s">Jailbreak</span><span className="lt-v" style={{ fontSize: '10px' }}>shashidharbabu/roberta-jailbreak-guardrails</span></div>
            <div className="lt-row"><span className="lt-s">Prompt injection</span><span className="lt-v" style={{ fontSize: '10px' }}>shashidharbabu/llama-prompt-guard-guardrails</span></div>
            <div className="lt-row"><span className="lt-s">Embedder</span><span className="lt-v" style={{ fontSize: '10px' }}>nvidia/llama-embed-nemotron-8b</span></div>
            <div className="lt-row"><span className="lt-s">LLM</span><span className="lt-v" style={{ fontSize: '10px' }}>qwen2.5:7b (Ollama)</span></div>
            <div className="lt-row"><span className="lt-s">MAD agents</span><span className="lt-v" style={{ fontSize: '10px' }}>qwen2.5:7b (Ollama)</span></div>
          </div>
        </div>

        <div className="card">
          <div className="sec-lbl">System</div>
          <div className="lat-tbl">
            <div className="lt-row"><span className="lt-s">Database</span><span className="lt-v" style={{ fontSize: '10px' }}>SQLite (app_sessions.db)</span></div>
            <div className="lt-row"><span className="lt-s">Vector store</span><span className="lt-v" style={{ fontSize: '10px' }}>Qdrant Cloud · GCP us-east4</span></div>
            <div className="lt-row"><span className="lt-s">Corpus chunks</span><span className="lt-v" style={{ fontSize: '10px' }}>4,664</span></div>
            <div className="lt-row"><span className="lt-s">Collection</span><span className="lt-v" style={{ fontSize: '10px' }}>ai_governance_chunks_nemotron8b</span></div>
            <div className="lt-row"><span className="lt-s">Embed dim</span><span className="lt-v" style={{ fontSize: '10px' }}>4096 (Nemotron-8B)</span></div>
            <div className="lt-row"><span className="lt-s">Similarity</span><span className="lt-v" style={{ fontSize: '10px' }}>cosine · HNSW</span></div>
          </div>
        </div>
      </div>
    </div>
  )
}
