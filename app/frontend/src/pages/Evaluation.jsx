import { GATEWAY_METRICS, RAG_METRICS, MAD_METRICS } from '../constants/evalMetrics'
import PageHeader from '../components/PageHeader'
import MetricCard from '../components/MetricCard'
import SectionHeader from '../components/SectionHeader'
import EmptyState from '../components/EmptyState'
import DataTable from '../components/DataTable'

const EMBEDDER_MAX = RAG_METRICS.embedderBenchmark[0].recall1

const CORPUS_TIERS = [
  { lbl: 'T0 · OWASP · security',      pct: 22, color: 'var(--red)' },
  { lbl: 'T1 · GDPR · HIPAA · AI Act', pct: 38, color: 'var(--blue)' },
  { lbl: 'T2 · EDPB · ICO · OCC',      pct: 24, color: 'var(--amber)' },
  { lbl: 'T3 · NIST · ISO 27001',      pct: 16, color: 'var(--teal)' },
]

/* ─── Reusable panel ─────────────────────────────────────────── */
function EvalPanel({ title, subtitle, children, badge }) {
  return (
    <div className="panel">
      <div className="panel-hdr">
        <div style={{ display: 'flex', alignItems: 'center', gap: 10, flex: 1 }}>
          <div className="panel-title">{title}</div>
          {badge}
        </div>
        {subtitle && (
          <div style={{ fontSize: 10, fontFamily: 'var(--font-ui)', color: 'var(--text-muted)' }}>
            {subtitle}
          </div>
        )}
      </div>
      <div className="panel-body">
        {children}
      </div>
    </div>
  )
}

/* ─── Pending eval card ──────────────────────────────────────── */
function PendingCard({ title, description, tag = 'pending' }) {
  return (
    <div className="panel" style={{ borderStyle: 'dashed', borderColor: 'var(--border-md)' }}>
      <EmptyState
        icon={
          <svg viewBox="0 0 22 22" fill="none" width="22" height="22" stroke="currentColor" strokeWidth="1.4">
            <circle cx="11" cy="11" r="8"/>
            <path d="M11 7v4l3 3" strokeLinecap="round"/>
          </svg>
        }
        title={title}
        description={description}
        action={<span className="badge b-gray" style={{ fontSize: '10px' }}>{tag}</span>}
      />
    </div>
  )
}

export default function Evaluation() {
  const gwTableRows = GATEWAY_METRICS.flatMap(gm =>
    gm.metrics.map((m, i) => ({
      id: `${gm.task}-${m.name}`,
      model:  i === 0 ? gm.model.split(' ')[0] : '',
      task:   i === 0 ? gm.task : '',
      metric: m.name,
      value:  m.value,
    }))
  )

  const gwColumns = [
    { key: 'model',  label: 'Model',  render: v => <span className="td-m" style={{ fontWeight: v ? 500 : 400 }}>{v}</span> },
    { key: 'task',   label: 'Task',   render: v => <span className="td-t">{v}</span> },
    { key: 'metric', label: 'Metric', render: v => <span className="td-t">{v}</span> },
    {
      key: 'value',
      label: 'Value',
      render: v => {
        const num = parseFloat(v)
        const cls = !isNaN(num) && num > 95 ? 'td-g' : !isNaN(num) && num > 80 ? 'td-tl' : 'td-g'
        return <span className={cls} style={{ fontFamily: 'var(--font-mono)', fontWeight: 600 }}>{v}</span>
      },
    },
  ]

  const gwWarning = GATEWAY_METRICS.find(g => g.warning)

  return (
    <div>
      <PageHeader
        title="Evaluation"
        sub="Research results · model benchmarks · CS298B final evaluation metrics"
        pills={[
          <span key="rag" className="badge b-health">RAG Pipeline</span>,
          <span key="gw" className="badge b-bank">Gateway Models</span>,
          <span key="mad" className="badge b-gray">MAD System</span>,
        ]}
      />

      {/* ─── RAG Section ───────────────────────────── */}
      <div className="eval-sec">
        <SectionHeader label="RAG Pipeline — Retrieval Quality" />

        <div className="stat-grid" style={{ marginBottom: 18 }}>
          {RAG_METRICS.topline.map(m => (
            <MetricCard
              key={m.label}
              label={m.label}
              value={m.value}
              sub="finetuned Qwen3-4B"
              accent="teal"
              valueColor="var(--teal-hi)"
            />
          ))}
        </div>

        <div className="two-col">
          <EvalPanel title="Embedding Model Benchmark" subtitle="Recall@1">
            <div className="bc-chart">
              {RAG_METRICS.embedderBenchmark.map(m => (
                <div key={m.model} className={`bc-row${m.highlight ? ' winner' : ''}`}>
                  <div className="bc-lbl" style={{ width: 190, fontSize: m.highlight ? 11 : 10 }}>
                    {m.highlight ? '★ ' : ''}{m.model}
                  </div>
                  <div className="bc-track">
                    <div
                      className="bc-fill"
                      style={{
                        width: `${(m.recall1 / EMBEDDER_MAX) * 100}%`,
                        background: m.highlight ? 'var(--teal)' : 'var(--blue)',
                      }}
                    />
                  </div>
                  <div className="bc-val">{m.recall1.toFixed(3)}</div>
                </div>
              ))}
            </div>
          </EvalPanel>

          <EvalPanel
            title="Knowledge Corpus"
            subtitle={`${RAG_METRICS.corpus.totalChunks.toLocaleString()} total chunks`}
          >
            <div className="bc-chart" style={{ marginBottom: 16 }}>
              {CORPUS_TIERS.map(r => (
                <div key={r.lbl} className="bc-row">
                  <div className="bc-lbl">{r.lbl}</div>
                  <div className="bc-track">
                    <div className="bc-fill" style={{ width: `${r.pct}%`, background: r.color }} />
                  </div>
                  <div className="bc-val">{r.pct}%</div>
                </div>
              ))}
            </div>
            <div style={{ fontFamily: 'var(--font-ui)', fontSize: 11, color: 'var(--text-muted)', lineHeight: 1.8 }}>
              {RAG_METRICS.corpus.similarity} similarity · {RAG_METRICS.corpus.index} · {RAG_METRICS.corpus.host}<br />
              2560-dim · Qwen3-4B finetuned embedding<br />
              Collection:{' '}
              <span style={{ color: 'var(--teal-hi)', fontFamily: 'var(--font-mono)', fontSize: 10 }}>
                {RAG_METRICS.corpus.collection}
              </span>
            </div>
          </EvalPanel>
        </div>
      </div>

      {/* ─── Gateway Section ────────────────────────── */}
      <div className="eval-sec">
        <SectionHeader label="Gateway Models — Classification Performance" />

        <div className="two-col">
          <EvalPanel title="In-Distribution Metrics">
            <DataTable columns={gwColumns} rows={gwTableRows} />
          </EvalPanel>

          <EvalPanel
            title="Jailbreak OOD — Per Attack Category"
            badge={<span className="badge b-esc" style={{ fontSize: 9 }}>OOD evaluation</span>}
          >
            <div className="bc-chart" style={{ marginBottom: 12 }}>
              {RAG_METRICS.oodJailbreak.map(r => {
                const color = r.accuracy >= 0.7 ? 'var(--green)'
                  : r.accuracy >= 0.3 ? 'var(--amber)'
                  : 'var(--red)'
                return (
                  <div key={r.category} className="bc-row">
                    <div className="bc-lbl">{r.category}</div>
                    <div className="bc-track">
                      <div
                        className="bc-fill"
                        style={{ width: `${Math.max(r.accuracy * 100, 2)}%`, background: color }}
                      />
                    </div>
                    <div className="bc-val" style={{ color }}>{(r.accuracy * 100).toFixed(0)}%</div>
                  </div>
                )
              })}
            </div>
            <div style={{ fontFamily: 'var(--font-ui)', fontSize: 11, color: 'var(--text-muted)', lineHeight: 1.6 }}>
              Known weaknesses documented in HuggingFace model card. MAD pipeline provides secondary verification for these failure modes.
            </div>
          </EvalPanel>
        </div>

        {gwWarning && (
          <div className="inline-warn" style={{ marginTop: 12 }}>
            <svg width="14" height="14" viewBox="0 0 14 14" fill="none" stroke="currentColor" strokeWidth="1.4" style={{ flexShrink: 0, marginTop: 1 }}>
              <path d="M7 1L1 12h12L7 1z" strokeLinejoin="round"/>
              <path d="M7 5.5v3M7 10h.01" strokeLinecap="round"/>
            </svg>
            {gwWarning.warning}
          </div>
        )}
      </div>

      {/* ─── MAD Section ────────────────────────────── */}
      <div className="eval-sec">
        <SectionHeader label="MAD Evaluation — Synthetic Healthcare Dataset" />
        <div className="two-col">
          <PendingCard
            title="MAD eval dataset — in progress"
            description={`${MAD_METRICS.evalDataset.size}-example ${MAD_METRICS.evalDataset.domain} synthetic set · ${MAD_METRICS.evalDataset.errorTypes.join(' · ')} · Human labels required · Claude API generation`}
            tag="pending · target: after pipeline running"
          />
          <EvalPanel title="Expected Eval Schema">
            <div style={{
              fontFamily: 'var(--font-mono)',
              fontSize: 11,
              color: 'var(--text-sec)',
              lineHeight: 1.9,
              background: 'var(--bg-surface)',
              padding: '12px 14px',
              borderRadius: 'var(--r-md)',
              border: '1px solid var(--border)',
            }}>
              query · domain · llm_answer<br />
              <span style={{ color: 'var(--teal-hi)' }}>error_type:</span> fully_correct | missing_caveat<br />
              <span style={{ paddingLeft: 80 }}>| hallucinated_specific | jurisdiction_blind</span><br />
              <span style={{ color: 'var(--teal-hi)' }}>claims:</span> [{'{'} claim, ground_truth {'}'}]<br />
              <span style={{ color: 'var(--teal-hi)' }}>expected_routing:</span> PASS | ESCALATE | BLOCK
            </div>
          </EvalPanel>
        </div>
      </div>

      {/* ─── Confidence Engine Section ───────────────── */}
      <div className="eval-sec">
        <SectionHeader label="Confidence Engine" />
        <PendingCard
          title="Confidence scoring — pending integration"
          description="Full formula: 0.30 × F_llm + 0.25 × (1 − H_llm) + 0.10 × relevancy + 0.35 × judge_eval_score · Currently using judge-only aggregate (v0.1). DeepEval integration is Phase 2."
          tag="future integration"
        />
      </div>
    </div>
  )
}
