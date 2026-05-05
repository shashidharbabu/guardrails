import { GATEWAY_METRICS, RAG_METRICS, MAD_METRICS } from '../constants/evalMetrics'

const EMBEDDER_MAX = RAG_METRICS.embedderBenchmark[0].recall1

export default function Evaluation() {
  return (
    <div>
      <div className="ph">
        <div className="pt">Evaluation</div>
        <div className="ps">Research results · model benchmarks · CS298B final metrics</div>
      </div>

      {/* RAG */}
      <div className="eval-sec">
        <div className="sec-lbl">RAG pipeline — retrieval quality</div>
        <div className="stat-grid">
          {RAG_METRICS.topline.map(m => (
            <div key={m.label} className="stat-card">
              <div className="sl">{m.label}</div>
              <div className="sv" style={{ color: 'var(--teal)' }}>{m.value}</div>
              <div className="ss">finetuned Qwen3-4B</div>
            </div>
          ))}
        </div>

        <div className="two-col">
          <div className="card">
            <div style={{ fontSize: '10px', fontFamily: 'var(--font-mono)', color: 'var(--text-muted)', letterSpacing: '0.1em', textTransform: 'uppercase', marginBottom: '12px' }}>
              Embedding model benchmark — Recall@1
            </div>
            <div className="bc-chart">
              {RAG_METRICS.embedderBenchmark.map(m => (
                <div key={m.model} className={`bc-row${m.highlight ? ' winner' : ''}`}>
                  <div className="bc-lbl">{m.highlight ? '★ ' : ''}{m.model}</div>
                  <div className="bc-track">
                    <div className="bc-fill" style={{ width: `${(m.recall1 / EMBEDDER_MAX) * 100}%`, background: m.highlight ? 'var(--teal)' : 'var(--blue)' }}/>
                  </div>
                  <div className="bc-val">{m.recall1.toFixed(3)}</div>
                </div>
              ))}
            </div>
          </div>

          <div className="card">
            <div style={{ fontSize: '10px', fontFamily: 'var(--font-mono)', color: 'var(--text-muted)', letterSpacing: '0.1em', textTransform: 'uppercase', marginBottom: '12px' }}>
              Corpus — {RAG_METRICS.corpus.totalChunks.toLocaleString()} chunks
            </div>
            <div className="bc-chart" style={{ marginBottom: '16px' }}>
              {[
                { lbl: 'T0 · OWASP · security', pct: 22, color: 'var(--red)' },
                { lbl: 'T1 · GDPR · HIPAA · AI Act', pct: 38, color: 'var(--blue)' },
                { lbl: 'T2 · EDPB · ICO · OCC', pct: 24, color: 'var(--amber)' },
                { lbl: 'T3 · NIST · ISO 27001', pct: 16, color: 'var(--teal)' },
              ].map(r => (
                <div key={r.lbl} className="bc-row">
                  <div className="bc-lbl">{r.lbl}</div>
                  <div className="bc-track"><div className="bc-fill" style={{ width: `${r.pct}%`, background: r.color }}/></div>
                  <div className="bc-val">{r.pct}%</div>
                </div>
              ))}
            </div>
            <div style={{ fontSize: '10px', fontFamily: 'var(--font-mono)', color: 'var(--text-muted)', lineHeight: 1.8 }}>
              {RAG_METRICS.corpus.similarity} similarity · {RAG_METRICS.corpus.index} · {RAG_METRICS.corpus.host}<br/>
              2560-dim · Qwen3-4B finetuned embedding<br/>
              Collection: {RAG_METRICS.corpus.collection}
            </div>
          </div>
        </div>
      </div>

      {/* Gateway */}
      <div className="eval-sec">
        <div className="sec-lbl">Gateway models — classification performance</div>
        <div className="two-col">
          <div className="card">
            <div style={{ fontSize: '10px', fontFamily: 'var(--font-mono)', color: 'var(--text-muted)', letterSpacing: '0.1em', textTransform: 'uppercase', marginBottom: '12px' }}>
              In-distribution metrics
            </div>
            <table className="mt">
              <thead>
                <tr><th>Model</th><th>Task</th><th>Metric</th><th>Value</th></tr>
              </thead>
              <tbody>
                {GATEWAY_METRICS.flatMap(gm =>
                  gm.metrics.map((m, i) => (
                    <tr key={`${gm.task}-${m.name}`}>
                      <td className="td-m">{i === 0 ? gm.model.split(' ')[0] : ''}</td>
                      <td className="td-t">{i === 0 ? gm.task : ''}</td>
                      <td className="td-t">{m.name}</td>
                      <td className={typeof m.value === 'string' && m.value.includes('%') && parseFloat(m.value) > 95 ? 'td-g' : parseFloat(m.value) > 80 ? 'td-w' : 'td-g'}>{m.value}</td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>

          <div className="card">
            <div style={{ fontSize: '10px', fontFamily: 'var(--font-mono)', color: 'var(--text-muted)', letterSpacing: '0.1em', textTransform: 'uppercase', marginBottom: '12px' }}>
              Jailbreak OOD — per attack category
            </div>
            <div className="bc-chart" style={{ marginBottom: '12px' }}>
              {RAG_METRICS.oodJailbreak.map(r => {
                const color = r.accuracy >= 0.7 ? 'var(--green)' : r.accuracy >= 0.3 ? 'var(--amber)' : 'var(--red)'
                return (
                  <div key={r.category} className="bc-row">
                    <div className="bc-lbl">{r.category}</div>
                    <div className="bc-track"><div className="bc-fill" style={{ width: `${Math.max(r.accuracy * 100, 2)}%`, background: color }}/></div>
                    <div className="bc-val" style={{ color }}>{(r.accuracy * 100).toFixed(0)}%</div>
                  </div>
                )
              })}
            </div>
            <div style={{ fontSize: '10px', fontFamily: 'var(--font-mono)', color: 'var(--text-muted)', lineHeight: 1.6 }}>
              Known weaknesses documented in HuggingFace model card. MAD pipeline provides secondary verification for these failure modes.
            </div>
          </div>
        </div>

        {GATEWAY_METRICS.find(g => g.warning) && (
          <div style={{ marginTop: '10px', padding: '10px 14px', background: 'var(--amber-dim)', border: '1px solid var(--amber-mid)', borderRadius: '6px', fontFamily: 'var(--font-mono)', fontSize: '11px', color: 'var(--amber)' }}>
            ⚠ {GATEWAY_METRICS.find(g => g.warning).warning}
          </div>
        )}
      </div>

      {/* MAD */}
      <div className="eval-sec">
        <div className="sec-lbl">MAD evaluation — synthetic healthcare dataset</div>
        <div className="two-col">
          <div className="ph-card">
            <div className="ph-t">MAD eval dataset — in progress</div>
            <div className="ph-s">
              {MAD_METRICS.evalDataset.size}-example {MAD_METRICS.evalDataset.domain} synthetic set<br/>
              {MAD_METRICS.evalDataset.errorTypes.join(' · ')}<br/>
              Human labels required · Claude API generation
            </div>
            <span className="ph-b">pending · target: after pipeline running</span>
          </div>
          <div className="card">
            <div style={{ fontSize: '10px', fontFamily: 'var(--font-mono)', color: 'var(--text-muted)', letterSpacing: '0.1em', textTransform: 'uppercase', marginBottom: '10px' }}>
              Expected eval schema
            </div>
            <div style={{ fontFamily: 'var(--font-mono)', fontSize: '11px', color: 'var(--text-sec)', lineHeight: 1.9, background: 'var(--bg-surface)', padding: '12px', borderRadius: '4px', border: '1px solid var(--border)' }}>
              query · domain · llm_answer<br/>
              <span style={{ color: 'var(--teal)' }}>error_type:</span> fully_correct | missing_caveat<br/>
              &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;| hallucinated_specific | jurisdiction_blind<br/>
              <span style={{ color: 'var(--teal)' }}>claims:</span> [{'{'} claim, ground_truth {'}'}]<br/>
              <span style={{ color: 'var(--teal)' }}>expected_routing:</span> PASS | ESCALATE | BLOCK
            </div>
          </div>
        </div>
      </div>

      {/* Confidence */}
      <div className="eval-sec">
        <div className="sec-lbl">Confidence engine</div>
        <div className="ph-card">
          <div className="ph-t">Confidence scoring — pending integration</div>
          <div className="ph-s">
            Full formula: 0.30 × F_llm + 0.25 × (1 − H_llm) + 0.10 × relevancy + 0.35 × judge_eval_score<br/>
            Currently using judge-only aggregate (v0.1). DeepEval integration is Phase 2.
          </div>
          <span className="ph-b">future integration</span>
        </div>
      </div>
    </div>
  )
}
