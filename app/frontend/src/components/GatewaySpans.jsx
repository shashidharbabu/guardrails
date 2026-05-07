import ScoreBar from './ScoreBar'
import Badge from './Badge'

export default function GatewaySpans({ payload, decision }) {
  if (!payload) return null

  const {
    threat_score    = 0,
    pii_score       = 0,
    jailbreak_score = 0,
    injection_score = 0,
    entities        = [],
  } = payload

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
      {/* Decision + scores */}
      <div className="panel">
        <div className="panel-hdr">
          <div className="panel-title">Gateway Decision</div>
          <Badge value={decision} />
        </div>
        <div className="panel-body">
          <div className="score-block">
            <ScoreBar label="Threat (overall)"  value={threat_score}    colorMode="semantic" />
            <ScoreBar label="PII"               value={pii_score}       colorMode="fixed" color="var(--blue)" />
            <ScoreBar label="Jailbreak"         value={jailbreak_score} colorMode="fixed" color="var(--pink)" />
            <ScoreBar label="Prompt injection"  value={injection_score} colorMode="fixed" color="var(--amber)" />
          </div>
        </div>
      </div>

      {/* PII entities */}
      {entities.length > 0 && (
        <div className="panel">
          <div className="panel-hdr">
            <div className="panel-title">Detected Entities</div>
            <span style={{ fontFamily: 'var(--font-mono)', fontSize: 10, color: 'var(--text-muted)' }}>
              {entities.length} found
            </span>
          </div>
          <div className="panel-body" style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
            {entities.map((ent, i) => (
              <div
                key={i}
                style={{
                  background: 'var(--amber-dim)',
                  border: '1px solid var(--amber-mid)',
                  borderRadius: 'var(--r-md)',
                  padding: '5px 10px',
                  display: 'flex',
                  alignItems: 'center',
                  gap: 6,
                }}
              >
                <span style={{
                  fontFamily: 'var(--font-mono)',
                  fontSize: 9,
                  fontWeight: 700,
                  color: 'var(--amber-hi)',
                  letterSpacing: '0.06em',
                  textTransform: 'uppercase',
                }}>
                  {ent.type}
                </span>
                <span style={{ fontFamily: 'var(--font-ui)', fontSize: 11, color: 'var(--amber-hi)', opacity: 0.8 }}>
                  {ent.text}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
