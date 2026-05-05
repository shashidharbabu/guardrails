import ScoreBar from './ScoreBar'
import Badge from './Badge'

export default function GatewaySpans({ payload, decision }) {
  if (!payload) return null

  const {
    threat_score = 0,
    pii_score = 0,
    jailbreak_score = 0,
    injection_score = 0,
    entities = [],
  } = payload

  return (
    <div className="space-y-4">
      {/* Decision + scores */}
      <div className="bg-white rounded-xl border border-gray-200 p-4">
        <div className="flex items-center justify-between mb-4">
          <p className="text-xs tracking-widest uppercase text-gray-400">Gateway Decision</p>
          <Badge value={decision} />
        </div>
        <div className="space-y-2">
          <ScoreBar label="Threat (overall)" value={threat_score} colorMode="threat" />
          <ScoreBar label="PII" value={pii_score} colorMode="threat" />
          <ScoreBar label="Jailbreak" value={jailbreak_score} colorMode="threat" />
          <ScoreBar label="Prompt inject" value={injection_score} colorMode="threat" />
        </div>
      </div>

      {/* PII entities */}
      {entities.length > 0 && (
        <div className="bg-white rounded-xl border border-gray-200 p-4">
          <p className="text-xs tracking-widest uppercase text-gray-400 mb-3">Detected Entities</p>
          <div className="flex flex-wrap gap-2">
            {entities.map((ent, i) => (
              <div
                key={i}
                className="bg-amber-50 border border-amber-200 rounded-lg px-2.5 py-1.5"
              >
                <span className="text-xs font-mono text-amber-800 font-semibold">{ent.type}</span>
                <span className="text-xs text-amber-600 ml-1.5">{ent.text}</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
