import { useState } from 'react'
import Badge from './Badge'

function VerdictPill({ verdict }) {
  const map = {
    SUPPORTED: 'bg-green-50 text-green-700 border-green-200',
    NEEDS_CAVEAT: 'bg-amber-50 text-amber-700 border-amber-200',
    MISSING_CAVEAT: 'bg-amber-50 text-amber-700 border-amber-200',
    UNSUPPORTED: 'bg-red-50 text-red-700 border-red-200',
  }
  const cls = map[verdict] || 'bg-gray-50 text-gray-600 border-gray-200'
  return (
    <span className={`inline-block border rounded px-1.5 py-0.5 text-xs font-mono ${cls}`}>
      {verdict ?? 'UNKNOWN'}
    </span>
  )
}

function ClaimCard({ claim }) {
  return (
    <div className="border border-gray-100 rounded-lg p-3">
      <div className="flex items-start justify-between gap-2 mb-1">
        <p className="text-xs text-gray-800 leading-relaxed flex-1">{claim.claim_text}</p>
        <VerdictPill verdict={claim.verdict} />
      </div>
      <div className="flex items-center gap-2 mt-1.5">
        <span className="text-xs font-mono text-gray-400">
          conf {((claim.confidence ?? 0) * 100).toFixed(0)}%
        </span>
        {claim.supporting_chunks?.length > 0 && (
          <span className="text-xs font-mono text-gray-400">
            · {claim.supporting_chunks.length} chunk{claim.supporting_chunks.length !== 1 ? 's' : ''}
          </span>
        )}
      </div>
    </div>
  )
}

function ChallengeCard({ challenge }) {
  return (
    <div className="border border-amber-100 bg-amber-50 rounded-lg p-3">
      <div className="flex items-center gap-2 mb-1">
        <Badge value={challenge.challenge_type} />
        {challenge.suggested_verdict && (
          <VerdictPill verdict={challenge.suggested_verdict} />
        )}
      </div>
      <p className="text-xs text-amber-800 leading-relaxed mt-1">{challenge.challenge_text}</p>
    </div>
  )
}

function DebateCycle({ cycle, index }) {
  const [open, setOpen] = useState(index === 0)
  return (
    <div className="border border-gray-200 rounded-xl overflow-hidden">
      <button
        onClick={() => setOpen(o => !o)}
        className="w-full flex items-center justify-between px-4 py-3 bg-gray-50 hover:bg-gray-100 transition-colors text-left"
      >
        <span className="text-xs font-medium text-gray-700">Cycle {cycle.cycle_index + 1}</span>
        <div className="flex items-center gap-2">
          <span className="text-xs font-mono text-gray-400">
            conf {((cycle.confidence_signal ?? 0) * 100).toFixed(0)}%
          </span>
          <span className="text-gray-400 text-xs">{open ? '▲' : '▼'}</span>
        </div>
      </button>

      {open && (
        <div className="p-4 space-y-4 bg-white">
          {/* Agent A initial */}
          {cycle.agent_a_report?.length > 0 && (
            <div>
              <p className="text-xs font-semibold text-blue-700 mb-2">Agent A — Initial Report</p>
              <div className="space-y-2">
                {cycle.agent_a_report.map((c, i) => (
                  <ClaimCard key={c.claim_id ?? i} claim={c} />
                ))}
              </div>
            </div>
          )}

          {/* Agent B challenges */}
          {cycle.agent_b_challenges?.length > 0 && (
            <div>
              <p className="text-xs font-semibold text-amber-700 mb-2">Agent B — Challenges</p>
              <div className="space-y-2">
                {cycle.agent_b_challenges.map((ch, i) => (
                  <ChallengeCard key={`${ch.claim_id}-${i}`} challenge={ch} />
                ))}
              </div>
            </div>
          )}

          {/* Agent A revised */}
          {cycle.agent_a_revised?.length > 0 && (
            <div>
              <p className="text-xs font-semibold text-green-700 mb-2">Agent A — Revised</p>
              <div className="space-y-2">
                {cycle.agent_a_revised.map((c, i) => (
                  <ClaimCard key={`rev-${c.claim_id ?? i}`} claim={c} />
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

export default function MADDebate({ madOutput }) {
  const [showTranscript, setShowTranscript] = useState(false)

  if (!madOutput) {
    return (
      <div className="bg-white rounded-xl border border-gray-200 p-4 text-xs text-gray-400">
        No MAD output (query was blocked at gateway).
      </div>
    )
  }

  const { debate_cycles = [], debate_transcript, routing_decision, confidence_score } = madOutput

  return (
    <div className="space-y-3">
      {/* Summary bar */}
      <div className="bg-white rounded-xl border border-gray-200 p-4 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <Badge value={routing_decision} />
          <span className="text-xs font-mono text-gray-500">
            confidence {((confidence_score ?? 0) * 100).toFixed(0)}%
          </span>
        </div>
        <span className="text-xs text-gray-400">
          {debate_cycles.length} debate cycle{debate_cycles.length !== 1 ? 's' : ''}
        </span>
      </div>

      {/* Debate cycles */}
      {debate_cycles.length > 0 ? (
        <div className="space-y-2">
          {debate_cycles.map((cycle, i) => (
            <DebateCycle key={cycle.cycle_index ?? i} cycle={cycle} index={i} />
          ))}
        </div>
      ) : (
        <div className="bg-white rounded-xl border border-gray-200 p-4 text-xs text-gray-400">
          No debate cycles recorded.
        </div>
      )}

      {/* Raw transcript (collapsible debug) */}
      {debate_transcript && (
        <div className="bg-white rounded-xl border border-gray-200 overflow-hidden">
          <button
            onClick={() => setShowTranscript(o => !o)}
            className="w-full flex items-center justify-between px-4 py-2.5 text-xs text-gray-400 hover:bg-gray-50 transition-colors"
          >
            <span>Raw transcript (debug)</span>
            <span>{showTranscript ? '▲' : '▼'}</span>
          </button>
          {showTranscript && (
            <pre className="px-4 pb-4 text-xs font-mono text-gray-500 whitespace-pre-wrap leading-relaxed">
              {debate_transcript}
            </pre>
          )}
        </div>
      )}
    </div>
  )
}
