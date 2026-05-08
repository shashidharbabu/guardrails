import { useState } from 'react'
import Badge from './Badge'

const VERDICT_STYLES = {
  SUPPORTED:      { color: 'var(--green-hi)',  bg: 'var(--green-dim)',  border: 'var(--green-mid)' },
  NEEDS_CAVEAT:   { color: 'var(--amber-hi)',  bg: 'var(--amber-dim)',  border: 'var(--amber-mid)' },
  MISSING_CAVEAT: { color: 'var(--amber-hi)',  bg: 'var(--amber-dim)',  border: 'var(--amber-mid)' },
  UNSUPPORTED:    { color: 'var(--red-hi)',    bg: 'var(--red-dim)',    border: 'var(--red-mid)'   },
}

function VerdictPill({ verdict }) {
  const s = VERDICT_STYLES[verdict] || { color: 'var(--text-muted)', bg: 'var(--bg-hover)', border: 'var(--border)' }
  return (
    <span style={{
      display: 'inline-block',
      border: `1px solid ${s.border}`,
      borderRadius: 'var(--r-sm)',
      padding: '2px 8px',
      fontSize: 9,
      fontFamily: 'var(--font-mono)',
      fontWeight: 700,
      letterSpacing: '0.06em',
      textTransform: 'uppercase',
      color: s.color,
      background: s.bg,
    }}>
      {verdict ?? 'UNKNOWN'}
    </span>
  )
}

function ClaimCard({ claim }) {
  return (
    <div style={{
      border: '1px solid var(--border)',
      borderRadius: 'var(--r-md)',
      padding: '10px 12px',
      background: 'var(--bg-base)',
    }}>
      <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', gap: 8, marginBottom: 6 }}>
        <p style={{ fontFamily: 'var(--font-ui)', fontSize: 12, color: 'var(--text-primary)', lineHeight: 1.6, flex: 1 }}>
          {claim.claim_text}
        </p>
        <VerdictPill verdict={claim.verdict} />
      </div>
      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
        <span style={{ fontFamily: 'var(--font-mono)', fontSize: 10, color: 'var(--text-muted)' }}>
          conf {((claim.confidence ?? 0) * 100).toFixed(0)}%
        </span>
        {claim.supporting_chunks?.length > 0 && (
          <span style={{ fontFamily: 'var(--font-mono)', fontSize: 10, color: 'var(--text-muted)' }}>
            · {claim.supporting_chunks.length} chunk{claim.supporting_chunks.length !== 1 ? 's' : ''}
          </span>
        )}
      </div>
    </div>
  )
}

function ChallengeCard({ challenge }) {
  return (
    <div style={{
      border: '1px solid var(--amber-mid)',
      borderRadius: 'var(--r-md)',
      padding: '10px 12px',
      background: 'var(--amber-dim)',
    }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 6 }}>
        <Badge value={challenge.challenge_type} />
        {challenge.suggested_verdict && <VerdictPill verdict={challenge.suggested_verdict} />}
      </div>
      <p style={{ fontFamily: 'var(--font-ui)', fontSize: 12, color: 'var(--amber-hi)', lineHeight: 1.6 }}>
        {challenge.challenge_text}
      </p>
    </div>
  )
}

function AgentSection({ label, color, items, renderItem }) {
  if (!items?.length) return null
  return (
    <div>
      <p style={{
        fontFamily: 'var(--font-ui)',
        fontSize: 10,
        fontWeight: 700,
        letterSpacing: '0.08em',
        textTransform: 'uppercase',
        color,
        marginBottom: 8,
      }}>
        {label}
      </p>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
        {items.map(renderItem)}
      </div>
    </div>
  )
}

function DebateCycle({ cycle, index }) {
  const [open, setOpen] = useState(index === 0)
  return (
    <div style={{ border: '1px solid var(--border-md)', borderRadius: 'var(--r-md)', overflow: 'hidden' }}>
      <button
        onClick={() => setOpen(o => !o)}
        style={{
          width: '100%',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          padding: '10px 14px',
          background: 'var(--bg-card)',
          border: 'none',
          cursor: 'pointer',
          textAlign: 'left',
        }}
      >
        <span style={{ fontFamily: 'var(--font-ui)', fontSize: 12, fontWeight: 600, color: 'var(--text-primary)' }}>
          Cycle {cycle.cycle_index + 1}
        </span>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <span style={{ fontFamily: 'var(--font-mono)', fontSize: 10, color: 'var(--text-muted)' }}>
            conf {((cycle.confidence_signal ?? 0) * 100).toFixed(0)}%
          </span>
          <span style={{ color: 'var(--text-muted)', fontSize: 10 }}>{open ? '▲' : '▼'}</span>
        </div>
      </button>

      {open && (
        <div style={{ padding: 14, display: 'flex', flexDirection: 'column', gap: 14, background: 'var(--bg-surface)' }}>
          <AgentSection
            label="Agent A — Initial Report"
            color="var(--blue-hi)"
            items={cycle.agent_a_report}
            renderItem={(c, i) => <ClaimCard key={c.claim_id ?? i} claim={c} />}
          />
          <AgentSection
            label="Agent B — Challenges"
            color="var(--amber-hi)"
            items={cycle.agent_b_challenges}
            renderItem={(ch, i) => <ChallengeCard key={`${ch.claim_id}-${i}`} challenge={ch} />}
          />
          <AgentSection
            label="Agent A — Revised"
            color="var(--green-hi)"
            items={cycle.agent_a_revised}
            renderItem={(c, i) => <ClaimCard key={`rev-${c.claim_id ?? i}`} claim={c} />}
          />
        </div>
      )}
    </div>
  )
}

function JudgeVerdictRow({ verdict, claim }) {
  const score = verdict?.score
  const scoreLabel = score === 1.0 ? 'SUPPORTED' : score === 0.5 ? 'PARTIAL' : 'NOT_SUPPORTED'
  const scoreColor = score === 1.0 ? 'var(--green-hi)' : score === 0.5 ? 'var(--amber-hi)' : 'var(--red-hi)'
  return (
    <div style={{
      border: '1px solid var(--border)',
      borderRadius: 'var(--r-md)',
      padding: '10px 12px',
      background: 'var(--bg-base)',
    }}>
      <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', gap: 8, marginBottom: 4 }}>
        <p style={{ fontFamily: 'var(--font-ui)', fontSize: 12, color: 'var(--text-primary)', lineHeight: 1.6, flex: 1 }}>
          {verdict?.claim_text ?? claim?.claim_text}
        </p>
        <div style={{ display: 'flex', gap: 4, flexShrink: 0 }}>
          {(verdict?.is_material ?? claim?.is_material) && (
            <span style={{ fontSize: 9, padding: '2px 6px', borderRadius: 'var(--r-sm)', background: 'var(--amber-dim)', color: 'var(--amber-hi)', fontFamily: 'var(--font-mono)', fontWeight: 700 }}>MATERIAL</span>
          )}
          <span style={{ fontSize: 9, padding: '2px 8px', borderRadius: 'var(--r-sm)', background: 'var(--bg-hover)', color: scoreColor, fontFamily: 'var(--font-mono)', fontWeight: 700, textTransform: 'uppercase' }}>
            {scoreLabel}
          </span>
        </div>
      </div>
      <div style={{ fontFamily: 'var(--font-mono)', fontSize: 10, color: 'var(--text-muted)' }}>
        judge score {score != null ? (score * 100).toFixed(0) : '—'}%
        {claim?.confidence != null && ` · agent conf ${(claim.confidence * 100).toFixed(0)}%`}
      </div>
      {verdict?.reasoning && (
        <p style={{ fontFamily: 'var(--font-ui)', fontSize: 11, color: 'var(--text-sec)', marginTop: 6, lineHeight: 1.5 }}>
          {verdict.reasoning.slice(0, 200)}{verdict.reasoning.length > 200 ? '…' : ''}
        </p>
      )}
    </div>
  )
}

export default function MADDebate({ madOutput }) {
  const [showTranscript, setShowTranscript] = useState(false)

  if (!madOutput) {
    return (
      <div style={{
        border: '1px solid var(--border)',
        borderRadius: 'var(--r-lg)',
        padding: 16,
        background: 'var(--bg-card)',
        fontFamily: 'var(--font-ui)',
        fontSize: 12,
        color: 'var(--text-muted)',
      }}>
        No MAD output — query was blocked at gateway.
      </div>
    )
  }

  const {
    debate_cycles = [],
    debate_transcript,
    routing_decision,
    confidence_score,
    aggregate_confidence,
    claims = [],
    judge_verdicts = [],
  } = madOutput

  const displayConfidence = aggregate_confidence ?? confidence_score ?? 0

  // V4MAD returns claims+judge_verdicts; legacy returns debate_cycles
  const isV4 = claims.length > 0

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
      {/* Summary bar */}
      <div className="panel">
        <div className="panel-body" style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
            <Badge value={routing_decision} size="lg" />
            <span style={{ fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--text-sec)' }}>
              confidence {(displayConfidence * 100).toFixed(0)}%
            </span>
          </div>
          <span style={{ fontFamily: 'var(--font-ui)', fontSize: 11, color: 'var(--text-muted)' }}>
            {isV4
              ? `${claims.length} claim${claims.length !== 1 ? 's' : ''} · R0+R1 debate`
              : `${debate_cycles.length} debate cycle${debate_cycles.length !== 1 ? 's' : ''}`}
          </span>
        </div>
      </div>

      {/* V4MAD: claims + judge verdicts */}
      {isV4 ? (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
          {claims.map((claim, i) => {
            const verdict = judge_verdicts.find(v => v.claim_id === claim.claim_id)
            return <JudgeVerdictRow key={claim.claim_id ?? i} verdict={verdict} claim={claim} />
          })}
        </div>
      ) : debate_cycles.length > 0 ? (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
          {debate_cycles.map((cycle, i) => (
            <DebateCycle key={cycle.cycle_index ?? i} cycle={cycle} index={i} />
          ))}
        </div>
      ) : (
        <div style={{
          border: '1px solid var(--border)',
          borderRadius: 'var(--r-lg)',
          padding: 16,
          background: 'var(--bg-card)',
          fontFamily: 'var(--font-ui)',
          fontSize: 12,
          color: 'var(--text-muted)',
        }}>
          No debate cycles recorded.
        </div>
      )}

      {/* Raw transcript (collapsible debug) */}
      {debate_transcript && (
        <div style={{ border: '1px solid var(--border)', borderRadius: 'var(--r-md)', overflow: 'hidden' }}>
          <button
            onClick={() => setShowTranscript(o => !o)}
            style={{
              width: '100%',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              padding: '8px 14px',
              background: 'var(--bg-card)',
              border: 'none',
              cursor: 'pointer',
              fontFamily: 'var(--font-ui)',
              fontSize: 11,
              color: 'var(--text-muted)',
            }}
          >
            <span>Raw transcript (debug)</span>
            <span>{showTranscript ? '▲' : '▼'}</span>
          </button>
          {showTranscript && (
            <pre style={{
              padding: '4px 14px 14px',
              fontFamily: 'var(--font-mono)',
              fontSize: 11,
              color: 'var(--text-sec)',
              whiteSpace: 'pre-wrap',
              lineHeight: 1.7,
              background: 'var(--bg-base)',
            }}>
              {debate_transcript}
            </pre>
          )}
        </div>
      )}
    </div>
  )
}
