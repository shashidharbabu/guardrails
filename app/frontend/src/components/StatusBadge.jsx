/**
 * StatusBadge — single source of truth for all status, decision, MAD,
 * claim, and operational state badges across the guardrails UI.
 *
 * Usage:
 *   <StatusBadge type="decision" value="BLOCK" />
 *   <StatusBadge type="mad" value="HUMAN_REVIEW" />
 *   <StatusBadge type="claim" value="SUPPORTED" />
 *   <StatusBadge type="state" value="online" />
 *   <StatusBadge cls="b-health" label="Gateway Online" dot="var(--teal)" />
 */

const DECISION_MAP = {
  BLOCK:    { cls: 'b-block',  label: 'BLOCK' },
  ESCALATE: { cls: 'b-esc',    label: 'ESCALATE' },
  PASS:     { cls: 'b-pass',   label: 'PASS' },
}

const MAD_MAP = {
  DELIVER:      { cls: 'b-deliver',      label: 'DELIVER' },
  RETRY:        { cls: 'b-retry',        label: 'RETRY' },
  HARD_BLOCK:   { cls: 'b-hard-block',   label: 'HARD BLOCK' },
  HUMAN_REVIEW: { cls: 'b-human-review', label: 'HUMAN REVIEW' },
}

const CLAIM_MAP = {
  SUPPORTED:      { cls: 'b-pass',  label: 'SUPPORTED' },
  UNSUPPORTED:    { cls: 'b-block', label: 'UNSUPPORTED' },
  NOT_SUPPORTED:  { cls: 'b-block', label: 'NOT SUPPORTED' },
  NEEDS_CAVEAT:   { cls: 'b-esc',   label: 'NEEDS CAVEAT' },
  MISSING_CAVEAT: { cls: 'b-esc',   label: 'MISSING CAVEAT' },
}

const STATE_MAP = {
  online:          { cls: 'b-health', label: 'Online',      dot: 'var(--teal)' },
  offline:         { cls: 'b-block',  label: 'Offline',     dot: 'var(--red)' },
  unreachable:     { cls: 'b-block',  label: 'Unreachable', dot: 'var(--red)' },
  degraded:        { cls: 'b-esc',    label: 'Degraded',    dot: 'var(--amber)' },
  checking:        { cls: 'b-esc',    label: 'Checking…',   dot: 'var(--amber)' },
  pending:         { cls: 'b-gray',   label: 'Pending' },
  active:          { cls: 'b-health', label: 'Active',      dot: 'var(--teal)' },
  inactive:        { cls: 'b-gray',   label: 'Inactive' },
  loaded:          { cls: 'b-health', label: 'Loaded',      dot: 'var(--green)' },
  'not loaded':    { cls: 'b-gray',   label: 'Not Loaded' },
  draft:           { cls: 'b-gray',   label: 'Draft' },
  'not configured':{ cls: 'b-gray',   label: 'Not Configured' },
  'key required':  { cls: 'b-esc',    label: 'Key Required', dot: 'var(--amber)' },
}

export function decisionCls(value) {
  return (DECISION_MAP[value] || { cls: 'b-gray' }).cls
}
export function madCls(value) {
  return (MAD_MAP[value] || { cls: 'b-gray' }).cls
}
export function claimCls(value) {
  return (CLAIM_MAP[value] || { cls: 'b-gray' }).cls
}

export default function StatusBadge({ type, value, cls, label, dot, style }) {
  let resolvedCls   = 'b-gray'
  let resolvedLabel = value || label || '—'
  let resolvedDot   = dot

  if (type === 'decision' && value) {
    const m = DECISION_MAP[value]
    if (m) { resolvedCls = m.cls; resolvedLabel = label ?? m.label }
  } else if (type === 'mad' && value) {
    const m = MAD_MAP[value]
    if (m) { resolvedCls = m.cls; resolvedLabel = label ?? m.label }
    else   { resolvedLabel = label ?? value }
  } else if (type === 'claim' && value) {
    const m = CLAIM_MAP[value]
    if (m) { resolvedCls = m.cls; resolvedLabel = label ?? m.label }
  } else if (type === 'state' && value) {
    const m = STATE_MAP[value?.toLowerCase()]
    if (m) { resolvedCls = m.cls; resolvedLabel = label ?? m.label; resolvedDot = resolvedDot ?? m.dot }
  } else if (cls) {
    resolvedCls   = cls
    resolvedLabel = label ?? value ?? '—'
  }

  return (
    <span className={`badge ${resolvedCls}`} style={style}>
      {resolvedDot && (
        <span className="badge-dot" style={{ background: resolvedDot }} />
      )}
      {resolvedLabel}
    </span>
  )
}
