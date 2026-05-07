/**
 * Badge — legacy badge component used in GatewaySpans and SessionTrace.
 * Maps string values to enterprise dark theme badge classes.
 */
const STYLES = {
  /* Gateway decisions */
  PASS:                { cls: 'b-pass',         label: 'PASS' },
  ESCALATE:            { cls: 'b-esc',          label: 'ESCALATE' },
  BLOCK:               { cls: 'b-block',        label: 'BLOCK' },
  /* MAD routing */
  DELIVER:             { cls: 'b-deliver',      label: 'DELIVER' },
  RETRY:               { cls: 'b-retry',        label: 'RETRY' },
  HARD_BLOCK:          { cls: 'b-hard-block',   label: 'HARD BLOCK' },
  HUMAN_REVIEW:        { cls: 'b-human-review', label: 'HUMAN REVIEW' },
  /* Misc */
  PENDING:             { cls: 'b-gray',         label: 'PENDING' },
  /* Challenge types */
  CHUNK_CURRENCY:      { cls: 'b-bank',         label: 'CHUNK CURRENCY' },
  JURISDICTION_SCOPE:  { cls: 'b-legal',        label: 'JURISDICTION' },
  EXCEPTION_EXISTENCE: { cls: 'b-esc',          label: 'EXCEPTION' },
  GAP_FINDING:         { cls: 'b-gray',         label: 'GAP FINDING' },
  /* Claim verdicts */
  SUPPORTED:           { cls: 'b-pass',         label: 'SUPPORTED' },
  PARTIAL:             { cls: 'b-esc',          label: 'PARTIAL' },
  NOT_SUPPORTED:       { cls: 'b-block',        label: 'NOT SUPPORTED' },
  IDK:                 { cls: 'b-gray',         label: 'IDK' },
}

export default function Badge({ value, size = 'sm' }) {
  const key    = (value || 'PENDING').toString().toUpperCase()
  const config = STYLES[key] || { cls: 'b-gray', label: value || 'PENDING' }
  const sizeStyle = size === 'lg'
    ? { padding: '4px 12px', fontSize: 12 }
    : { padding: '2px 8px',  fontSize: 10 }
  return (
    <span className={`badge ${config.cls}`} style={sizeStyle}>
      {config.label}
    </span>
  )
}
