/**
 * ScoreBar — consolidated score bar component used in both SessionTrace
 * sidebar and Gateway Live Test results.
 *
 * colorMode: 'semantic' uses red/amber/green by threshold
 *            'category' uses the fixed threat-type colors (pii/jb/pi/comp)
 *            'fixed' uses the provided `color` CSS value
 *
 * Usage:
 *   <ScoreBar label="PII" value={0.78} colorMode="category" category="pii" />
 *   <ScoreBar label="Jailbreak" value={0.4} colorMode="semantic" />
 *   <ScoreBar label="Overall" value={0.9} colorMode="fixed" color="var(--teal)" />
 */

const CATEGORY_COLORS = {
  pii:    'var(--blue)',
  jb:     'var(--pink)',
  pi:     'var(--amber)',
  comp:   'var(--teal)',
  threat: 'var(--red)',
}

function semanticColor(value) {
  if (value >= 0.7) return 'var(--red)'
  if (value >= 0.4) return 'var(--amber)'
  return 'var(--teal)'
}

export default function ScoreBar({ label, value, colorMode = 'semantic', category, color, style }) {
  const pct = Math.min(100, Math.max(0, (value ?? 0) * 100))

  let barColor = color
  if (!barColor) {
    if (colorMode === 'category' && category) {
      barColor = CATEGORY_COLORS[category] ?? 'var(--teal)'
    } else if (colorMode === 'semantic') {
      barColor = semanticColor(value ?? 0)
    } else {
      barColor = 'var(--teal)'
    }
  }

  const displayVal = value != null ? value.toFixed(2) : '—'

  return (
    <div className="s-row" style={style}>
      <div className="s-meta">
        <span className="s-key">{label}</span>
        <span className="s-val">{displayVal}</span>
      </div>
      <div className="s-track">
        <div
          className="s-fill"
          style={{ width: `${pct}%`, background: barColor }}
        />
      </div>
    </div>
  )
}
