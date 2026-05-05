// colorMode: 'threat' (red=high), 'confidence' (green=high), 'neutral' (blue always)
function getColor(value, colorMode) {
  if (colorMode === 'neutral') return 'bg-blue-500'
  if (colorMode === 'confidence') {
    if (value >= 0.8) return 'bg-green-500'
    if (value >= 0.4) return 'bg-amber-500'
    return 'bg-red-500'
  }
  // threat mode (default)
  if (value >= 0.7) return 'bg-red-500'
  if (value >= 0.3) return 'bg-amber-500'
  return 'bg-green-500'
}

export default function ScoreBar({ value, label, colorMode = 'threat', showValue = true }) {
  const pct = Math.min(100, Math.max(0, Math.round((value ?? 0) * 100)))
  const color = getColor(value ?? 0, colorMode)

  return (
    <div className="flex items-center gap-2">
      {label && (
        <span className="w-28 shrink-0 text-xs text-gray-500 font-mono truncate">{label}</span>
      )}
      <div className="flex-1 h-1.5 bg-gray-200 rounded-full overflow-hidden">
        <div className={`h-full rounded-full transition-all ${color}`} style={{ width: `${pct}%` }} />
      </div>
      {showValue && (
        <span className="w-10 shrink-0 text-right text-xs font-mono text-gray-600">
          {(value ?? 0).toFixed(3)}
        </span>
      )}
    </div>
  )
}
