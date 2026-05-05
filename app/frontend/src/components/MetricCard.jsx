export default function MetricCard({ label, value, sub, accent }) {
  const accentMap = {
    green:  'border-t-green-400',
    amber:  'border-t-amber-400',
    red:    'border-t-red-400',
    blue:   'border-t-blue-400',
    purple: 'border-t-purple-400',
  }
  const border = accentMap[accent] || 'border-t-gray-300'

  return (
    <div className={`bg-white rounded-xl border border-gray-200 border-t-2 ${border} p-4`}>
      <p className="text-xs tracking-widest uppercase text-gray-400 mb-1">{label}</p>
      <p className="text-2xl font-semibold text-gray-800 font-mono">{value ?? '—'}</p>
      {sub && <p className="text-xs text-gray-400 mt-0.5">{sub}</p>}
    </div>
  )
}
