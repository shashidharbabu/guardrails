const STYLES = {
  // Gateway decisions
  PASS:         'bg-green-100 text-green-800 border border-green-200',
  ESCALATE:     'bg-amber-100 text-amber-800 border border-amber-200',
  BLOCK:        'bg-red-100 text-red-800 border border-red-200',
  // MAD routing
  DELIVER:      'bg-green-100 text-green-800 border border-green-200',
  RETRY:        'bg-amber-100 text-amber-800 border border-amber-200',
  HARD_BLOCK:   'bg-red-100 text-red-800 border border-red-200',
  HUMAN_REVIEW: 'bg-orange-100 text-orange-800 border border-orange-200',
  // Misc
  PENDING:      'bg-gray-100 text-gray-600 border border-gray-200',
  // Challenge types
  CHUNK_CURRENCY:      'bg-blue-100 text-blue-700 border border-blue-200',
  JURISDICTION_SCOPE:  'bg-purple-100 text-purple-700 border border-purple-200',
  EXCEPTION_EXISTENCE: 'bg-yellow-100 text-yellow-700 border border-yellow-200',
  GAP_FINDING:         'bg-pink-100 text-pink-700 border border-pink-200',
  // Claim verdicts
  SUPPORTED:     'bg-green-100 text-green-700 border border-green-200',
  PARTIAL:       'bg-amber-100 text-amber-700 border border-amber-200',
  NOT_SUPPORTED: 'bg-red-100 text-red-700 border border-red-200',
  IDK:           'bg-gray-100 text-gray-600 border border-gray-200',
}

export default function Badge({ value, size = 'sm' }) {
  const key = (value || 'PENDING').toString().toUpperCase()
  const style = STYLES[key] || 'bg-gray-100 text-gray-600 border border-gray-200'
  const padding = size === 'lg' ? 'px-3 py-1 text-sm' : 'px-2 py-0.5 text-xs'
  return (
    <span className={`inline-flex items-center rounded-full font-medium font-mono ${padding} ${style}`}>
      {value || 'PENDING'}
    </span>
  )
}
