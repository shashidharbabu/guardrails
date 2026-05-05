export default function RAGSpans({ evidencePool = [] }) {
  if (!evidencePool.length) {
    return (
      <div className="bg-white rounded-xl border border-gray-200 p-4 text-xs text-gray-400">
        No evidence retrieved.
      </div>
    )
  }

  return (
    <div className="bg-white rounded-xl border border-gray-200 p-4">
      <p className="text-xs tracking-widest uppercase text-gray-400 mb-3">
        Evidence Pool ({evidencePool.length} chunks)
      </p>
      <div className="space-y-3">
        {evidencePool.map((chunk, i) => (
          <div key={chunk.chunk_id ?? i} className="border border-gray-100 rounded-lg p-3">
            <div className="flex items-start justify-between gap-2 mb-1.5">
              <p className="text-xs font-mono text-blue-600 truncate">{chunk.source}</p>
              <span className="shrink-0 text-xs font-mono text-gray-400">
                {chunk.relevance_score != null
                  ? `${(chunk.relevance_score * 100).toFixed(0)}% rel`
                  : ''}
              </span>
            </div>
            <p className="text-xs text-gray-600 leading-relaxed">{chunk.text}</p>
          </div>
        ))}
      </div>
    </div>
  )
}
