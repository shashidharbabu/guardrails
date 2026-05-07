/**
 * LoadingSkeleton — shimmer skeletons for rows, stat grids, cards, and tables.
 *
 * type: 'rows' | 'stat-grid' | 'cards' | 'table'
 */
function Bone({ width = '100%', height = 11, style = {}, rounded = false }) {
  return (
    <div
      className="skeleton"
      style={{
        width,
        height,
        borderRadius: rounded ? 'var(--r-pill)' : 'var(--r-sm)',
        ...style,
      }}
    />
  )
}

function ConvRowSkeleton() {
  return (
    <div className="skeleton-row">
      <div style={{ display: 'flex', justifyContent: 'space-between', gap: 16 }}>
        <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: 8 }}>
          <Bone width="72%" height={13} />
          <div style={{ display: 'flex', gap: 6 }}>
            <Bone width={60} height={18} rounded />
            <Bone width={70} height={18} rounded />
            <Bone width={80} height={12} style={{ marginTop: 3 }} />
          </div>
        </div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 6, alignItems: 'flex-end' }}>
          <div style={{ display: 'flex', gap: 4 }}>
            <Bone width={55} height={20} rounded />
            <Bone width={55} height={20} rounded />
          </div>
          <Bone width={70} height={10} />
        </div>
      </div>
    </div>
  )
}

function StatCardSkeleton() {
  return (
    <div className="stat-card" style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
      <Bone width={100} height={11} />
      <Bone width="55%" height={28} />
      <Bone width="70%" height={10} />
    </div>
  )
}

function CardSkeleton() {
  return (
    <div className="card" style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
      <Bone width="45%" height={12} />
      <Bone width="100%" height={10} />
      <Bone width="85%" height={10} />
      <Bone width="60%" height={10} />
    </div>
  )
}

function TableSkeleton({ rows = 5, cols = 4 }) {
  return (
    <div className="mt-wrap">
      <table className="mt" style={{ width: '100%' }}>
        <thead>
          <tr>
            {Array.from({ length: cols }).map((_, i) => (
              <th key={i}>
                <Bone width={50 + i * 12} height={10} />
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {Array.from({ length: rows }).map((_, ri) => (
            <tr key={ri}>
              {Array.from({ length: cols }).map((_, ci) => (
                <td key={ci}>
                  <Bone width={`${50 + (ci * 13 + ri * 7) % 35}%`} height={10} />
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

export default function LoadingSkeleton({ type = 'rows', count = 4, rows = 5, cols = 4 }) {
  if (type === 'stat-grid') {
    return (
      <div className="stat-grid" style={{ marginBottom: 28 }}>
        {Array.from({ length: count }).map((_, i) => (
          <StatCardSkeleton key={i} />
        ))}
      </div>
    )
  }

  if (type === 'cards') {
    return (
      <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
        {Array.from({ length: count }).map((_, i) => (
          <CardSkeleton key={i} />
        ))}
      </div>
    )
  }

  if (type === 'table') {
    return <TableSkeleton rows={rows} cols={cols} />
  }

  /* default: 'rows' */
  return (
    <div className="conv-list">
      {Array.from({ length: count }).map((_, i) => (
        <ConvRowSkeleton key={i} />
      ))}
    </div>
  )
}
