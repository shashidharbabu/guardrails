/**
 * DataTable — enterprise table with sticky header, hover rows,
 * sort affordances, and skeleton fallback.
 */
import LoadingSkeleton from './LoadingSkeleton'
import EmptyState from './EmptyState'
import { NoDataIcon } from './EmptyState'

export default function DataTable({
  columns = [],
  rows = [],
  loading = false,
  empty,
  onRowClick,
  sortKey,
  sortDir,
  onSort,
  className = '',
}) {
  if (loading) {
    return <LoadingSkeleton type="table" rows={5} cols={columns.length || 4} />
  }

  return (
    <div className={`mt-wrap ${className}`}>
      <table className="mt">
        <thead>
          <tr>
            {columns.map(col => (
              <th
                key={col.key}
                style={{
                  width: col.width,
                  cursor: onSort && col.sortable !== false ? 'pointer' : 'default',
                  userSelect: 'none',
                }}
                onClick={() => onSort && col.sortable !== false && onSort(col.key)}
              >
                <span style={{ display: 'inline-flex', alignItems: 'center', gap: 5 }}>
                  {col.label}
                  {onSort && col.sortable !== false && (
                    <SortIcon active={sortKey === col.key} dir={sortDir} />
                  )}
                </span>
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.length === 0 ? (
            <tr>
              <td colSpan={columns.length} style={{ padding: 0, background: 'var(--bg-card)' }}>
                {empty || (
                  <EmptyState
                    icon={<NoDataIcon />}
                    title="No data available"
                    description="Nothing to show for the current filters."
                  />
                )}
              </td>
            </tr>
          ) : (
            rows.map((row, ri) => (
              <tr
                key={row.id ?? row.key ?? ri}
                onClick={onRowClick ? () => onRowClick(row) : undefined}
                style={onRowClick ? { cursor: 'pointer' } : {}}
              >
                {columns.map(col => (
                  <td key={col.key} className={col.tdClass || ''}>
                    {col.render
                      ? col.render(row[col.key], row)
                      : row[col.key] ?? (
                          <span style={{ color: 'var(--text-muted)' }}>—</span>
                        )}
                  </td>
                ))}
              </tr>
            ))
          )}
        </tbody>
      </table>
    </div>
  )
}

function SortIcon({ active, dir }) {
  return (
    <svg
      width="8"
      height="10"
      viewBox="0 0 8 10"
      fill="none"
      style={{ opacity: active ? 1 : 0.3, flexShrink: 0 }}
    >
      <path
        d="M4 1v8M1 4l3-3 3 3"
        stroke={active && dir === 'asc' ? 'var(--blue-hi)' : 'currentColor'}
        strokeWidth="1.2"
        strokeLinecap="round"
      />
      <path
        d="M1 6l3 3 3-3"
        stroke={active && dir === 'desc' ? 'var(--blue-hi)' : 'currentColor'}
        strokeWidth="1.2"
        strokeLinecap="round"
      />
    </svg>
  )
}
