/**
 * FilterBar — search input + pill filter chips + optional children.
 *
 * Usage:
 *   <FilterBar
 *     search={search}
 *     onSearch={setSearch}
 *     placeholder="Search sessions…"
 *     filters={['ALL', 'BLOCK', 'ESCALATE', 'PASS']}
 *     activeFilter={filter}
 *     onFilter={setFilter}
 *     count={12}
 *   />
 */
export default function FilterBar({
  search,
  onSearch,
  placeholder = 'Search…',
  filters = [],
  activeFilter,
  onFilter,
  count,
  children,
}) {
  return (
    <div className="filter-bar">
      {onSearch && (
        <div className="filter-search">
          <svg
            className="filter-search-icon"
            width="13"
            height="13"
            viewBox="0 0 14 14"
            fill="none"
            aria-hidden="true"
          >
            <circle cx="6" cy="6" r="4" stroke="currentColor" strokeWidth="1.3"/>
            <path d="M9 9L12.5 12.5" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round"/>
          </svg>
          <input
            className="s-inp"
            placeholder={placeholder}
            value={search}
            onChange={e => onSearch(e.target.value)}
            aria-label={placeholder}
          />
        </div>
      )}

      {filters.length > 0 && (
        <div className="filter-chips">
          {filters.map(f => (
            <button
              key={f}
              type="button"
              className={`chip${activeFilter === f ? ' active' : ''}`}
              onClick={() => onFilter && onFilter(f)}
            >
              {f}
            </button>
          ))}
        </div>
      )}

      {children}

      {count != null && (
        <span className="filter-count">{count.toLocaleString()} results</span>
      )}
    </div>
  )
}
