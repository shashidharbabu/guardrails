import { Link } from 'react-router-dom'

export default function AppFooter() {
  return (
    <footer className="app-footer" aria-label="Application footer">
      <div>
        <strong>SpartanGuard</strong>
        <span>Enterprise AI safety operations</span>
      </div>
      <nav aria-label="Footer links">
        <Link to="/system-health">System health</Link>
        <Link to="/audit">Audit logs</Link>
        <Link to="/settings">Settings</Link>
      </nav>
    </footer>
  )
}
