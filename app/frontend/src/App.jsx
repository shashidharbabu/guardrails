import { BrowserRouter, Navigate, Route, Routes, useLocation } from 'react-router-dom'
import { AuthProvider, useAuth } from './context/AuthContext'
import Layout from './components/Layout'
import Landing from './pages/public/Landing'
import { RequestAccess, SignIn } from './pages/auth/AuthPages'
import Conversations from './pages/Conversations'
import NewQuery from './pages/NewQuery'
import SessionTrace from './pages/SessionTrace'
import Analytics from './pages/Analytics'
import Evaluation from './pages/Evaluation'
import Feedback from './pages/Feedback'
import Settings from './pages/Settings'
import Gateway from './pages/Gateway'
import SystemHealth from './pages/SystemHealth'
import HumanReview from './pages/HumanReview'
import AuditLogs from './pages/AuditLogs'

function ProtectedRoute() {
  const { ready, isAuthenticated } = useAuth()
  const location = useLocation()

  if (!ready) {
    return (
      <div className="auth-loading">
        <span className="brand-mark" />
        <p>Preparing secure workspace…</p>
      </div>
    )
  }

  if (!isAuthenticated) {
    return <Navigate to="/sign-in" replace state={{ from: location }} />
  }

  return <Layout />
}

export default function App() {
  return (
    <AuthProvider>
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<Landing />} />
          <Route path="/sign-in" element={<SignIn />} />
          <Route path="/request-access" element={<RequestAccess />} />
          <Route element={<ProtectedRoute />}>
            <Route path="/conversations" element={<Conversations />} />
            <Route path="/new" element={<NewQuery />} />
            <Route path="/sessions/:id" element={<SessionTrace />} />
            <Route path="/gateway" element={<Gateway />} />
            <Route path="/analytics" element={<Analytics />} />
            <Route path="/evaluation" element={<Evaluation />} />
            <Route path="/feedback" element={<Feedback />} />
            <Route path="/human-review" element={<HumanReview />} />
            <Route path="/system-health" element={<SystemHealth />} />
            <Route path="/audit" element={<AuditLogs />} />
            <Route path="/settings" element={<Settings />} />
          </Route>
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </BrowserRouter>
    </AuthProvider>
  )
}
