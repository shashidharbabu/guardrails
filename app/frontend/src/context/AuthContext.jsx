import { createContext, useContext, useEffect, useMemo, useState } from 'react'

const AuthContext = createContext(null)
const STORAGE_KEY = 'guardrails.auth.session'

const DEFAULT_USER = {
  name: 'Avery Stone',
  email: 'avery@northstar.ai',
  org: 'Northstar AI Governance',
  role: 'Safety Operations Lead',
}

export function AuthProvider({ children }) {
  const [user, setUser] = useState(null)
  const [ready, setReady] = useState(false)

  useEffect(() => {
    try {
      const stored = window.localStorage.getItem(STORAGE_KEY)
      setUser(stored ? JSON.parse(stored) : null)
    } catch {
      setUser(null)
    } finally {
      setReady(true)
    }
  }, [])

  const value = useMemo(() => ({
    ready,
    user,
    isAuthenticated: Boolean(user),
    signIn(profile = {}) {
      const nextUser = { ...DEFAULT_USER, ...profile }
      window.localStorage.setItem(STORAGE_KEY, JSON.stringify(nextUser))
      setUser(nextUser)
      return nextUser
    },
    signOut() {
      window.localStorage.removeItem(STORAGE_KEY)
      setUser(null)
    },
  }), [ready, user])

  return (
    <AuthContext.Provider value={value}>
      {children}
    </AuthContext.Provider>
  )
}

export function useAuth() {
  const ctx = useContext(AuthContext)
  if (!ctx) throw new Error('useAuth must be used inside AuthProvider')
  return ctx
}
