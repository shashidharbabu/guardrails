import { createContext, useContext, useEffect, useMemo, useState } from 'react'

const AuthContext = createContext(null)
const STORAGE_KEY = 'guardrails.auth.session'

export function AuthProvider({ children }) {
  const [session, setSession] = useState(null)
  const [ready, setReady] = useState(false)

  useEffect(() => {
    try {
      const stored = window.localStorage.getItem(STORAGE_KEY)
      setSession(stored ? JSON.parse(stored) : null)
    } catch {
      setSession(null)
    } finally {
      setReady(true)
    }
  }, [])

  const value = useMemo(() => ({
    ready,
    user: session,
    isAuthenticated: Boolean(session?.access_token),
    token: session?.access_token || null,

    async signIn(username, password) {
      const res = await fetch('/api/auth/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username, password }),
      })
      if (!res.ok) {
        const err = await res.json().catch(() => ({}))
        throw new Error(err.detail || 'Login failed')
      }
      const data = await res.json()
      window.localStorage.setItem(STORAGE_KEY, JSON.stringify(data))
      setSession(data)
      return data
    },

    signOut() {
      window.localStorage.removeItem(STORAGE_KEY)
      setSession(null)
    },
  }), [ready, session])

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
