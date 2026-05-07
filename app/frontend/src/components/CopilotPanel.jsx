import { useEffect, useRef, useState } from 'react'
import { AnimatePresence, motion } from 'framer-motion'
import { sendCopilotMessage } from '../api/client'

// ---------------------------------------------------------------------------
// Page-aware suggestion chips
// ---------------------------------------------------------------------------
function getSuggestions(pageContext) {
  const { page, session_id } = pageContext || {}

  if (page === 'session_trace' && session_id) {
    return [
      `Why was session ${session_id.slice(0, 8)} blocked or escalated?`,
      `Explain the MAD debate for session ${session_id.slice(0, 8)}`,
      `What PII or threats were detected in session ${session_id.slice(0, 8)}?`,
    ]
  }
  if (page === 'human_review') {
    return [
      'How many sessions are pending human review?',
      'What criteria trigger HUMAN_REVIEW routing?',
      'Walk me through the review workflow',
    ]
  }
  if (page === 'analytics') {
    return [
      "Summarize today's block rate and key metrics",
      'What is the average pipeline latency?',
      'Which threat type is triggering the most blocks?',
    ]
  }
  if (page === 'gateway') {
    return [
      'Explain the three gateway detection models',
      'What does a gateway_score above 0.7 mean?',
      'Show me the most recent gateway events',
    ]
  }
  if (page === 'audit_logs') {
    return [
      'What actions were taken in the last hour?',
      'Show me recent human review decisions',
      'Who has been querying the co-pilot?',
    ]
  }
  return [
    'Show me the latest blocked sessions',
    'What is the current system health?',
    'How does the MAD debate pipeline work?',
  ]
}

// ---------------------------------------------------------------------------
// ToolCallDisclosure — collapsible log of tool calls Claude made
// ---------------------------------------------------------------------------
function ToolCallDisclosure({ toolCalls }) {
  const [open, setOpen] = useState(false)
  if (!toolCalls || toolCalls.length === 0) return null

  return (
    <div className="copilot-tool-log">
      <button
        type="button"
        className="copilot-tool-toggle"
        onClick={() => setOpen(o => !o)}
        aria-expanded={open}
      >
        <svg width="12" height="12" viewBox="0 0 12 12" fill="none" aria-hidden="true"
          style={{ transform: open ? 'rotate(90deg)' : 'rotate(0deg)', transition: 'transform 0.15s' }}>
          <path d="M4 2l4 4-4 4" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round"/>
        </svg>
        {toolCalls.length} tool{toolCalls.length > 1 ? 's' : ''} used
      </button>
      {open && (
        <div className="copilot-tool-list">
          {toolCalls.map((tc, i) => (
            <div key={i} className="copilot-tool-item">
              <div className="copilot-tool-name">
                <span className="copilot-tool-fn">{tc.tool_name}</span>
                {Object.keys(tc.input).length > 0 && (
                  <span className="copilot-tool-args">
                    {Object.entries(tc.input)
                      .map(([k, v]) => `${k}=${JSON.stringify(v)}`)
                      .join(', ')}
                  </span>
                )}
              </div>
              <div className="copilot-tool-out">
                {Array.isArray(tc.output)
                  ? `→ ${tc.output.length} item${tc.output.length !== 1 ? 's' : ''} returned`
                  : typeof tc.output === 'object' && tc.output !== null
                    ? `→ ${Object.keys(tc.output).length} fields`
                    : `→ ${String(tc.output).slice(0, 80)}`}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

// ---------------------------------------------------------------------------
// TypingIndicator
// ---------------------------------------------------------------------------
function TypingIndicator() {
  return (
    <div className="copilot-msg copilot-msg-assistant">
      <div className="copilot-typing">
        <span /><span /><span />
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// MessageBubble
// ---------------------------------------------------------------------------
function MessageBubble({ msg }) {
  const isUser = msg.role === 'user'
  return (
    <div className={`copilot-msg ${isUser ? 'copilot-msg-user' : 'copilot-msg-assistant'}`}>
      {!isUser && (
        <div className="copilot-msg-avatar" aria-hidden="true">
          <svg width="14" height="14" viewBox="0 0 16 16" fill="none">
            <path d="M8 2l1.5 3L13 5.5l-2.5 2.5.6 3.5L8 9.8l-3.1 1.7.6-3.5L3 5.5 6.5 5z"
              stroke="var(--blue)" strokeWidth="1.2" fill="rgba(59,130,246,0.1)"/>
          </svg>
        </div>
      )}
      <div className="copilot-msg-body">
        <div className="copilot-msg-text">{msg.content}</div>
        {!isUser && <ToolCallDisclosure toolCalls={msg.tool_calls} />}
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Main CopilotPanel
// ---------------------------------------------------------------------------
export default function CopilotPanel({ onClose, pageContext }) {
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const messagesEndRef = useRef(null)
  const textareaRef = useRef(null)

  // Auto-scroll to bottom when messages change
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, loading])

  // Close on Escape
  useEffect(() => {
    const handler = e => { if (e.key === 'Escape') onClose() }
    document.addEventListener('keydown', handler)
    return () => document.removeEventListener('keydown', handler)
  }, [onClose])

  // Focus textarea when panel opens
  useEffect(() => {
    setTimeout(() => textareaRef.current?.focus(), 100)
  }, [])

  async function handleSend(overrideText) {
    const text = (overrideText ?? input).trim()
    if (!text || loading) return

    const userMsg = { role: 'user', content: text }
    setMessages(prev => [...prev, userMsg])
    setInput('')
    setLoading(true)
    setError(null)

    // Reset textarea height
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto'
    }

    try {
      const apiHistory = messages.map(m => ({ role: m.role, content: m.content }))
      const result = await sendCopilotMessage(text, apiHistory, pageContext || {})
      setMessages(prev => [
        ...prev,
        {
          role: 'assistant',
          content: result.reply,
          tool_calls: result.tool_calls || [],
        },
      ])
    } catch (e) {
      setError(e.message || 'Request failed')
    } finally {
      setLoading(false)
    }
  }

  function handleKeyDown(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  function handleTextareaInput(e) {
    const el = e.target
    el.style.height = 'auto'
    el.style.height = Math.min(el.scrollHeight, 96) + 'px'
    setInput(el.value)
  }

  const suggestions = getSuggestions(pageContext)

  return (
    <>
      {/* Backdrop */}
      <motion.div
        className="copilot-backdrop"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        transition={{ duration: 0.15 }}
        onClick={onClose}
        aria-hidden="true"
      />

      {/* Panel */}
      <motion.div
        className="copilot-panel"
        role="dialog"
        aria-label="AI Co-pilot"
        aria-modal="true"
        initial={{ x: 420, opacity: 0 }}
        animate={{ x: 0, opacity: 1, transition: { duration: 0.22, ease: [0.16, 1, 0.3, 1] } }}
        exit={{ x: 420, opacity: 0, transition: { duration: 0.14 } }}
      >
        {/* Header */}
        <div className="copilot-header">
          <div className="copilot-header-left">
            <div className="copilot-header-icon" aria-hidden="true">
              <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
                <path d="M8 2l1.5 3L13 5.5l-2.5 2.5.6 3.5L8 9.8l-3.1 1.7.6-3.5L3 5.5 6.5 5z"
                  stroke="var(--blue)" strokeWidth="1.3" fill="rgba(59,130,246,0.12)"/>
              </svg>
            </div>
            <div>
              <div className="copilot-title">Co-pilot</div>
              <div className="copilot-subtitle">AI Safety Assistant</div>
            </div>
          </div>
          <div className="copilot-header-right">
            <span className="copilot-model-badge">Claude</span>
            <button
              type="button"
              className="copilot-close-btn"
              onClick={onClose}
              aria-label="Close co-pilot"
            >
              <svg width="16" height="16" viewBox="0 0 16 16" fill="none" aria-hidden="true">
                <path d="M4 4l8 8M12 4l-8 8" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/>
              </svg>
            </button>
          </div>
        </div>

        {/* Message list */}
        <div className="copilot-messages" role="log" aria-live="polite" aria-label="Conversation">
          {messages.length === 0 && !loading && (
            <div className="copilot-empty">
              <div className="copilot-empty-icon" aria-hidden="true">
                <svg width="32" height="32" viewBox="0 0 32 32" fill="none">
                  <path d="M16 4l3 6 6.5 1L21 15.5l1.2 7L16 19.5l-6.2 3 1.2-7L6.5 11 13 10z"
                    stroke="var(--blue)" strokeWidth="1.5" fill="rgba(59,130,246,0.08)"/>
                </svg>
              </div>
              <p className="copilot-empty-label">Ask anything about your sessions,<br />pipeline decisions, or system health.</p>
              <div className="copilot-chips" role="list">
                {suggestions.map((s, i) => (
                  <button
                    key={i}
                    type="button"
                    className="copilot-chip"
                    role="listitem"
                    onClick={() => handleSend(s)}
                  >
                    {s}
                  </button>
                ))}
              </div>
            </div>
          )}

          {messages.map((msg, i) => (
            <MessageBubble key={i} msg={msg} />
          ))}

          {loading && <TypingIndicator />}

          {error && (
            <div className="copilot-error" role="alert">
              <svg width="14" height="14" viewBox="0 0 14 14" fill="none" aria-hidden="true">
                <circle cx="7" cy="7" r="6" stroke="var(--red)" strokeWidth="1.2"/>
                <path d="M7 4v3.5M7 9.5v.5" stroke="var(--red)" strokeWidth="1.4" strokeLinecap="round"/>
              </svg>
              {error}
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>

        {/* Input row */}
        <div className="copilot-input-row">
          <textarea
            ref={textareaRef}
            className="copilot-textarea"
            placeholder="Ask about sessions, decisions, health…"
            value={input}
            onInput={handleTextareaInput}
            onChange={e => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            rows={1}
            aria-label="Message input"
            disabled={loading}
          />
          <button
            type="button"
            className="copilot-send-btn"
            onClick={() => handleSend()}
            disabled={!input.trim() || loading}
            aria-label="Send message"
          >
            <svg width="16" height="16" viewBox="0 0 16 16" fill="none" aria-hidden="true">
              <path d="M2 8l12-5-5 12-2-4.5L2 8z" stroke="currentColor" strokeWidth="1.4"
                strokeLinecap="round" strokeLinejoin="round" fill="none"/>
            </svg>
          </button>
        </div>
      </motion.div>
    </>
  )
}
