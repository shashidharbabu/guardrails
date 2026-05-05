import { BrowserRouter, Routes, Route } from 'react-router-dom'
import Layout from './components/Layout'
import Conversations from './pages/Conversations'
import NewQuery from './pages/NewQuery'
import SessionTrace from './pages/SessionTrace'
import Analytics from './pages/Analytics'
import Evaluation from './pages/Evaluation'
import Feedback from './pages/Feedback'
import Settings from './pages/Settings'
import Gateway from './pages/Gateway'

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Layout />}>
          <Route index element={<Conversations />} />
          <Route path="new" element={<NewQuery />} />
          <Route path="sessions/:id" element={<SessionTrace />} />
          <Route path="gateway" element={<Gateway />} />
          <Route path="analytics" element={<Analytics />} />
          <Route path="evaluation" element={<Evaluation />} />
          <Route path="feedback" element={<Feedback />} />
          <Route path="settings" element={<Settings />} />
        </Route>
      </Routes>
    </BrowserRouter>
  )
}
