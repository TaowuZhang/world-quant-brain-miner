import { Routes, Route } from '@solidjs/router'
import Dashboard from './pages/Dashboard'
import Agents from './pages/Agents'
import Workflows from './pages/Workflows'
import Layout from './components/Layout'

function App() {
  return (
    <Layout>
      <Routes>
        <Route path="/" component={Dashboard} />
        <Route path="/agents" component={Agents} />
        <Route path="/workflows" component={Workflows} />
      </Routes>
    </Layout>
  )
}

export default App 