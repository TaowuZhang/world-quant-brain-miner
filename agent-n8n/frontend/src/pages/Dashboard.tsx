import { Component, createSignal, onMount, onCleanup } from 'solid-js'
import { io, Socket } from 'socket.io-client'
import axios from 'axios'
import { 
  Activity, 
  Users, 
  Workflow, 
  TrendingUp,
  Zap,
  AlertCircle,
  CheckCircle
} from 'lucide-solid'

interface Agent {
  id: string
  name: string
  type: string
  status: 'online' | 'offline' | 'error'
  last_seen: string
  metadata: Record<string, any>
}

interface SystemMetrics {
  total_agents: number
  active_agents: number
  total_workflows: number
  active_workflows: number
  alpha_generated: number
  system_health: 'healthy' | 'warning' | 'error'
}

const Dashboard: Component = () => {
  const [agents, setAgents] = createSignal<Agent[]>([])
  const [metrics, setMetrics] = createSignal<SystemMetrics>({
    total_agents: 0,
    active_agents: 0,
    total_workflows: 0,
    active_workflows: 0,
    alpha_generated: 0,
    system_health: 'healthy'
  })
  const [socket, setSocket] = createSignal<Socket | null>(null)
  const [isGenerating, setIsGenerating] = createSignal(false)

  // Fetch initial data
  const fetchData = async () => {
    try {
      const [agentsRes, metricsRes] = await Promise.all([
        axios.get('/api/agents'),
        axios.get('/api/metrics')
      ])
      setAgents(agentsRes.data)
      setMetrics(metricsRes.data)
    } catch (error) {
      console.error('Error fetching data:', error)
    }
  }

  // Connect to WebSocket
  const connectWebSocket = () => {
    const socketInstance = io('/ws', {
      transports: ['websocket']
    })

    socketInstance.on('connect', () => {
      console.log('Connected to Agent Hub WebSocket')
    })

    socketInstance.on('agent_update', (data) => {
      setAgents(prev => prev.map(agent => 
        agent.id === data.agent_id 
          ? { ...agent, ...data.updates }
          : agent
      ))
    })

    socketInstance.on('metrics_update', (data) => {
      setMetrics(prev => ({ ...prev, ...data }))
    })

    socketInstance.on('disconnect', () => {
      console.log('Disconnected from Agent Hub WebSocket')
    })

    setSocket(socketInstance)
  }

  // Generate alpha
  const generateAlpha = async () => {
    setIsGenerating(true)
    try {
      await axios.post('/api/alpha/generate', {
        parameters: {
          max_ideas: 5,
          test_alphas: true
        }
      })
    } catch (error) {
      console.error('Error generating alpha:', error)
    } finally {
      setIsGenerating(false)
    }
  }

  onMount(() => {
    fetchData()
    connectWebSocket()
  })

  onCleanup(() => {
    const currentSocket = socket()
    if (currentSocket) {
      currentSocket.disconnect()
    }
  })

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'online': return 'text-green-600'
      case 'offline': return 'text-gray-600'
      case 'error': return 'text-red-600'
      default: return 'text-gray-600'
    }
  }

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'online': return <CheckCircle class="w-4 h-4 text-green-600" />
      case 'offline': return <AlertCircle class="w-4 h-4 text-gray-600" />
      case 'error': return <AlertCircle class="w-4 h-4 text-red-600" />
      default: return <AlertCircle class="w-4 h-4 text-gray-600" />
    }
  }

  return (
    <div class="space-y-6">
      {/* Header */}
      <div class="flex justify-between items-center">
        <div>
          <h1 class="text-2xl font-bold text-gray-900">Dashboard</h1>
          <p class="text-gray-600">Monitor your agent network and system health</p>
        </div>
        <button
          onClick={generateAlpha}
          disabled={isGenerating()}
          class="btn-primary flex items-center space-x-2 disabled:opacity-50"
        >
          <Zap class="w-4 h-4" />
          <span>{isGenerating() ? 'Generating...' : 'Generate Alpha'}</span>
        </button>
      </div>

      {/* Metrics Grid */}
      <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <div class="metric-card">
          <div class="flex items-center">
            <Users class="w-8 h-8 text-white/80" />
            <div class="ml-4">
              <p class="text-white/80 text-sm font-medium">Total Agents</p>
              <p class="text-white text-2xl font-bold">{metrics().total_agents}</p>
            </div>
          </div>
        </div>

        <div class="metric-card">
          <div class="flex items-center">
            <Activity class="w-8 h-8 text-white/80" />
            <div class="ml-4">
              <p class="text-white/80 text-sm font-medium">Active Agents</p>
              <p class="text-white text-2xl font-bold">{metrics().active_agents}</p>
            </div>
          </div>
        </div>

        <div class="metric-card">
          <div class="flex items-center">
            <Workflow class="w-8 h-8 text-white/80" />
            <div class="ml-4">
              <p class="text-white/80 text-sm font-medium">Workflows</p>
              <p class="text-white text-2xl font-bold">{metrics().total_workflows}</p>
            </div>
          </div>
        </div>

        <div class="metric-card">
          <div class="flex items-center">
            <TrendingUp class="w-8 h-8 text-white/80" />
            <div class="ml-4">
              <p class="text-white/80 text-sm font-medium">Alphas Generated</p>
              <p class="text-white text-2xl font-bold">{metrics().alpha_generated}</p>
            </div>
          </div>
        </div>
      </div>

      {/* Agents List */}
      <div class="card">
        <h2 class="text-lg font-semibold text-gray-900 mb-4">Active Agents</h2>
        <div class="overflow-x-auto">
          <table class="min-w-full divide-y divide-gray-200">
            <thead class="bg-gray-50">
              <tr>
                <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Agent
                </th>
                <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Type
                </th>
                <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Status
                </th>
                <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Last Seen
                </th>
              </tr>
            </thead>
            <tbody class="bg-white divide-y divide-gray-200">
              {agents().map((agent) => (
                <tr>
                  <td class="px-6 py-4 whitespace-nowrap">
                    <div class="flex items-center">
                      <div class="flex-shrink-0 h-8 w-8">
                        <div class="h-8 w-8 rounded-full bg-gray-300 flex items-center justify-center">
                          <span class="text-sm font-medium text-gray-700">
                            {agent.name.charAt(0).toUpperCase()}
                          </span>
                        </div>
                      </div>
                      <div class="ml-4">
                        <div class="text-sm font-medium text-gray-900">{agent.name}</div>
                        <div class="text-sm text-gray-500">{agent.id}</div>
                      </div>
                    </div>
                  </td>
                  <td class="px-6 py-4 whitespace-nowrap">
                    <span class="inline-flex px-2 py-1 text-xs font-semibold rounded-full bg-blue-100 text-blue-800">
                      {agent.type}
                    </span>
                  </td>
                  <td class="px-6 py-4 whitespace-nowrap">
                    <div class="flex items-center">
                      {getStatusIcon(agent.status)}
                      <span class={`ml-2 text-sm font-medium ${getStatusColor(agent.status)}`}>
                        {agent.status}
                      </span>
                    </div>
                  </td>
                  <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                    {new Date(agent.last_seen).toLocaleString()}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}

export default Dashboard 