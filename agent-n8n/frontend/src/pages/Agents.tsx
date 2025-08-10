import { Component, createSignal, onMount } from 'solid-js'
import axios from 'axios'
import { 
  Plus, 
  Trash2, 
  Settings,
  Activity,
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

const Agents: Component = () => {
  const [agents, setAgents] = createSignal<Agent[]>([])
  const [loading, setLoading] = createSignal(true)

  const fetchAgents = async () => {
    try {
      const response = await axios.get('/api/agents')
      setAgents(response.data)
    } catch (error) {
      console.error('Error fetching agents:', error)
    } finally {
      setLoading(false)
    }
  }

  const deleteAgent = async (agentId: string) => {
    try {
      await axios.delete(`/api/agents/${agentId}`)
      setAgents(prev => prev.filter(agent => agent.id !== agentId))
    } catch (error) {
      console.error('Error deleting agent:', error)
    }
  }

  onMount(() => {
    fetchAgents()
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
          <h1 class="text-2xl font-bold text-gray-900">Agents</h1>
          <p class="text-gray-600">Manage your agent network</p>
        </div>
        <button class="btn-primary flex items-center space-x-2">
          <Plus class="w-4 h-4" />
          <span>Add Agent</span>
        </button>
      </div>

      {/* Agents Grid */}
      {loading() ? (
        <div class="flex justify-center items-center h-64">
          <div class="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
        </div>
      ) : (
        <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {agents().map((agent) => (
            <div class="card">
              <div class="flex justify-between items-start mb-4">
                <div class="flex items-center">
                  <div class="h-10 w-10 rounded-full bg-gray-300 flex items-center justify-center">
                    <span class="text-sm font-medium text-gray-700">
                      {agent.name.charAt(0).toUpperCase()}
                    </span>
                  </div>
                  <div class="ml-3">
                    <h3 class="text-lg font-medium text-gray-900">{agent.name}</h3>
                    <p class="text-sm text-gray-500">{agent.type}</p>
                  </div>
                </div>
                <div class="flex space-x-2">
                  <button class="p-1 text-gray-400 hover:text-gray-600">
                    <Settings class="w-4 h-4" />
                  </button>
                  <button 
                    onClick={() => deleteAgent(agent.id)}
                    class="p-1 text-red-400 hover:text-red-600"
                  >
                    <Trash2 class="w-4 h-4" />
                  </button>
                </div>
              </div>

              <div class="space-y-3">
                <div class="flex items-center justify-between">
                  <span class="text-sm text-gray-500">Status</span>
                  <div class="flex items-center">
                    {getStatusIcon(agent.status)}
                    <span class={`ml-1 text-sm font-medium ${getStatusColor(agent.status)}`}>
                      {agent.status}
                    </span>
                  </div>
                </div>

                <div class="flex items-center justify-between">
                  <span class="text-sm text-gray-500">Last Seen</span>
                  <span class="text-sm text-gray-900">
                    {new Date(agent.last_seen).toLocaleString()}
                  </span>
                </div>

                <div class="flex items-center justify-between">
                  <span class="text-sm text-gray-500">ID</span>
                  <span class="text-sm text-gray-900 font-mono">
                    {agent.id.substring(0, 8)}...
                  </span>
                </div>
              </div>

              <div class="mt-4 pt-4 border-t border-gray-200">
                <button class="w-full btn-primary">
                  <Activity class="w-4 h-4 mr-2" />
                  View Details
                </button>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

export default Agents 