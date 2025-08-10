import { Component, createSignal, onMount } from 'solid-js'
import axios from 'axios'
import { 
  Plus, 
  Play, 
  Pause, 
  Trash2, 
  Settings,
  Clock,
  CheckCircle,
  AlertCircle
} from 'lucide-solid'

interface Workflow {
  id: string
  name: string
  description: string
  status: 'active' | 'inactive' | 'error'
  created_at: string
  last_executed: string
  execution_count: number
  metadata: Record<string, any>
}

const Workflows: Component = () => {
  const [workflows, setWorkflows] = createSignal<Workflow[]>([])
  const [loading, setLoading] = createSignal(true)

  const fetchWorkflows = async () => {
    try {
      const response = await axios.get('/api/workflows')
      setWorkflows(response.data)
    } catch (error) {
      console.error('Error fetching workflows:', error)
    } finally {
      setLoading(false)
    }
  }

  const toggleWorkflow = async (workflowId: string, action: 'start' | 'stop') => {
    try {
      await axios.post(`/api/workflows/${workflowId}/${action}`)
      setWorkflows(prev => prev.map(workflow => 
        workflow.id === workflowId 
          ? { ...workflow, status: action === 'start' ? 'active' : 'inactive' }
          : workflow
      ))
    } catch (error) {
      console.error(`Error ${action}ing workflow:`, error)
    }
  }

  const deleteWorkflow = async (workflowId: string) => {
    try {
      await axios.delete(`/api/workflows/${workflowId}`)
      setWorkflows(prev => prev.filter(workflow => workflow.id !== workflowId))
    } catch (error) {
      console.error('Error deleting workflow:', error)
    }
  }

  onMount(() => {
    fetchWorkflows()
  })

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'active': return 'text-green-600'
      case 'inactive': return 'text-gray-600'
      case 'error': return 'text-red-600'
      default: return 'text-gray-600'
    }
  }

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'active': return <CheckCircle class="w-4 h-4 text-green-600" />
      case 'inactive': return <AlertCircle class="w-4 h-4 text-gray-600" />
      case 'error': return <AlertCircle class="w-4 h-4 text-red-600" />
      default: return <AlertCircle class="w-4 h-4 text-gray-600" />
    }
  }

  return (
    <div class="space-y-6">
      {/* Header */}
      <div class="flex justify-between items-center">
        <div>
          <h1 class="text-2xl font-bold text-gray-900">Workflows</h1>
          <p class="text-gray-600">Manage your automation workflows</p>
        </div>
        <button class="btn-primary flex items-center space-x-2">
          <Plus class="w-4 h-4" />
          <span>Create Workflow</span>
        </button>
      </div>

      {/* Workflows Grid */}
      {loading() ? (
        <div class="flex justify-center items-center h-64">
          <div class="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
        </div>
      ) : (
        <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {workflows().map((workflow) => (
            <div class="card">
              <div class="flex justify-between items-start mb-4">
                <div>
                  <h3 class="text-lg font-medium text-gray-900">{workflow.name}</h3>
                  <p class="text-sm text-gray-500 mt-1">{workflow.description}</p>
                </div>
                <div class="flex space-x-2">
                  <button class="p-1 text-gray-400 hover:text-gray-600">
                    <Settings class="w-4 h-4" />
                  </button>
                  <button 
                    onClick={() => deleteWorkflow(workflow.id)}
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
                    {getStatusIcon(workflow.status)}
                    <span class={`ml-1 text-sm font-medium ${getStatusColor(workflow.status)}`}>
                      {workflow.status}
                    </span>
                  </div>
                </div>

                <div class="flex items-center justify-between">
                  <span class="text-sm text-gray-500">Executions</span>
                  <span class="text-sm text-gray-900 font-medium">
                    {workflow.execution_count}
                  </span>
                </div>

                <div class="flex items-center justify-between">
                  <span class="text-sm text-gray-500">Last Executed</span>
                  <span class="text-sm text-gray-900">
                    {workflow.last_executed ? new Date(workflow.last_executed).toLocaleString() : 'Never'}
                  </span>
                </div>

                <div class="flex items-center justify-between">
                  <span class="text-sm text-gray-500">Created</span>
                  <span class="text-sm text-gray-900">
                    {new Date(workflow.created_at).toLocaleDateString()}
                  </span>
                </div>
              </div>

              <div class="mt-4 pt-4 border-t border-gray-200 flex space-x-2">
                {workflow.status === 'active' ? (
                  <button 
                    onClick={() => toggleWorkflow(workflow.id, 'stop')}
                    class="flex-1 bg-gray-100 hover:bg-gray-200 text-gray-700 font-medium py-2 px-4 rounded-lg transition-colors duration-200 flex items-center justify-center"
                  >
                    <Pause class="w-4 h-4 mr-2" />
                    Stop
                  </button>
                ) : (
                  <button 
                    onClick={() => toggleWorkflow(workflow.id, 'start')}
                    class="flex-1 btn-primary flex items-center justify-center"
                  >
                    <Play class="w-4 h-4 mr-2" />
                    Start
                  </button>
                )}
                <button class="bg-blue-100 hover:bg-blue-200 text-blue-700 font-medium py-2 px-4 rounded-lg transition-colors duration-200">
                  <Clock class="w-4 h-4" />
                </button>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

export default Workflows 