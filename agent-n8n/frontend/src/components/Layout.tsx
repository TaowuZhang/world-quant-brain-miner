import { Component, JSX } from 'solid-js'
import { A } from '@solidjs/router'
import { 
  Home, 
  Users, 
  Workflow, 
  Activity,
  Settings 
} from 'lucide-solid'

interface LayoutProps {
  children: JSX.Element
}

const Layout: Component<LayoutProps> = (props) => {
  return (
    <div class="min-h-screen bg-gray-50">
      {/* Navigation */}
      <nav class="bg-white shadow-sm border-b border-gray-200">
        <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div class="flex justify-between h-16">
            <div class="flex">
              <div class="flex-shrink-0 flex items-center">
                <h1 class="text-xl font-bold text-gray-900">Agent Network</h1>
              </div>
              <div class="hidden sm:ml-6 sm:flex sm:space-x-8">
                <A 
                  href="/" 
                  class="border-transparent text-gray-500 hover:border-gray-300 hover:text-gray-700 inline-flex items-center px-1 pt-1 border-b-2 text-sm font-medium"
                  activeClass="border-blue-500 text-gray-900"
                >
                  <Home class="w-4 h-4 mr-2" />
                  Dashboard
                </A>
                <A 
                  href="/agents" 
                  class="border-transparent text-gray-500 hover:border-gray-300 hover:text-gray-700 inline-flex items-center px-1 pt-1 border-b-2 text-sm font-medium"
                  activeClass="border-blue-500 text-gray-900"
                >
                  <Users class="w-4 h-4 mr-2" />
                  Agents
                </A>
                <A 
                  href="/workflows" 
                  class="border-transparent text-gray-500 hover:border-gray-300 hover:text-gray-700 inline-flex items-center px-1 pt-1 border-b-2 text-sm font-medium"
                  activeClass="border-blue-500 text-gray-900"
                >
                  <Workflow class="w-4 h-4 mr-2" />
                  Workflows
                </A>
              </div>
            </div>
            <div class="flex items-center">
              <button class="p-2 rounded-md text-gray-400 hover:text-gray-500 hover:bg-gray-100">
                <Activity class="w-5 h-5" />
              </button>
              <button class="ml-3 p-2 rounded-md text-gray-400 hover:text-gray-500 hover:bg-gray-100">
                <Settings class="w-5 h-5" />
              </button>
            </div>
          </div>
        </div>
      </nav>

      {/* Main content */}
      <main class="max-w-7xl mx-auto py-6 sm:px-6 lg:px-8">
        {props.children}
      </main>
    </div>
  )
}

export default Layout 