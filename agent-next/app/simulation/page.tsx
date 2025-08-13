'use client';

import { useState, useEffect } from 'react';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from '@/components/ui/dialog';
import { Loader2, Play, CheckCircle, Clock, AlertCircle } from 'lucide-react';
import { useRouter } from 'next/navigation';
import { getStoredCredentials, getStoredJWT } from '@/lib/auth';
import { getSimulations, addSimulation, updateSimulation } from '@/lib/indexedDB';
import { Textarea } from '@/components/ui/textarea';
import { Label } from '@/components/ui/label';
import { Input } from '@/components/ui/input';
import { ScrollArea } from '@/components/ui/scroll-area';
import { openDatabase } from '@/lib/indexedDB';

interface Simulation {
  id: string;
  alpha_expression: string;
  status: 'queued' | 'simulating' | 'completed' | 'error';
  progress: number;
  result?: {
    fitness: number;
    sharpe: number;
    turnover: number;
    [key: string]: any;
  };
  error?: string;
  created_at: number;
  updated_at: number;
  progress_url?: string;
  attempts?: number;
  location?: string;
}

interface BatchSimulation {
  id: string;
  simulations: Simulation[];
  status: 'queued' | 'simulating' | 'completed' | 'error';
  progress: number;
  completed: number;
  total: number;
  error?: string;
  created_at: number;
  updated_at: number;
}

interface LogEntry {
  timestamp: number;
  type: 'info' | 'success' | 'error' | 'warning';
  message: string;
}

interface PendingSimulation {
  id: string;
  alpha_expression: string;
  location: string;
  attempts: number;
  status: 'pending' | 'complete' | 'error';
  lastCheck?: number;
}

export default function SimulationPage() {
  const router = useRouter();
  const [isAuthenticated, setIsAuthenticated] = useState<boolean | null>(null);
  const [simulations, setSimulations] = useState<Simulation[]>([]);
  const [selectedSimulation, setSelectedSimulation] = useState<Simulation | null>(null);
  const [isDetailsOpen, setIsDetailsOpen] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [batchExpressions, setBatchExpressions] = useState('');
  const [batchSimulations, setBatchSimulations] = useState<BatchSimulation[]>([]);
  const [batchSize, setBatchSize] = useState('5');
  const [batchDelay, setBatchDelay] = useState('1000');
  const [pendingResults, setPendingResults] = useState<Record<string, Simulation>>({});
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [pendingSimulations, setPendingSimulations] = useState<Record<string, PendingSimulation>>({});
  const [maxSimulations, setMaxSimulations] = useState<number>(0);
  const [currentProgress, setCurrentProgress] = useState<Record<string, number>>({});
  const [checkInterval, setCheckInterval] = useState<NodeJS.Timeout | null>(null);
  const [initialSetInterval, setInitialSetInterval] = useState<NodeJS.Timeout | null>(null);
  const [retryAfter, setRetryAfter] = useState<number>(0);
  const [simulatingSimulations, setSimulatingSimulations] = useState<Record<string, Simulation>>({});
  const [waitingQueue, setWaitingQueue] = useState<Simulation[]>([]);
  const [currentPage, setCurrentPage] = useState<{ [key: string]: number }>({
    queued: 1,
    simulating: 1,
    completed: 1
  });
  const [itemsPerPage] = useState(5);
  const [showPromisingOnly, setShowPromisingOnly] = useState(false);

  // Check if user is already authenticated
  useEffect(() => {
    const credentials = getStoredCredentials();
    console.log(credentials);
    if (credentials) {
      setIsAuthenticated(true);
      loadSimulations();
    } else {
      // Redirect to login page if not authenticated
      router.push('/login');
    }
  }, [router]);
  
  const loadSimulations = async () => {
    try {
      const db = await openDatabase();
      const allSimulations = await getSimulations(db);

      console.log('allSimulations', allSimulations);
      setSimulations(allSimulations || []);

      // Check for any simulating simulations
      const simulating = (allSimulations || []).filter(s => 
        s.status === 'simulating'
      );
      const interval = setInterval(() => {
        checkSimulationProgress();
      }, 5000);

      if (simulating.length > 0) {
        const simulatingMap: Record<string, Simulation> = {};
        const progress: Record<string, number> = {};

        simulating.forEach(sim => {
          simulatingMap[sim.id] = sim;
          progress[sim.id] = sim.progress || 0;
        });

        setSimulatingSimulations(simulatingMap);
        setCurrentProgress(progress);
        addLog('info', `Found ${simulating.length} simulating simulations`);

        setInitialSetInterval(interval);

        return () => clearInterval(interval);
      }
      else {
        clearInterval(interval);
        setInitialSetInterval(null);
      }
    } catch (error) {
      console.error('Error loading simulations:', error);
      setSimulations([]);
    }
  };

  const handleViewDetails = (simulation: Simulation) => {
    setSelectedSimulation(simulation);
    setIsDetailsOpen(true);
  };

  const addLog = (type: LogEntry['type'], message: string) => {
    const entry: LogEntry = {
      timestamp: Date.now(),
      type,
      message
    };
    setLogs(prev => [...prev, entry]);
  };

  const submitSimulation = async (simulation: Simulation): Promise<{ status: string; result?: any; message?: string }> => {
    try {
      addLog('info', `Submitting simulation ${simulation.id}: ${simulation.alpha_expression}`);
      
      const response = await fetch('/api/simulations', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          alpha_expression: simulation.alpha_expression,
          jwtToken: getStoredCredentials()?.jwtToken
        })
      });

      console.log('response', response);

      if (response.status === 429) {
        addLog('warning', `Simulation limit exceeded for ${simulation.id}`);
        setMaxSimulations(prev => Math.max(prev, Object.keys(simulatingSimulations).length));
        // Add to waiting queue
        setWaitingQueue(prev => [...prev, simulation]);
        return { status: 'queued', message: 'Added to waiting queue' };
      }


      const data = await response.json();
      const location = response.headers.get('location');
      
      if (location) {
        const updatedSimulation: Simulation = {
          ...simulation,
          status: 'simulating',
          progress_url: data.progress_url,
          location: location,
          attempts: (simulation.attempts || 0) + 1
        };
        
        // Add to simulating simulations
        setSimulatingSimulations(prev => ({
          ...prev,
          [simulation.id]: updatedSimulation
        }));
        setCurrentProgress(prev => ({
          ...prev,
          [simulation.id]: 0
        }));
        
        addLog('success', `Simulation ${simulation.id} added to simulating queue`);
      }

      return {
        status: 'success',
        result: {
          id: simulation.id,
          progress_url: data.progress_url,
          location: location
        }
      };
    } catch (error) {
      addLog('error', `Error submitting simulation: ${error instanceof Error ? error.message : 'Unknown error'}`);
      return { status: 'error', message: error instanceof Error ? error.message : 'Unknown error' };
    }
  };

  const checkSimulationProgress = async () => {
    const now = Date.now();
    const completed: string[] = [];
    const retryQueue: string[] = [];
    const db = await openDatabase();

    console.log('checking simulation progress');

    if(Object.keys(simulatingSimulations).length === 0) {
      clearInterval(initialSetInterval as NodeJS.Timeout);
      setInitialSetInterval(null);
      return;
    }

    for (const [id, sim] of Object.entries(simulatingSimulations)) {
      if (sim.status !== 'simulating') continue;

      // Check if we need to wait due to rate limiting
      if (retryAfter > now) continue;

      try {
        addLog('info', `Checking progress for simulation ${id}`);
      
        const jwtToken = getStoredCredentials()?.jwtToken;
        
        const response = await fetch(`/api/simulations/progress?url=${encodeURIComponent(sim.progress_url!)}`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            jwtToken
          })
        });
        
        // Handle rate limits
        if (response.status === 429) {
          const retryAfterHeader = response.headers.get('Retry-After');
          if (retryAfterHeader) {
            const waitTime = parseInt(retryAfterHeader) * 1000;
            setRetryAfter(now + waitTime);
            addLog('warning', `Rate limit hit, will retry in ${retryAfterHeader}s`);
            continue;
          }
        }

        if (!response.ok) {
          const errorText = await response.text();
          let error;
          try {
            error = JSON.parse(errorText);
          } catch (e) {
            error = { error: errorText };
          }
          throw new Error(error.error || `HTTP error! status: ${response.status}`);
        }

        // Handle SSE response
        const reader = response.body?.getReader();
        if (!reader) {
          throw new Error('No response body');
        }

        const decoder = new TextDecoder();
        let data = '';
        
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          
          data += decoder.decode(value, { stream: true });
          
          // Process complete SSE messages
          const messages = data.split('\n\n');
          data = messages.pop() || ''; // Keep incomplete message
          
          for (const message of messages) {
            if (!message.startsWith('data: ')) continue;
            
            try {
              const jsonData = JSON.parse(message.slice(6)); // Remove 'data: ' prefix
          
              if (jsonData.status === 'complete') {
                await updateSimulation(db, {
                  id,
                  status: 'completed',
                  progress: 100,
                  result: jsonData.result
                });
                completed.push(id);
                addLog('success', `Simulation ${id} completed successfully`);
              } else if (jsonData.status === 'error') {
                // Check if it's a simulation limit error
                if (jsonData.detail?.includes('SIMULATION_LIMIT_EXCEEDED')) {
                  addLog('warning', `Simulation limit exceeded for ${id}, moving to waiting queue`);
                  // Move to waiting queue
                  setWaitingQueue(prev => [...prev, sim]);
                  retryQueue.push(id);
                } else {
                  // Mark as failed for any other error
                  await updateSimulation(db, {
                    id,
                    status: 'error',
                    error: jsonData.error || jsonData.detail || 'Unknown error'
                  });
                  completed.push(id);
                  addLog('error', `Simulation ${id} failed: ${jsonData.error || jsonData.detail || 'Unknown error'}`);
                }
              } else if (jsonData.status === 'in_progress') {
                setCurrentProgress(prev => ({
                  ...prev,
                  [id]: jsonData.progress
                }));
                // Update the simulation progress in the database
                await updateSimulation(db, {
                  id,
                  progress: jsonData.progress
                });
              }
            } catch (error) {
              addLog('error', `Error parsing SSE message for simulation ${id}: ${error instanceof Error ? error.message : 'Unknown error'}`);
            }
          }
        }
      } catch (error) {
        // Mark as failed for any error during progress check
        await updateSimulation(db, {
          id,
          status: 'error',
          error: error instanceof Error ? error.message : 'Unknown error'
        });
        completed.push(id);
        addLog('error', `Error checking progress for simulation ${id}: ${error instanceof Error ? error.message : 'Unknown error'}`);
      }
    }

    // Remove completed simulations
    setSimulatingSimulations(prev => {
      const updated = { ...prev };
      completed.forEach(id => delete updated[id]);
      return updated;
    });

    setCurrentProgress(prev => {
      const updated = { ...prev };
      completed.forEach(id => delete updated[id]);
      return updated;
    });

    // Requeue failed simulations
    retryQueue.forEach(id => {
      setSimulatingSimulations(prev => {
        const updated = { ...prev };
        delete updated[id];
        return updated;
      });
      setCurrentProgress(prev => {
        const updated = { ...prev };
        delete updated[id];
        return updated;
      });
      // Add back to queued state
      updateSimulation(db, {
        id,
        status: 'queued',
        progress: 0,
        result: undefined,
        error: undefined,
        progress_url: undefined,
        updated_at: now
      });
      addLog('info', `Simulation ${id} requeued due to simulation limit`);
    });

    // If we have space in simulating simulations and waiting queue is not empty
    if (Object.keys(simulatingSimulations).length < maxSimulations && waitingQueue.length > 0) {
      const nextSimulation = waitingQueue[0];
      setWaitingQueue(prev => prev.slice(1));
      
      // Add to simulating simulations
      setSimulatingSimulations(prev => ({
        ...prev,
        [nextSimulation.id]: nextSimulation
      }));
      setCurrentProgress(prev => ({
        ...prev,
        [nextSimulation.id]: 0
      }));
      
      addLog('info', `Starting waiting simulation ${nextSimulation.id}`);
    }

    // Refresh simulations list if any completed
    if (completed.length > 0 || retryQueue.length > 0) {
      loadSimulations();
    }
  };

  // Start progress checking when there are simulating simulations
  useEffect(() => {
    console.log('simulatingSimulations', simulatingSimulations);
    if (Object.keys(simulatingSimulations).length > 0 && !checkInterval) {
      const interval = setInterval(checkSimulationProgress, 5000); // Check every 5 seconds
      setCheckInterval(interval);
      addLog('info', 'Started progress checking interval');
    } else if (Object.keys(simulatingSimulations).length === 0 && checkInterval) {
      clearInterval(checkInterval);
      setCheckInterval(null);
      addLog('info', 'Stopped progress checking interval');
    }

    return () => {
      if (checkInterval) {
        clearInterval(checkInterval);
        setCheckInterval(null);
    }
  };
  }, [simulatingSimulations]);

  const handleBatchSubmit = async () => {
    const queuedSimulations = simulations.filter(s => s.status === 'queued');
    if (queuedSimulations.length === 0) {
      addLog('warning', 'No queued simulations to process');
      return;
    }

    addLog('info', `Starting batch of ${queuedSimulations.length} simulations`);
    const batchId = `batch-${Date.now()}`;
    const batchSimulation: BatchSimulation = {
      id: batchId,
      simulations: queuedSimulations,
      status: 'queued',
      progress: 0,
      completed: 0,
      total: queuedSimulations.length,
      created_at: Date.now(),
      updated_at: Date.now()
    };

    setBatchSimulations(prev => [...prev, batchSimulation]);

    // Process batch with rate limiting
    const size = Math.min(parseInt(batchSize), queuedSimulations.length);
    const delay = parseInt(batchDelay);
    const db = await openDatabase();
    
    for (let i = 0; i < queuedSimulations.length; i += size) {
      const batch = queuedSimulations.slice(i, i + size);
      
      // Update batch status
      setBatchSimulations(prev => prev.map(bs => 
        bs.id === batchId 
          ? { ...bs, status: 'simulating', progress: (i / queuedSimulations.length) * 100 }
          : bs
      ));

      // Process each simulation in the current batch
      for (const simulation of batch) {
        try {
          // Check if we need to wait due to rate limiting
          if (retryAfter > Date.now()) {
            const waitTime = retryAfter - Date.now();
            addLog('warning', `Waiting ${Math.ceil(waitTime / 1000)}s due to rate limit`);
            await new Promise(resolve => setTimeout(resolve, waitTime));
          }

          const result = await submitSimulation(simulation);
          
          if (result.status === 'success' && result.result) {
            const updatedSimulation: Simulation = {
              ...simulation,
              status: 'simulating',
              progress_url: result.result.progress_url,
              location: result.result.location, // Store the string URL
              attempts: (simulation.attempts || 0) + 1
            };

            await updateSimulation(db, updatedSimulation);
            
            // Add to simulating simulations
            setSimulatingSimulations(prev => ({
              ...prev,
              [simulation.id]: updatedSimulation
            }));
            setCurrentProgress(prev => ({
              ...prev,
              [simulation.id]: 0
            }));
            
            // Update batch progress
            setBatchSimulations(prev => prev.map(bs => 
              bs.id === batchId 
                ? { ...bs, completed: bs.completed + 1 }
                : bs
            ));

            addLog('success', `Simulation ${simulation.id} submitted successfully`);
          } else {
            await updateSimulation(db, {
              id: simulation.id,
              status: 'error',
              error: result.message || 'Failed to submit simulation'
            });
            addLog('error', `Failed to submit simulation ${simulation.id}: ${result.message}`);
          }

          // Add delay between API calls
          await new Promise(resolve => setTimeout(resolve, delay));
        } catch (error) {
          console.error('Error submitting simulation:', error);
          await updateSimulation(db, {
            id: simulation.id,
            status: 'error',
            error: error instanceof Error ? error.message : 'Failed to submit simulation'
          });
          addLog('error', `Error submitting simulation ${simulation.id}: ${error instanceof Error ? error.message : 'Unknown error'}`);
        }
      }
    }
  };

  const handleRequeue = async (simulation: Simulation) => {
    try {
      const db = await openDatabase();
      await updateSimulation(db, {
        id: simulation.id,
        status: 'queued',
        progress: 0,
        result: undefined,
        error: undefined,
        progress_url: undefined,
        updated_at: Date.now()
      });
      addLog('info', `Simulation ${simulation.id} moved back to queued state`);
      loadSimulations(); // Refresh the list
    } catch (error) {
      addLog('error', `Failed to requeue simulation ${simulation.id}: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  };

  const isPromisingAlpha = (simulation: Simulation) => {
    if (!simulation.result) return false;
    return (
      simulation.result.sharpe > 1.25 &&
      simulation.result.turnover > 0.01 &&
      simulation.result.turnover < 0.7 &&
      simulation.result.fitness >= 1.0
    );
  };

  const getFilteredSimulations = (status: string) => {
    let filtered = simulations.filter(s => s.status === status);
    if (showPromisingOnly && status === 'completed') {
      filtered = filtered.filter(isPromisingAlpha);
    }
    return filtered;
  };

  const getPaginatedSimulations = (status: string) => {
    const filtered = getFilteredSimulations(status);
    const startIndex = (currentPage[status] - 1) * itemsPerPage;
    return filtered.slice(startIndex, startIndex + itemsPerPage);
  };

  const totalPages = (status: string) => {
    const filtered = getFilteredSimulations(status);
    return Math.ceil(filtered.length / itemsPerPage);
    };

  const handlePageChange = (status: string, page: number) => {
    setCurrentPage(prev => ({
      ...prev,
      [status]: page
    }));
  };

  if (isAuthenticated === null) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <Loader2 className="h-8 w-8 animate-spin text-primary" />
      </div>
    );
  }

  if (!isAuthenticated) {
    return null;
  }

  return (
    // Dark mode
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-blue-900 to-gray-900 text-white">

    <div className="container mx-auto py-4 space-y-4">
      <div className="flex flex-col space-y-1">
        <h1 className="text-2xl font-bold">Simulation Management</h1>
        <p className="text-sm text-muted-foreground">
          Monitor and manage your alpha simulations
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        {/* Batch Simulation Section */}
        <div className="space-y-2">
          <Card>
            <CardHeader className="p-4">
              <CardTitle className="text-lg">Batch Simulation</CardTitle>
              <CardDescription className="text-xs">Run queued simulations in batches</CardDescription>
            </CardHeader>
            <CardContent className="p-4 space-y-2">
              <div className="grid grid-cols-2 gap-2">
                <div className="space-y-1">
                  <Label htmlFor="batchSize" className="text-xs">Batch Size</Label>
                  <Input
                    id="batchSize"
                    type="number"
                    value={batchSize}
                    onChange={(e) => setBatchSize(e.target.value)}
                    min="1"
                    max="10"
                    className="h-8"
                  />
                </div>
                <div className="space-y-1">
                  <Label htmlFor="batchDelay" className="text-xs">Delay (ms)</Label>
                  <Input
                    id="batchDelay"
                    type="number"
                    value={batchDelay}
                    onChange={(e) => setBatchDelay(e.target.value)}
                    min="500"
                    max="5000"
                    className="h-8"
                  />
                </div>
              </div>
              <Button 
                onClick={handleBatchSubmit}
                disabled={simulations.filter(s => s.status === 'queued').length === 0}
                className="h-8"
              >
                Run Batch
              </Button>
            </CardContent>
          </Card>

          {/* Batch Simulation Progress */}
          {batchSimulations.length > 0 && (
            <div className="space-y-2">
              <h3 className="text-sm font-semibold">Batch Progress</h3>
              {batchSimulations.map(batch => (
                <Card key={batch.id} className="p-2">
                  <CardHeader className="p-2">
                    <CardTitle className="text-sm">Batch {batch.id}</CardTitle>
                    <CardDescription className="text-xs">
                      {batch.status === 'completed' 
                        ? 'Completed' 
                        : `${batch.completed} of ${batch.total} completed`}
                    </CardDescription>
                  </CardHeader>
                  <CardContent className="p-2">
                    <div className="space-y-2">
                      <Progress value={batch.progress} className="h-1" />
                      {batch.error && (
                        <Alert variant="destructive" className="p-2">
                          <AlertCircle className="h-3 w-3" />
                          <AlertTitle className="text-xs">Error</AlertTitle>
                          <AlertDescription className="text-xs">{batch.error}</AlertDescription>
                        </Alert>
                      )}
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          )}
        </div>

        {/* Terminal Section */}
        <Card>
          <CardHeader className="p-4">
            <CardTitle className="text-lg">Simulation Terminal</CardTitle>
            <CardDescription className="text-xs">Real-time simulation status and logs</CardDescription>
          </CardHeader>
          <CardContent className="p-4">
            <div className="bg-black rounded-lg p-2 font-mono text-xs">
              <ScrollArea className="h-[200px]">
                <div className="space-y-0.5">
                  {logs.map((log, index) => (
                    <div key={index} className="flex items-start">
                      <span className="text-gray-500 mr-1 text-xs">
                        {new Date(log.timestamp).toLocaleTimeString()}
                      </span>
                      <span className={`text-xs ${
                        log.type === 'info' ? 'text-blue-400' :
                        log.type === 'success' ? 'text-green-400' :
                        log.type === 'error' ? 'text-red-400' :
                        'text-yellow-400'
                      }`}>
                        {log.message}
                      </span>
                    </div>
                  ))}
                </div>
              </ScrollArea>
            </div>
          </CardContent>
        </Card>

        {/* Simulation Queue Status Card */}
        <Card>
          <CardHeader className="p-4">
            <CardTitle className="text-lg">Simulation Queue</CardTitle>
            <CardDescription className="text-xs">
              {maxSimulations > 0 ? `Maximum concurrent simulations: ${maxSimulations}` : 'No simulation limit set'}
            </CardDescription>
          </CardHeader>
          <CardContent className="p-4">
            <div className="space-y-2">
              {Object.entries(simulatingSimulations).map(([id, sim]) => (
                <div key={id} className="space-y-1">
                  <div className="flex justify-between items-center">
                    <span className="text-xs font-medium">Simulation {id}</span>
                    <span className="text-xs text-muted-foreground">
                      Attempt {sim.attempts}
                    </span>
                  </div>
                  <Progress value={currentProgress[id] || 0} className="h-1" />
                  <pre className="bg-muted p-1 rounded text-xs overflow-x-auto">
                    {sim.alpha_expression}
                  </pre>
                </div>
              ))}
              {Object.keys(simulatingSimulations).length === 0 && (
                <p className="text-xs text-muted-foreground">No simulations in progress</p>
              )}
            </div>
          </CardContent>
        </Card>

        {/* Waiting Queue Card */}
        <Card>
          <CardHeader className="p-4">
            <CardTitle className="text-lg">Waiting Queue</CardTitle>
            <CardDescription className="text-xs">
              Simulations waiting due to simulation limit
            </CardDescription>
          </CardHeader>
          <CardContent className="p-4">
            <div className="space-y-2">
              {waitingQueue.map(sim => (
                <div key={sim.id} className="space-y-1">
                  <div className="flex justify-between items-center">
                    <span className="text-xs font-medium">Simulation {sim.id}</span>
                    <span className="text-xs text-muted-foreground">
                      Attempt {sim.attempts || 1}
                    </span>
                  </div>
                  <pre className="bg-muted p-1 rounded text-xs overflow-x-auto">
                    {sim.alpha_expression}
                  </pre>
                </div>
              ))}
              {waitingQueue.length === 0 && (
                <p className="text-xs text-muted-foreground">No simulations in waiting queue</p>
              )}
            </div>
          </CardContent>
        </Card>

        {/* Simulation Tabs */}
        <div className="lg:col-span-3">
          <Tabs defaultValue="queued" className="space-y-2">
            <div className="flex justify-between items-center">
            <TabsList className="grid grid-cols-3 w-full max-w-md">
              <TabsTrigger value="queued">Queued</TabsTrigger>
              <TabsTrigger value="simulating">Simulating</TabsTrigger>
              <TabsTrigger value="completed">Completed</TabsTrigger>
            </TabsList>
              <div className="flex items-center space-x-2">
                <Label htmlFor="promising-filter" className="text-xs">Show Promising Only</Label>
                <input
                  id="promising-filter"
                  type="checkbox"
                  checked={showPromisingOnly}
                  onChange={(e) => setShowPromisingOnly(e.target.checked)}
                  className="h-4 w-4 rounded border-gray-300 text-primary focus:ring-primary"
                />
              </div>
            </div>

            <TabsContent value="queued" className="space-y-2">
              {totalPages('queued') > 1 && (
                <div className="flex justify-between items-center mb-2">
                  <span className="text-xs text-muted-foreground">
                    Showing {getPaginatedSimulations('queued').length} of {getFilteredSimulations('queued').length} simulations
                  </span>
                  <div className="flex items-center space-x-2">
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => handlePageChange('queued', currentPage.queued - 1)}
                      disabled={currentPage.queued === 1}
                    >
                      Previous
                    </Button>
                    <span className="text-xs flex items-center">
                      Page {currentPage.queued} of {totalPages('queued')}
                    </span>
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => handlePageChange('queued', currentPage.queued + 1)}
                      disabled={currentPage.queued === totalPages('queued')}
                    >
                      Next
                    </Button>
                  </div>
                </div>
              )}
              {getPaginatedSimulations('queued').map(simulation => (
                <Card key={simulation.id} className="hover:shadow-lg transition-shadow p-2">
                  <CardHeader className="p-2">
                    <CardTitle className="text-sm">Simulation {simulation.id}</CardTitle>
                    <CardDescription className="text-xs">Queued for processing</CardDescription>
                    </CardHeader>
                  <CardContent className="p-2">
                    <div className="space-y-1">
                      <pre className="bg-muted p-1 rounded text-xs overflow-x-auto">
                          {simulation.alpha_expression}
                        </pre>
                        <Button 
                          variant="outline" 
                          onClick={() => handleViewDetails(simulation)}
                        className="h-7 text-xs"
                        >
                          View Details
                        </Button>
                      </div>
                    </CardContent>
                  </Card>
                ))}
              {getFilteredSimulations('queued').length === 0 && (
                <p className="text-xs text-muted-foreground text-center py-4">No queued simulations</p>
              )}
            </TabsContent>

            <TabsContent value="simulating" className="space-y-2">
              {totalPages('simulating') > 1 && (
                <div className="flex justify-between items-center mb-2">
                  <span className="text-xs text-muted-foreground">
                    Showing {getPaginatedSimulations('simulating').length} of {getFilteredSimulations('simulating').length} simulations
                  </span>
                  <div className="flex items-center space-x-2">
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => handlePageChange('simulating', currentPage.simulating - 1)}
                      disabled={currentPage.simulating === 1}
                    >
                      Previous
                    </Button>
                    <span className="text-xs flex items-center">
                      Page {currentPage.simulating} of {totalPages('simulating')}
                    </span>
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => handlePageChange('simulating', currentPage.simulating + 1)}
                      disabled={currentPage.simulating === totalPages('simulating')}
                    >
                      Next
                    </Button>
                  </div>
                </div>
              )}
              {getPaginatedSimulations('simulating').map(simulation => (
                <Card key={simulation.id} className="hover:shadow-lg transition-shadow p-2">
                  <CardHeader className="p-2">
                    <CardTitle className="text-sm">Simulation {simulation.id}</CardTitle>
                    <CardDescription className="text-xs">In progress</CardDescription>
                    </CardHeader>
                  <CardContent className="p-2">
                    <div className="space-y-2">
                      <Progress value={simulation.progress} className="h-1" />
                      <pre className="bg-muted p-1 rounded text-xs overflow-x-auto">
                          {simulation.alpha_expression}
                        </pre>
                        <Button 
                          variant="outline" 
                          onClick={() => handleViewDetails(simulation)}
                        className="h-7 text-xs"
                        >
                          View Details
                        </Button>
                      </div>
                    </CardContent>
                  </Card>
                ))}
              {getFilteredSimulations('simulating').length === 0 && (
                <p className="text-xs text-muted-foreground text-center py-4">No simulating simulations</p>
              )}
            </TabsContent>

            <TabsContent value="completed" className="space-y-2">
              {totalPages('completed') > 1 && (
                <div className="flex justify-between items-center mb-2">
                  <span className="text-xs text-muted-foreground">
                    Showing {getPaginatedSimulations('completed').length} of {getFilteredSimulations('completed').length} simulations
                  </span>
                  <div className="flex items-center space-x-2">
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => handlePageChange('completed', currentPage.completed - 1)}
                      disabled={currentPage.completed === 1}
                    >
                      Previous
                    </Button>
                    <span className="text-xs flex items-center">
                      Page {currentPage.completed} of {totalPages('completed')}
                    </span>
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => handlePageChange('completed', currentPage.completed + 1)}
                      disabled={currentPage.completed === totalPages('completed')}
                    >
                      Next
                    </Button>
                  </div>
                </div>
              )}
              {getPaginatedSimulations('completed').map(simulation => (
                <Card key={simulation.id} className="hover:shadow-lg transition-shadow p-2">
                  <CardHeader className="p-2">
                    <CardTitle className="text-sm">Simulation {simulation.id}</CardTitle>
                    <CardDescription className="text-xs">
                        {simulation.status === 'completed' ? 'Completed' : 'Error'}
                      </CardDescription>
                    </CardHeader>
                  <CardContent className="p-2">
                    <div className="space-y-2">
                        {simulation.status === 'completed' && simulation.result && (
                        <div className="grid grid-cols-3 gap-2">
                            <div>
                            <p className="text-xs font-medium">Fitness</p>
                            <p className="text-lg font-bold">{simulation.result.fitness.toFixed(3)}</p>
                            </div>
                            <div>
                            <p className="text-xs font-medium">Sharpe</p>
                            <p className="text-lg font-bold">{simulation.result.sharpe.toFixed(3)}</p>
                            </div>
                            <div>
                            <p className="text-xs font-medium">Turnover</p>
                            <p className="text-lg font-bold">{simulation.result.turnover.toFixed(3)}</p>
                            </div>
                          </div>
                        )}
                        {simulation.status === 'error' && (
                        <Alert variant="destructive" className="p-2">
                          <AlertCircle className="h-3 w-3" />
                          <AlertTitle className="text-xs">Error</AlertTitle>
                          <AlertDescription className="text-xs">{simulation.error}</AlertDescription>
                          </Alert>
                        )}
                      <pre className="bg-muted p-1 rounded text-xs overflow-x-auto">
                          {simulation.alpha_expression}
                        </pre>
                      <div className="flex gap-1">
                        <Button 
                          variant="outline" 
                          onClick={() => handleViewDetails(simulation)}
                          className="h-7 text-xs"
                        >
                          View Details
                        </Button>
                        <Button 
                          variant="outline" 
                          onClick={() => handleRequeue(simulation)}
                          className="h-7 text-xs"
                        >
                          Requeue
                        </Button>
                      </div>
                      </div>
                    </CardContent>
                  </Card>
                ))}
              {getFilteredSimulations('completed').length === 0 && (
                <p className="text-xs text-muted-foreground text-center py-4">
                  {showPromisingOnly ? 'No promising simulations found' : 'No completed simulations'}
                </p>
              )}
            </TabsContent>
          </Tabs>
        </div>
      </div>

      <Dialog open={isDetailsOpen} onOpenChange={setIsDetailsOpen}>
        <DialogContent className="sm:max-w-[600px]">
          <DialogHeader>
            <DialogTitle>Simulation Details</DialogTitle>
            <DialogDescription>
              Detailed information about the simulation
            </DialogDescription>
          </DialogHeader>
          {selectedSimulation && (
            <div className="space-y-4">
              <div>
                <h4 className="text-sm font-medium">Status</h4>
                <p className="text-sm capitalize">{selectedSimulation.status}</p>
              </div>
              <div>
                <h4 className="text-sm font-medium">Alpha Expression</h4>
                <pre className="bg-muted p-2 rounded text-sm overflow-x-auto">
                  {selectedSimulation.alpha_expression}
                </pre>
              </div>
              {selectedSimulation.status === 'simulating' && (
                <div>
                  <h4 className="text-sm font-medium">Progress</h4>
                  <Progress value={selectedSimulation.progress} />
                </div>
              )}
              {selectedSimulation.status === 'completed' && selectedSimulation.result && (
                <div className="space-y-4">
                  <h4 className="text-sm font-medium">Results</h4>
                  <div className="grid grid-cols-3 gap-4">
                    <div>
                      <p className="text-sm font-medium">Fitness</p>
                      <p className="text-2xl font-bold">{selectedSimulation.result.fitness.toFixed(3)}</p>
                    </div>
                    <div>
                      <p className="text-sm font-medium">Sharpe</p>
                      <p className="text-2xl font-bold">{selectedSimulation.result.sharpe.toFixed(3)}</p>
                    </div>
                    <div>
                      <p className="text-sm font-medium">Turnover</p>
                      <p className="text-2xl font-bold">{selectedSimulation.result.turnover.toFixed(3)}</p>
                    </div>
                  </div>
                </div>
              )}
              {selectedSimulation.status === 'error' && (
                <div>
                  <h4 className="text-sm font-medium">Error</h4>
                  <Alert variant="destructive">
                    <AlertCircle className="h-4 w-4" />
                    <AlertTitle>Error</AlertTitle>
                    <AlertDescription>{selectedSimulation.error}</AlertDescription>
                  </Alert>
                </div>
              )}
            </div>
          )}
        </DialogContent>
      </Dialog>
    </div>
    </div>
  );
}