'use client';

import { useState } from 'react';
import { toast } from 'sonner';
import { Play, Search, Database, CheckCircle, AlertCircle } from 'lucide-react';
import { openDatabase, addSimulation } from '@/lib/indexedDB';
import { Button } from '@/components/ui/button';

interface Simulation {
  id: string;
  alpha_expression: string;
  status: 'queued' | 'simulating' | 'completed' | 'failed';
  progress: number;
  result?: {
    sharpe_ratio: number;
    sortino_ratio: number;
    max_drawdown: number;
    annualized_return: number;
  };
  created_at: number;
  updated_at: number;
}

interface Field {
  id: string;
  description: string;
  type: string;
}

interface RelevantDataField {
  id: string;
  score: number;
  name: string;
  category: string;
  description: string;
}

interface AlphaGeneratorProps {
  selectedFields: Field[];
  selectedOperators: string[];
  pdfFile: File | null;
  onUseDiscoveredFields?: (fields: RelevantDataField[]) => void;
}

export default function AlphaGenerator({ selectedFields, selectedOperators, pdfFile, onUseDiscoveredFields }: AlphaGeneratorProps) {
  const [alphaIdeas, setAlphaIdeas] = useState<any[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [relevantDataFields, setRelevantDataFields] = useState<RelevantDataField[]>([]);
  const [isSearchingFields, setIsSearchingFields] = useState(false);
  const [useDiscoveredFields, setUseDiscoveredFields] = useState(false);

  console.log(selectedFields);

  const searchRelevantFields = async () => {
    if (!pdfFile) {
      setError('Please upload a research paper first');
      return;
    }

    setIsSearchingFields(true);
    setError(null);
    
    try {
      const formData = new FormData();
      formData.append('pdf', pdfFile);
      formData.append('topK', '15');
      
      const response = await fetch('/api/search-data-fields', {
        method: 'POST',
        body: formData,
      });
      
      if (!response.ok) {
        throw new Error('Failed to search for relevant data fields');
      }
      
      const data = await response.json();
      setRelevantDataFields(data.results || []);
      toast.success(`Found ${data.results?.length || 0} relevant data fields`);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to search for relevant data fields');
      toast.error('Failed to search for relevant data fields');
    } finally {
      setIsSearchingFields(false);
    }
  };

  const handleUseDiscoveredFields = () => {
    if (relevantDataFields.length > 0) {
      setUseDiscoveredFields(true);
      onUseDiscoveredFields?.(relevantDataFields);
      toast.success(`Using ${relevantDataFields.length} discovered data fields`);
    }
  };

  const handleUseManualFields = () => {
    setUseDiscoveredFields(false);
    toast.success('Switched to manual field selection');
  };

  const generateAlpha = async () => {
    if (!pdfFile) {
      setError('Please upload a research paper first');
      return;
    }

    // Check if we have either manual fields or discovered fields
    const hasManualFields = selectedFields.length > 0;
    const hasDiscoveredFields = relevantDataFields.length > 0 && useDiscoveredFields;
    
    if (!hasManualFields && !hasDiscoveredFields) {
      setError('Please select data fields manually or search for relevant fields from the database');
      return;
    }

    if (selectedOperators.length === 0) {
      setError('Please select at least one operator');
      return;
    }

    setIsLoading(true);
    setError(null);
    
    try {
      const formData = new FormData();
      formData.append('pdf', pdfFile);
      
      // Use discovered fields if available and selected, otherwise use manual fields
      const fieldsToUse = useDiscoveredFields && relevantDataFields.length > 0 
        ? relevantDataFields.map(field => ({
            id: field.id,
            description: field.description,
            type: field.category
          }))
        : selectedFields;
      
      formData.append('fields', JSON.stringify(fieldsToUse));
      formData.append('operators', JSON.stringify(selectedOperators));
      
      const response = await fetch('/api/generate-alpha', {
        method: 'POST',
        body: formData,
      });
      
      if (!response.ok) {
        throw new Error('Failed to generate alpha ideas');
      }
      
      const data = await response.json();
      setAlphaIdeas(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to generate alpha ideas');
    } finally {
      setIsLoading(false);
    }
  };

  const handleQueueSimulation = async (alpha: string) => {
    try {
      const db = await openDatabase();
      const tx = db.transaction('simulations', 'readwrite');
      const store = tx.objectStore('simulations');
      
      const simulation: Simulation = {
        id: Date.now().toString(),
        alpha_expression: alpha,
        status: 'queued',
        progress: 0,
        created_at: Date.now(),
        updated_at: Date.now()
      };
      
      await store.add(simulation);
      toast.success('Alpha queued for simulation');
    } catch (error) {
      console.error('Error queueing simulation:', error);
      toast.error('Failed to queue simulation');
    }
  };

  return (
    <div className="backdrop-blur-md bg-white/10 p-6 rounded-xl border border-white/20">
      <h2 className="text-xl font-semibold mb-4">Generate Alpha Ideas</h2>
      
      <div className="space-y-6">
        {/* Field Selection Status */}
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="text-lg font-medium text-blue-200">Data Fields</h3>
            {relevantDataFields.length > 0 && (
              <div className="flex gap-2">
                <Button
                  onClick={handleUseDiscoveredFields}
                  disabled={useDiscoveredFields}
                  className={`text-xs px-3 py-1 ${
                    useDiscoveredFields 
                      ? 'bg-green-600/30 text-green-200 border-green-500/30' 
                      : 'bg-blue-600/20 hover:bg-blue-600/30 text-blue-200 border-blue-500/30'
                  }`}
                >
                  <Database className="w-3 h-3 mr-1" />
                  Use Discovered ({relevantDataFields.length})
                </Button>
                <Button
                  onClick={handleUseManualFields}
                  disabled={!useDiscoveredFields}
                  className={`text-xs px-3 py-1 ${
                    !useDiscoveredFields 
                      ? 'bg-green-600/30 text-green-200 border-green-500/30' 
                      : 'bg-blue-600/20 hover:bg-blue-600/30 text-blue-200 border-blue-500/30'
                  }`}
                >
                  <CheckCircle className="w-3 h-3 mr-1" />
                  Use Manual ({selectedFields.length})
                </Button>
              </div>
            )}
          </div>
          
          {/* Current Field Selection Display */}
          <div className="backdrop-blur-sm bg-white/5 p-4 rounded-lg border border-white/10">
            {useDiscoveredFields && relevantDataFields.length > 0 ? (
              <div>
                <div className="flex items-center mb-2">
                  <Database className="w-4 h-4 mr-2 text-green-400" />
                  <span className="text-sm font-medium text-green-200">
                    Using {relevantDataFields.length} discovered data fields
                  </span>
                </div>
                <div className="flex flex-wrap gap-2">
                  {relevantDataFields.slice(0, 5).map((field) => (
                    <span
                      key={field.id}
                      className="px-2 py-1 bg-green-900/30 text-green-200 rounded-full text-xs"
                      title={`${field.name} (${field.id})`}
                    >
                      {field.name} ({field.id})
                    </span>
                  ))}
                  {relevantDataFields.length > 5 && (
                    <span className="px-2 py-1 bg-green-900/30 text-green-200 rounded-full text-xs">
                      +{relevantDataFields.length - 5} more
                    </span>
                  )}
                </div>
              </div>
            ) : (
              <div>
                <div className="flex items-center mb-2">
                  <CheckCircle className="w-4 h-4 mr-2 text-blue-400" />
                  <span className="text-sm font-medium text-blue-200">
                    Using {selectedFields.length} manually selected data fields
                  </span>
                </div>
                <div className="flex flex-wrap gap-2">
                  {selectedFields.map((field) => (
                    <span
                      key={field.id}
                      className="px-2 py-1 bg-blue-900/30 text-blue-200 rounded-full text-xs"
                    >
                      {field.id}
                    </span>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
        
        <div className="flex flex-col space-y-4">
          <div>
            <h3 className="text-sm font-medium text-blue-200 mb-2">Selected Operators</h3>
            <div className="flex flex-wrap gap-2">
              {selectedOperators.map((operator) => (
                <span
                  key={operator}
                  className="px-3 py-1 bg-blue-900/30 text-blue-200 rounded-full text-sm"
                >
                  {operator}
                </span>
              ))}
            </div>
          </div>
        </div>

        {/* Relevant Data Fields Section */}
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="text-lg font-medium text-blue-200">Discover Relevant Fields</h3>
            <Button
              onClick={searchRelevantFields}
              disabled={isSearchingFields || !pdfFile}
              className="bg-green-600/20 hover:bg-green-600/30 text-green-200 border border-green-500/30"
            >
              <Search className="w-4 h-4 mr-2" />
              {isSearchingFields ? 'Searching...' : 'Search Database'}
            </Button>
          </div>
          
          {relevantDataFields.length > 0 && (
            <div className="backdrop-blur-sm bg-white/5 p-4 rounded-lg border border-white/10">
              <div className="flex items-center mb-3">
                <Database className="w-4 h-4 mr-2 text-blue-300" />
                <span className="text-sm font-medium text-blue-300">
                  Found {relevantDataFields.length} relevant data fields from Pinecone database
                </span>
              </div>
              <div className="space-y-2 max-h-60 overflow-y-auto">
                {relevantDataFields.map((field, index) => (
                  <div 
                    key={field.id} 
                    className="flex items-start justify-between p-3 bg-black/20 rounded-lg border border-white/5"
                  >
                    <div className="flex-1">
                      <div className="flex items-center gap-2 mb-1">
                        <span className="text-sm font-medium text-blue-200">{field.name}</span>
                        <span className="px-2 py-0.5 bg-blue-900/30 text-blue-300 rounded text-xs">
                          {field.category}
                        </span>
                        <span className="text-xs text-green-400">
                          Score: {field.score?.toFixed(3)}
                        </span>
                      </div>
                      <div className="flex items-center gap-2 mb-1">
                        <span className="text-xs text-yellow-400 font-mono">
                          ID: {field.id}
                        </span>
                      </div>
                      <p className="text-xs text-blue-100/70">{field.description}</p>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        <div className="mb-6">
          <button
            onClick={generateAlpha}
            disabled={isLoading || !pdfFile || 
              (selectedFields.length === 0 && (relevantDataFields.length === 0 || !useDiscoveredFields)) || 
              selectedOperators.length === 0}
            className="w-full py-2 px-4 bg-blue-600 hover:bg-blue-700 text-white font-medium rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {isLoading ? (
              <span className="flex items-center justify-center">
                <svg className="animate-spin -ml-1 mr-2 h-4 w-4 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                </svg>
                Generating...
              </span>
            ) : (
              'Generate Alpha Ideas'
            )}
          </button>
        </div>

        {error && (
          <div className="p-4 bg-red-900/50 text-red-200 rounded">
            {error}
          </div>
        )}

        {alphaIdeas.length > 0 && (
          <div className="space-y-4">
            <h3 className="text-lg font-medium">Generated Alpha Ideas</h3>
            <div className="space-y-3">
              {alphaIdeas.map((idea, index) => (
                <div 
                  key={index} 
                  className="backdrop-blur-sm bg-white/5 p-6 rounded-xl border border-white/10 hover:border-white/20 transition-all duration-300 hover:shadow-lg hover:shadow-blue-500/10"
                >
                  <h3 className="text-xl font-semibold text-blue-200 mb-3">{idea.title}</h3>
                  <p className="text-blue-100/80 mb-4">{idea.description}</p>
                  
                  <div className="space-y-4">
                    <div className="backdrop-blur-sm bg-white/5 p-4 rounded-lg">
                      <h4 className="text-sm font-medium text-blue-300 mb-2">Implementation</h4>
                      <p className="text-blue-100/80">{idea.implementation}</p>
                    </div>
                    
                    <div className="backdrop-blur-sm bg-white/5 p-4 rounded-lg">
                      <h4 className="text-sm font-medium text-blue-300 mb-2">Rationale</h4>
                      <p className="text-blue-100/80">{idea.rationale}</p>
                    </div>
                    
                    <div className="backdrop-blur-sm bg-white/5 p-4 rounded-lg">
                      <h4 className="text-sm font-medium text-blue-300 mb-2">Risks</h4>
                      <p className="text-blue-100/80">{idea.risks}</p>
                    </div>
                    
                    <div className="backdrop-blur-sm bg-white/5 p-4 rounded-lg">
                      <h4 className="text-sm font-medium text-blue-300 mb-2">Alpha Expression</h4>
                      <pre className="bg-black/20 p-4 rounded-lg overflow-x-auto text-sm font-mono text-blue-100/90 border border-white/10">
                        {idea.alpha_expression.split(';').map((line: string, i: number, arr: string[]) => (
                          <span key={i}>
                            {line.trim()}
                            {i < arr.length - 1 ? ';\n' : ''}
                          </span>
                        ))}
                      </pre>
                    </div>
                    
                    <Button 
                      onClick={() => handleQueueSimulation(idea.alpha_expression)}
                      className="w-full bg-blue-500/20 hover:bg-blue-500/30 text-blue-200 border border-blue-500/30"
                    >
                      <Play className="w-4 h-4 mr-2" />
                      Queue for Simulation
                    </Button>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
} 