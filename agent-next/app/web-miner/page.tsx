'use client';

import { useState, useEffect } from 'react';
import { FileUploader } from '../../components/web-miner/FileUploader';
import DataFieldSelector from '../../components/DataFieldSelector';
import OperatorSelector from '../../components/OperatorSelector';
import AlphaGenerator from '../../components/AlphaGenerator';
import { authenticateWorldQuant, getStoredCredentials, clearStoredCredentials } from '../../lib/auth';
import { FloatingDock } from '../../components/ui/floating-dock';
import { sharedNavItems } from '../../components/ui/shared-navigation';
import { useRouter } from 'next/navigation';
import { Button } from '../../components/ui/button';
import { Check, X } from 'lucide-react';
import { toast } from 'sonner';

interface Field {
  id: string;
  description: string;
  type: string;
}

export default function WebMinerPage() {
  const [selectedFields, setSelectedFields] = useState<Field[]>([]);
  const [selectedOperators, setSelectedOperators] = useState<string[]>([]);
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [username, setUsername] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [allFields, setAllFields] = useState<Field[]>([]);
  const [allOperators, setAllOperators] = useState<string[]>([]);
  const [dataset, setDataset] = useState('fundamental6');
  const [selectedCategory, setSelectedCategory] = useState('all');
  const [offset, setOffset] = useState('0');
  const router = useRouter();

  // Check if user is already authenticated
  useEffect(() => {
    const credentials = getStoredCredentials();
    if (credentials) {
      setIsAuthenticated(true);
      setUsername(credentials.username);
    } else {
      // Redirect to login page if not authenticated
      router.push('/login');
    }
  }, [router]);

  // Fetch all available fields and operators on component mount
  useEffect(() => {
    const fetchAllFieldsAndOperators = async () => {
      try {
        // Get the JWT token
        const jwtToken = getStoredCredentials()?.jwtToken;
        console.log('JWT Token:', jwtToken);
        
        if (!jwtToken) {
          return;
        }
        
        // Fetch operators
        const operatorsResponse = await fetch('/api/operators', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            jwtToken,
          }),
        });
        
        if (operatorsResponse.ok) {
          const operatorsData = await operatorsResponse.json();
          const operatorNames = operatorsData.map((op: any) => op.name);
          setAllOperators(operatorNames);
        }
        
        // Fetch data fields with current dataset, category, and offset
        const fieldsResponse = await fetch('/api/data-fields', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            jwtToken,
            dataset,
            limit: '20',
            instrumentType: 'EQUITY',
            region: 'USA',
            universe: 'TOP3000',
            delay: '1',
            offset,
            category: selectedCategory !== 'all' ? selectedCategory : undefined,
          }),
        });
        
        console.log('Fields Response:', fieldsResponse);

        if (fieldsResponse.ok) {
          const fieldsData = await fieldsResponse.json();
          // It should form a list of objects with the following properties: id, description, type.
          const fieldsList = fieldsData.results.map((field: any) => ({
            id: field.id,
            description: field.description,
            type: field.type,
          }));
          setAllFields(fieldsList);
        }
      } catch (error) {
        console.error('Error fetching fields and operators:', error);
      }
    };

    fetchAllFieldsAndOperators();
  }, [dataset, selectedCategory, offset]);

  const handleLogin = async (username: string, password: string) => {
    try {
      await authenticateWorldQuant({ username, password });
      setIsAuthenticated(true);
      setUsername(username);
      setError(null);
    } catch (error) {
      console.error('Authentication error:', error);
      setError(error instanceof Error ? error.message : 'Authentication failed');
    }
  };

  const handleLogout = () => {
    clearStoredCredentials();
    setIsAuthenticated(false);
    setUsername(null);
    router.push('/login');
  };

  const handleFileUploaded = (file: File) => {
    setUploadedFile(file);
  };

  const handleFieldsSelected = (fieldIds: string[]) => {
    // Convert field IDs to Field objects by looking them up in allFields
    const newFields = fieldIds.map(id => {
      const field = allFields.find(f => f.id === id);
      if (!field) {
        console.error(`Field with ID ${id} not found in allFields`);
        return { id, description: '', type: '' };
      }
      return field;
    });
    setSelectedFields(newFields);
  };

  // Handle select all fields
  const handleSelectAllFields = () => {
    setSelectedFields(allFields);
  };

  // Handle deselect all fields
  const handleDeselectAllFields = () => {
    setSelectedFields([]);
  };

  // Handle select all operators
  const handleSelectAllOperators = () => {
    setSelectedOperators(allOperators);
  };

  // Handle deselect all operators
  const handleDeselectAllOperators = () => {
    setSelectedOperators([]);
  };

  // Add handlers for dataset, category, and page changes
  const handleDatasetChange = (newDataset: string) => {
    setDataset(newDataset);
    // Reset offset when dataset changes
    setOffset('0');
  };

  const handleCategoryChange = (newCategory: string) => {
    setSelectedCategory(newCategory);
    // Reset offset when category changes
    setOffset('0');
  };

  const handlePageChange = (newPage: string) => {
    setOffset(newPage);
  };

  const handleUseDiscoveredFields = (discoveredFields: any[]) => {
    // Convert discovered fields to the format expected by the data field selector
    const convertedFields = discoveredFields.map(field => ({
      id: field.id,
      description: field.description,
      type: field.category
    }));
    
    setSelectedFields(convertedFields);
    toast.success(`Switched to ${convertedFields.length} discovered data fields`);
  };

  // Show loading state while checking authentication
  if (!isAuthenticated) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-gray-900 via-blue-900 to-gray-900 text-white flex items-center justify-center p-4">
        <div className="w-full max-w-md">
          <div className="backdrop-blur-md bg-white/10 p-6 rounded-xl border border-white/20">
            <div className="flex flex-col items-center justify-center">
              <div className="w-12 h-12 border-4 border-blue-500 border-t-transparent rounded-full animate-spin mb-4"></div>
              <p className="text-blue-200">Redirecting to login page...</p>
            </div>
          </div>
        </div>
        <FloatingDock items={sharedNavItems} />
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-blue-900 to-gray-900 text-white p-4 md:p-8">
      <div className="max-w-7xl mx-auto space-y-8">
        {/* Header */}
        <div className="backdrop-blur-md bg-white/10 p-6 rounded-xl border border-white/20">
          <div className="flex justify-between items-center">
            <div>
              <h1 className="text-3xl font-bold bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent">
                Web Crawler
              </h1>
              <p className="text-blue-200 mt-2">
                Upload research papers and generate alpha ideas using WorldQuant data
              </p>
            </div>
            
            <div className="flex items-center gap-4">
              <div className="text-right">
                <p className="text-sm text-blue-200">Logged in as</p>
                <p className="font-medium">{username}</p>
              </div>
              <button
                onClick={handleLogout}
                className="p-2 rounded-lg bg-white/10 hover:bg-white/20 transition-colors"
                title="Logout"
              >
                <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
                  <path fillRule="evenodd" d="M3 3a1 1 0 00-1 1v12a1 1 0 001 1h12a1 1 0 001-1V7.414l-5-5H3zm7 8a1 1 0 10-2 0v3a1 1 0 102 0v-3z" clipRule="evenodd" />
                </svg>
              </button>
            </div>
          </div>
        </div>

        {/* Main Content */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Left Column */}
          <div className="space-y-8">
            <FileUploader onFileUploaded={handleFileUploaded} />
            
            <div className="backdrop-blur-md bg-white/10 p-6 rounded-xl border border-white/20">
              <h2 className="text-xl font-semibold mb-4">Data Fields</h2>
              <div className="mb-4 flex space-x-2">
                <Button 
                  variant="outline" 
                  size="sm" 
                  onClick={handleSelectAllFields}
                  className="flex items-center text-black"
                >
                  <Check className="h-4 w-4 mr-1" />
                  Select All
                </Button>
                <Button 
                  variant="outline" 
                  size="sm" 
                  onClick={handleDeselectAllFields}
                  className="flex items-center text-black"
                >
                  <X className="h-4 w-4 mr-1" />
                  Deselect All
                </Button>
              </div>
              <DataFieldSelector 
                onFieldsSelected={handleFieldsSelected} 
                selectedFields={selectedFields.map(f => f.id)}
                onDatasetChange={handleDatasetChange}
                onCategoryChange={handleCategoryChange}
                onPageChange={handlePageChange}
              />
            </div>
            
            <div className="backdrop-blur-md bg-white/10 p-6 rounded-xl border border-white/20">
              <h2 className="text-xl font-semibold mb-4">Operators</h2>
              <div className="mb-4 flex space-x-2">
                <Button 
                  variant="outline" 
                  size="sm" 
                  onClick={handleSelectAllOperators}
                  className="flex items-center text-black"
                >
                  <Check className="h-4 w-4 mr-1" />
                  Select All
                </Button>
                <Button 
                  variant="outline" 
                  size="sm" 
                  onClick={handleDeselectAllOperators}
                  className="flex items-center text-black"
                >
                  <X className="h-4 w-4 mr-1" />
                  Deselect All
                </Button>
              </div>
              <OperatorSelector 
                onOperatorsSelected={setSelectedOperators} 
                selectedOperators={selectedOperators}
              />
            </div>
          </div>
          
          {/* Right Column */}
          <div>
            <AlphaGenerator
              selectedFields={selectedFields}
              selectedOperators={selectedOperators}
              pdfFile={uploadedFile}
              onUseDiscoveredFields={handleUseDiscoveredFields}
            />
          </div>
        </div>
      </div>
      
      {/* Floating Navigation Dock */}
      <FloatingDock items={sharedNavItems} />
    </div>
  );
} 