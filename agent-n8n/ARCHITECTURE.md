# Agent Network Architecture

## Overview
Ein Docker-basiertes Agentennetzwerk, das verschiedene AI-Agenten, Werkzeuge und Services integriert, um ein vollständiges System für Alpha-Generierung, Workflow-Automatisierung und Frontend-Management zu schaffen.

## System Architecture

### Core Components

#### 1. Agent Network Hub
- **Purpose**: Zentrale Koordination aller Agenten und Services
- **Technology**: FastAPI + WebSocket für Echtzeit-Kommunikation
- **Features**:
  - Agent Discovery und Registration
  - Message Routing zwischen Agenten
  - Load Balancing
  - Health Monitoring

#### 2. Alpha Generator Agent
- **Purpose**: Automatische Generierung von Alpha-Faktoren
- **Technology**: Python mit WorldQuant Brain API Integration
- **Features**:
  - Moonshot API Integration für Alpha-Ideen
  - WorldQuant Brain Simulation
  - Retry Logic für Simulation Limits
  - Performance Tracking

#### 3. n8n Workflow Engine
- **Purpose**: Workflow-Automatisierung und Integration
- **Technology**: n8n Docker Container
- **Features**:
  - Custom Nodes für Agent-Integration
  - Webhook Endpoints
  - Database Integration
  - Workflow Orchestration

#### 4. Frontend Application
- **Purpose**: Benutzeroberfläche für Agent-Management
- **Technology**: Next.js 14 mit TypeScript
- **Features**:
  - Real-time Agent Monitoring
  - Workflow Designer
  - Alpha Performance Dashboard
  - Agent Configuration

#### 5. Database Layer
- **Purpose**: Persistente Datenspeicherung
- **Technology**: PostgreSQL
- **Features**:
  - Agent States
  - Workflow History
  - Alpha Results
  - User Sessions

## Docker Architecture

### Service Composition
```
agent-network/
├── docker-compose.yml
├── agent-hub/
│   ├── Dockerfile
│   └── src/
├── alpha-generator/
│   ├── Dockerfile
│   └── src/
├── n8n/
│   ├── Dockerfile
│   └── custom-nodes/
├── frontend/
│   ├── Dockerfile
│   └── src/
├── database/
│   └── init/
└── nginx/
    └── nginx.conf
```

### Network Configuration
- **Internal Network**: `agent-network-internal`
- **External Network**: `agent-network-external`
- **Port Mapping**:
  - Frontend: 3000
  - Agent Hub: 8000
  - n8n: 5678
  - Database: 5432
  - Nginx: 80, 443

## Agent Communication Protocol

### Message Format
```json
{
  "id": "uuid",
  "timestamp": "2024-01-01T00:00:00Z",
  "sender": "agent-id",
  "recipient": "agent-id",
  "type": "request|response|event",
  "payload": {},
  "metadata": {}
}
```

### Agent Types
1. **Alpha Generator Agent**
   - Generiert Alpha-Faktoren
   - Führt Simulationen durch
   - Berichtet Performance-Metriken

2. **Workflow Agent**
   - Orchestriert n8n Workflows
   - Überwacht Workflow-Status
   - Trigger Events

3. **Data Agent**
   - Sammelt Marktdaten
   - Bereinigt und validiert Daten
   - Stellt Daten für andere Agenten bereit

4. **Analysis Agent**
   - Analysiert Alpha-Performance
   - Generiert Reports
   - Identifiziert Optimierungsmöglichkeiten

## Integration Points

### n8n Custom Nodes
- **Alpha Generator Node**: Trigger Alpha-Generierung
- **WorldQuant Node**: API Integration
- **Agent Communication Node**: Inter-Agent Messaging
- **Database Node**: CRUD Operations

### Frontend Integration
- **WebSocket Connection**: Real-time Updates
- **REST API**: CRUD Operations
- **File Upload**: Alpha-Expressions
- **Dashboard**: Performance Metrics

## Security Architecture

### Authentication
- JWT-based Authentication
- API Key Management
- Role-based Access Control

### Network Security
- Internal Service Communication
- HTTPS for External Access
- Rate Limiting
- Input Validation

## Monitoring & Logging

### Metrics
- Agent Health Status
- Workflow Execution Times
- Alpha Performance Metrics
- System Resource Usage

### Logging
- Centralized Logging (ELK Stack)
- Structured Logging
- Error Tracking
- Performance Monitoring

## Deployment Strategy

### Development
- Docker Compose für lokale Entwicklung
- Hot Reload für Frontend und Backend
- Volume Mounts für Code-Änderungen

### Production
- Kubernetes Deployment
- Auto-scaling basierend auf Workload
- Load Balancing
- Backup Strategy

## Data Flow

1. **Alpha Generation**:
   - Frontend → Agent Hub → Alpha Generator
   - Alpha Generator → WorldQuant API
   - Results → Database → Frontend

2. **Workflow Execution**:
   - Frontend → n8n → Custom Nodes
   - Custom Nodes → Agent Hub
   - Agent Hub → Database

3. **Real-time Updates**:
   - Agent Events → WebSocket → Frontend
   - Status Changes → Database → Frontend

## Configuration Management

### Environment Variables
- API Keys
- Database Connections
- Service URLs
- Feature Flags

### Configuration Files
- Agent Settings
- Workflow Templates
- UI Configuration
- Security Policies

## Future Enhancements

### Planned Features
- Machine Learning Model Integration
- Advanced Alpha Optimization
- Multi-market Support
- Mobile Application
- Advanced Analytics Dashboard

### Scalability Considerations
- Horizontal Scaling
- Microservice Architecture
- Event-driven Architecture
- Caching Strategy
