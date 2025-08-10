# Agent Network - Docker-basiertes Agentennetzwerk

Ein vollständiges Docker-basiertes Agentennetzwerk mit Alpha-Generierung, n8n Workflow-Integration und modernem Frontend.

## 🏗️ Architektur

Das System besteht aus folgenden Komponenten:

- **Agent Hub**: Zentrale Koordination aller Agenten (FastAPI + WebSocket)
- **Alpha Generator Agent**: Automatische Alpha-Faktor-Generierung mit WorldQuant Brain
- **n8n Workflow Engine**: Workflow-Automatisierung mit Custom Nodes
- **Frontend**: Next.js Dashboard für Agent-Management
- **Database**: PostgreSQL für persistente Datenspeicherung
- **Redis**: Caching und Session-Management
- **Nginx**: Reverse Proxy und Load Balancing
- **Monitoring**: Prometheus + Grafana für System-Monitoring

## 🚀 Schnellstart

### Voraussetzungen

- Docker und Docker Compose
- WorldQuant Brain Credentials
- Moonshot API Key

### 1. Credentials einrichten

Erstellen Sie eine `credentials` Datei im Root-Verzeichnis:

```bash
mkdir credentials
echo '["your_username", "your_password"]' > credentials/credential.txt
```

### 2. Environment Variables

Erstellen Sie eine `.env` Datei:

```bash
# Moonshot API Key
MOONSHOT_API_KEY=sk-your-moonshot-api-key

# Optional: Customize URLs
AGENT_HUB_URL=http://localhost:8000
N8N_URL=http://localhost:5678
```

### 3. System starten

```bash
# Alle Services starten
docker-compose up -d

# Logs anzeigen
docker-compose logs -f

# Spezifische Service-Logs
docker-compose logs -f agent-hub
docker-compose logs -f alpha-generator
docker-compose logs -f frontend
```

### 4. Zugriff auf Services

- **Frontend Dashboard**: http://localhost:3000
- **Agent Hub API**: http://localhost:8000
- **n8n Workflow Engine**: http://localhost:5678
- **Grafana Monitoring**: http://localhost:3001
- **Prometheus**: http://localhost:9090

## 📊 Features

### Agent Hub
- ✅ Agent Registration und Discovery
- ✅ WebSocket-basierte Echtzeit-Kommunikation
- ✅ Message Routing zwischen Agenten
- ✅ Health Monitoring
- ✅ REST API für externe Integration

### Alpha Generator Agent
- ✅ Moonshot API Integration für Alpha-Ideen
- ✅ WorldQuant Brain Simulation
- ✅ Retry Logic für Simulation Limits
- ✅ Batch Processing
- ✅ Performance Tracking

### n8n Integration
- ✅ Custom Agent Hub Node
- ✅ Webhook Integration
- ✅ Workflow Orchestration
- ✅ Real-time Status Updates

### Frontend Dashboard
- ✅ Real-time Agent Monitoring
- ✅ Alpha Performance Dashboard
- ✅ WebSocket-basierte Updates
- ✅ Modern UI mit Tailwind CSS
- ✅ Responsive Design

## 🔧 Konfiguration

### Agent Hub Konfiguration

```yaml
# docker-compose.yml
agent-hub:
  environment:
    - DATABASE_URL=postgresql://agent_user:agent_password@postgres:5432/agent_network
    - REDIS_URL=redis://redis:6379
    - JWT_SECRET=your-super-secret-jwt-key
    - MOONSHOT_API_KEY=${MOONSHOT_API_KEY}
```

### Alpha Generator Konfiguration

```yaml
alpha-generator:
  environment:
    - AGENT_HUB_URL=http://agent-hub:8000
    - MOONSHOT_API_KEY=${MOONSHOT_API_KEY}
    - WORLDQUANT_CREDENTIALS_PATH=/app/credentials/credential.txt
```

### n8n Konfiguration

```yaml
n8n:
  environment:
    - N8N_BASIC_AUTH_ACTIVE=true
    - N8N_BASIC_AUTH_USER=admin
    - N8N_BASIC_AUTH_PASSWORD=admin123
    - DB_TYPE=postgresdb
    - DB_POSTGRESDB_HOST=postgres
    - DB_POSTGRESDB_DATABASE=agent_network
```

## 🔌 API Endpoints

### Agent Hub API

```bash
# Agent Management
GET    /agents                    # List all agents
POST   /agents                    # Register new agent
GET    /agents/{id}              # Get agent details
DELETE /agents/{id}              # Unregister agent

# Message Routing
POST   /messages                  # Send message to agent
GET    /messages/{agent_id}      # Get messages for agent

# Alpha Generation
POST   /alpha/generate           # Trigger alpha generation
GET    /alpha/results            # Get alpha results

# Workflow Management
POST   /workflows                # Create workflow
GET    /workflows                # List workflows
POST   /workflows/{id}/execute   # Execute workflow

# System Metrics
GET    /metrics                  # Get system metrics
GET    /health                   # Health check

# WebSocket
WS     /ws/{client_id}          # WebSocket connection
```

### n8n Custom Nodes

- **Agent Hub Node**: Interaktion mit Agent Hub API
- **Alpha Generator Node**: Alpha-Faktor-Generierung
- **WorldQuant Node**: WorldQuant Brain Integration

## 📈 Monitoring

### Prometheus Metrics

- Agent Health Status
- WebSocket Connection Count
- Message Processing Rate
- Alpha Generation Success Rate
- System Resource Usage

### Grafana Dashboards

- Agent Network Overview
- Alpha Performance Metrics
- System Health Dashboard
- Workflow Execution Analytics

## 🛠️ Entwicklung

### Lokale Entwicklung

```bash
# Development Mode starten
docker-compose -f docker-compose.dev.yml up -d

# Code-Änderungen werden automatisch reloaded
```

### Custom Nodes entwickeln

```bash
# n8n Custom Nodes
cd n8n/custom-nodes
npm install
npm run build
```

### Frontend entwickeln

```bash
# Frontend Development
cd frontend
npm install
npm run dev
```

## 🔒 Sicherheit

### Authentication
- JWT-based Authentication
- API Key Management
- Role-based Access Control

### Network Security
- Internal Service Communication
- HTTPS für externe Zugriffe
- Rate Limiting
- Input Validation

## 📝 Logging

### Log Levels
- **DEBUG**: Detaillierte Debug-Informationen
- **INFO**: Allgemeine System-Informationen
- **WARNING**: Warnungen und Hinweise
- **ERROR**: Fehler und Ausnahmen

### Log Locations
```bash
# Agent Hub Logs
docker-compose logs agent-hub

# Alpha Generator Logs
docker-compose logs alpha-generator

# n8n Logs
docker-compose logs n8n

# Frontend Logs
docker-compose logs frontend
```

## 🚨 Troubleshooting

### Häufige Probleme

1. **Agent Hub nicht erreichbar**
   ```bash
   docker-compose logs agent-hub
   # Prüfen Sie die Datenbankverbindung
   ```

2. **Alpha Generator Fehler**
   ```bash
   docker-compose logs alpha-generator
   # Prüfen Sie die Credentials und API Keys
   ```

3. **n8n Custom Nodes nicht verfügbar**
   ```bash
   # Custom Nodes neu bauen
   docker-compose exec n8n npm run build
   ```

4. **WebSocket Verbindungsprobleme**
   ```bash
   # Nginx Logs prüfen
   docker-compose logs nginx
   ```

### Debug-Modus

```bash
# Debug-Logs aktivieren
docker-compose -f docker-compose.debug.yml up -d
```

## 📚 Dokumentation

- [Architektur-Dokumentation](ARCHITECTURE.md)
- [API Dokumentation](http://localhost:8000/docs)
- [n8n Dokumentation](https://docs.n8n.io/)

## 🤝 Beitragen

1. Fork das Repository
2. Erstellen Sie einen Feature Branch
3. Committen Sie Ihre Änderungen
4. Erstellen Sie einen Pull Request

## 📄 Lizenz

Dieses Projekt ist unter der MIT-Lizenz lizenziert.

## 🆘 Support

Bei Fragen oder Problemen:

1. Prüfen Sie die [Troubleshooting](#-troubleshooting) Sektion
2. Schauen Sie in die [Issues](../../issues)
3. Erstellen Sie ein neues Issue mit detaillierten Informationen

---

**Hinweis**: Stellen Sie sicher, dass Sie gültige WorldQuant Brain Credentials und einen Moonshot API Key haben, bevor Sie das System starten. 