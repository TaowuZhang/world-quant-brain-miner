from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from contextlib import asynccontextmanager
import json
import asyncio
import logging
from typing import Dict, List, Optional
import os
from datetime import datetime, timedelta

from .database import get_db, init_db
from .models import Agent, Message, Workflow
from .schemas import AgentCreate, AgentResponse, MessageCreate, WorkflowCreate
from .auth import get_current_user, create_access_token
from .websocket_manager import WebSocketManager
from .agent_manager import AgentManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables
websocket_manager = WebSocketManager()
agent_manager = AgentManager()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting Agent Network Hub...")
    await init_db()
    await agent_manager.start()
    yield
    # Shutdown
    logger.info("Shutting down Agent Network Hub...")
    await agent_manager.stop()

app = FastAPI(
    title="Agent Network Hub",
    description="Central hub for coordinating AI agents and workflows",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()

@app.get("/")
async def root():
    return {"message": "Agent Network Hub is running"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "agents": len(agent_manager.active_agents),
        "websocket_connections": len(websocket_manager.active_connections)
    }

# Authentication endpoints
@app.post("/auth/login")
async def login(credentials: dict):
    # Simple authentication for demo purposes
    # In production, implement proper user authentication
    if credentials.get("username") == "admin" and credentials.get("password") == "admin123":
        token = create_access_token(data={"sub": credentials["username"]})
        return {"access_token": token, "token_type": "bearer"}
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Incorrect username or password"
    )

# Agent management endpoints
@app.post("/agents", response_model=AgentResponse)
async def register_agent(agent_data: AgentCreate, db=Depends(get_db)):
    """Register a new agent with the hub"""
    agent = await agent_manager.register_agent(agent_data, db)
    return agent

@app.get("/agents", response_model=List[AgentResponse])
async def list_agents(db=Depends(get_db)):
    """List all registered agents"""
    return await agent_manager.list_agents(db)

@app.get("/agents/{agent_id}", response_model=AgentResponse)
async def get_agent(agent_id: str, db=Depends(get_db)):
    """Get agent details"""
    agent = await agent_manager.get_agent(agent_id, db)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    return agent

@app.delete("/agents/{agent_id}")
async def unregister_agent(agent_id: str, db=Depends(get_db)):
    """Unregister an agent"""
    success = await agent_manager.unregister_agent(agent_id, db)
    if not success:
        raise HTTPException(status_code=404, detail="Agent not found")
    return {"message": "Agent unregistered successfully"}

# Message routing
@app.post("/messages")
async def send_message(message: MessageCreate, db=Depends(get_db)):
    """Send a message to an agent"""
    result = await agent_manager.send_message(message, db)
    if not result:
        raise HTTPException(status_code=404, detail="Recipient agent not found")
    return {"message": "Message sent successfully"}

@app.get("/messages/{agent_id}")
async def get_messages(agent_id: str, db=Depends(get_db)):
    """Get messages for an agent"""
    messages = await agent_manager.get_messages(agent_id, db)
    return messages

# Workflow management
@app.post("/workflows")
async def create_workflow(workflow: WorkflowCreate, db=Depends(get_db)):
    """Create a new workflow"""
    workflow_id = await agent_manager.create_workflow(workflow, db)
    return {"workflow_id": workflow_id}

@app.get("/workflows")
async def list_workflows(db=Depends(get_db)):
    """List all workflows"""
    workflows = await agent_manager.list_workflows(db)
    return workflows

@app.post("/workflows/{workflow_id}/execute")
async def execute_workflow(workflow_id: str, db=Depends(get_db)):
    """Execute a workflow"""
    result = await agent_manager.execute_workflow(workflow_id, db)
    return result

# WebSocket endpoint for real-time communication
@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await websocket_manager.connect(websocket, client_id)
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            # Handle different message types
            if message.get("type") == "agent_register":
                await agent_manager.register_websocket_agent(client_id, websocket)
            elif message.get("type") == "send_message":
                await agent_manager.route_message(message, websocket_manager)
            elif message.get("type") == "workflow_update":
                await websocket_manager.broadcast_to_agents(message)
            
            # Echo back for testing
            await websocket.send_text(json.dumps({
                "type": "ack",
                "message": "Message received",
                "timestamp": datetime.utcnow().isoformat()
            }))
            
    except WebSocketDisconnect:
        websocket_manager.disconnect(client_id)
        await agent_manager.unregister_websocket_agent(client_id)

# Alpha generation endpoints
@app.post("/alpha/generate")
async def generate_alpha(parameters: dict):
    """Trigger alpha generation"""
    result = await agent_manager.trigger_alpha_generation(parameters)
    return result

@app.get("/alpha/results")
async def get_alpha_results():
    """Get alpha generation results"""
    results = await agent_manager.get_alpha_results()
    return results

# n8n integration endpoints
@app.post("/n8n/webhook")
async def n8n_webhook(payload: dict):
    """Handle webhooks from n8n"""
    result = await agent_manager.handle_n8n_webhook(payload)
    return result

@app.get("/n8n/workflows")
async def get_n8n_workflows():
    """Get n8n workflows"""
    workflows = await agent_manager.get_n8n_workflows()
    return workflows

# Metrics endpoint
@app.get("/metrics")
async def get_metrics():
    """Get system metrics"""
    return {
        "active_agents": len(agent_manager.active_agents),
        "websocket_connections": len(websocket_manager.active_connections),
        "total_messages": agent_manager.total_messages,
        "system_uptime": datetime.utcnow().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 