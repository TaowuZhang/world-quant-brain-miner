from typing import Dict, List, Optional, Any
from sqlalchemy.orm import Session
from sqlalchemy.ext.asyncio import AsyncSession
import httpx
import json
import logging
from datetime import datetime, timedelta
import asyncio
from .models import Agent, Message, Workflow, AlphaResult, SystemEvent
from .schemas import AgentCreate, MessageCreate, WorkflowCreate
from .websocket_manager import WebSocketManager
import os

logger = logging.getLogger(__name__)

class AgentManager:
    def __init__(self):
        self.active_agents: Dict[str, dict] = {}
        self.websocket_agents: Dict[str, Any] = {}
        self.total_messages = 0
        self.alpha_generator_url = None
        self.n8n_url = None
        
    async def start(self):
        """Initialize the agent manager"""
        logger.info("Starting Agent Manager...")
        # Start background tasks
        asyncio.create_task(self._monitor_agents())
        asyncio.create_task(self._cleanup_inactive_agents())
        
    async def stop(self):
        """Stop the agent manager"""
        logger.info("Stopping Agent Manager...")
        
    async def register_agent(self, agent_data: AgentCreate, db: AsyncSession) -> Agent:
        """Register a new agent"""
        try:
            agent = Agent(
                name=agent_data.name,
                agent_type=agent_data.agent_type.value,
                capabilities=agent_data.capabilities,
                metadata=agent_data.metadata
            )
            
            db.add(agent)
            await db.commit()
            await db.refresh(agent)
            
            # Add to active agents
            self.active_agents[agent.id] = {
                "agent": agent,
                "last_heartbeat": datetime.utcnow(),
                "status": "active"
            }
            
            # Log system event
            await self._log_system_event("agent_register", agent.id, {
                "agent_name": agent.name,
                "agent_type": agent.agent_type
            }, db)
            
            logger.info(f"Registered agent: {agent.name} ({agent.id})")
            return agent
            
        except Exception as e:
            logger.error(f"Error registering agent: {e}")
            await db.rollback()
            raise
    
    async def list_agents(self, db: AsyncSession) -> List[Agent]:
        """List all registered agents"""
        try:
            result = await db.execute("SELECT * FROM agents ORDER BY created_at DESC")
            agents = result.fetchall()
            return [Agent(**dict(agent)) for agent in agents]
        except Exception as e:
            logger.error(f"Error listing agents: {e}")
            return []
    
    async def get_agent(self, agent_id: str, db: AsyncSession) -> Optional[Agent]:
        """Get agent by ID"""
        try:
            result = await db.execute("SELECT * FROM agents WHERE id = :agent_id", {"agent_id": agent_id})
            agent_data = result.fetchone()
            if agent_data:
                return Agent(**dict(agent_data))
            return None
        except Exception as e:
            logger.error(f"Error getting agent {agent_id}: {e}")
            return None
    
    async def unregister_agent(self, agent_id: str, db: AsyncSession) -> bool:
        """Unregister an agent"""
        try:
            # Remove from active agents
            if agent_id in self.active_agents:
                del self.active_agents[agent_id]
            
            # Remove from websocket agents
            if agent_id in self.websocket_agents:
                del self.websocket_agents[agent_id]
            
            # Update database
            await db.execute("UPDATE agents SET status = 'inactive' WHERE id = :agent_id", {"agent_id": agent_id})
            await db.commit()
            
            # Log system event
            await self._log_system_event("agent_unregister", agent_id, {}, db)
            
            logger.info(f"Unregistered agent: {agent_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error unregistering agent {agent_id}: {e}")
            await db.rollback()
            return False
    
    async def send_message(self, message_data: MessageCreate, db: AsyncSession) -> bool:
        """Send a message to an agent"""
        try:
            message = Message(
                sender_id=message_data.sender_id,
                recipient_id=message_data.recipient_id,
                message_type=message_data.message_type.value,
                payload=message_data.payload,
                metadata=message_data.metadata
            )
            
            db.add(message)
            await db.commit()
            await db.refresh(message)
            
            # Send via WebSocket if agent is connected
            if message_data.recipient_id in self.websocket_agents:
                websocket = self.websocket_agents[message_data.recipient_id]
                await websocket.send_text(json.dumps({
                    "type": "message",
                    "message_id": message.id,
                    "sender_id": message_data.sender_id,
                    "payload": message_data.payload,
                    "timestamp": datetime.utcnow().isoformat()
                }))
            
            self.total_messages += 1
            logger.info(f"Message sent from {message_data.sender_id} to {message_data.recipient_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error sending message: {e}")
            await db.rollback()
            return False
    
    async def get_messages(self, agent_id: str, db: AsyncSession) -> List[Message]:
        """Get messages for an agent"""
        try:
            result = await db.execute(
                "SELECT * FROM messages WHERE recipient_id = :agent_id ORDER BY created_at DESC LIMIT 100",
                {"agent_id": agent_id}
            )
            messages = result.fetchall()
            return [Message(**dict(msg)) for msg in messages]
        except Exception as e:
            logger.error(f"Error getting messages for {agent_id}: {e}")
            return []
    
    async def register_websocket_agent(self, agent_id: str, websocket):
        """Register an agent's WebSocket connection"""
        self.websocket_agents[agent_id] = websocket
        logger.info(f"WebSocket agent registered: {agent_id}")
    
    async def unregister_websocket_agent(self, agent_id: str):
        """Unregister an agent's WebSocket connection"""
        if agent_id in self.websocket_agents:
            del self.websocket_agents[agent_id]
            logger.info(f"WebSocket agent unregistered: {agent_id}")
    
    async def route_message(self, message: dict, websocket_manager: WebSocketManager):
        """Route a message between agents"""
        recipient_id = message.get("recipient_id")
        if recipient_id and recipient_id in self.websocket_agents:
            await websocket_manager.send_personal_message(recipient_id, message)
    
    # Workflow management
    async def create_workflow(self, workflow_data: WorkflowCreate, db: AsyncSession) -> str:
        """Create a new workflow"""
        try:
            workflow = Workflow(
                name=workflow_data.name,
                description=workflow_data.description,
                agent_id=workflow_data.agent_id,
                workflow_type=workflow_data.workflow_type.value,
                configuration=workflow_data.configuration
            )
            
            db.add(workflow)
            await db.commit()
            await db.refresh(workflow)
            
            logger.info(f"Created workflow: {workflow.name} ({workflow.id})")
            return workflow.id
            
        except Exception as e:
            logger.error(f"Error creating workflow: {e}")
            await db.rollback()
            raise
    
    async def list_workflows(self, db: AsyncSession) -> List[Workflow]:
        """List all workflows"""
        try:
            result = await db.execute("SELECT * FROM workflows ORDER BY created_at DESC")
            workflows = result.fetchall()
            return [Workflow(**dict(wf)) for wf in workflows]
        except Exception as e:
            logger.error(f"Error listing workflows: {e}")
            return []
    
    async def execute_workflow(self, workflow_id: str, db: AsyncSession) -> dict:
        """Execute a workflow"""
        try:
            # Get workflow details
            result = await db.execute("SELECT * FROM workflows WHERE id = :workflow_id", {"workflow_id": workflow_id})
            workflow_data = result.fetchone()
            
            if not workflow_data:
                return {"error": "Workflow not found"}
            
            workflow = Workflow(**dict(workflow_data))
            
            # Execute based on workflow type
            if workflow.workflow_type == "n8n":
                return await self._execute_n8n_workflow(workflow)
            else:
                return await self._execute_custom_workflow(workflow)
                
        except Exception as e:
            logger.error(f"Error executing workflow {workflow_id}: {e}")
            return {"error": str(e)}
    
    async def _execute_n8n_workflow(self, workflow: Workflow) -> dict:
        """Execute an n8n workflow"""
        try:
            n8n_url = os.getenv("N8N_URL", "http://n8n:5678")
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{n8n_url}/api/v1/workflows/{workflow.configuration.get('n8n_workflow_id')}/execute",
                    json=workflow.configuration.get("parameters", {})
                )
                
                if response.status_code == 200:
                    return {"status": "success", "result": response.json()}
                else:
                    return {"status": "error", "message": response.text}
                    
        except Exception as e:
            logger.error(f"Error executing n8n workflow: {e}")
            return {"status": "error", "message": str(e)}
    
    async def _execute_custom_workflow(self, workflow: Workflow) -> dict:
        """Execute a custom workflow"""
        try:
            # Custom workflow execution logic
            return {"status": "success", "message": "Custom workflow executed"}
        except Exception as e:
            logger.error(f"Error executing custom workflow: {e}")
            return {"status": "error", "message": str(e)}
    
    # Alpha generation
    async def trigger_alpha_generation(self, parameters: dict) -> dict:
        """Trigger alpha generation"""
        try:
            # Send message to alpha generator agent
            if "alpha_generator" in self.active_agents:
                message = MessageCreate(
                    sender_id="system",
                    recipient_id="alpha_generator",
                    message_type="request",
                    payload={
                        "action": "generate_alpha",
                        "parameters": parameters
                    }
                )
                
                # This would be sent via the message system
                return {"status": "triggered", "message": "Alpha generation triggered"}
            else:
                return {"status": "error", "message": "Alpha generator agent not available"}
                
        except Exception as e:
            logger.error(f"Error triggering alpha generation: {e}")
            return {"status": "error", "message": str(e)}
    
    async def get_alpha_results(self) -> List[dict]:
        """Get alpha generation results"""
        try:
            # This would fetch from database or cache
            return []
        except Exception as e:
            logger.error(f"Error getting alpha results: {e}")
            return []
    
    # n8n integration
    async def handle_n8n_webhook(self, payload: dict) -> dict:
        """Handle webhooks from n8n"""
        try:
            workflow_id = payload.get("workflow_id")
            execution_id = payload.get("execution_id")
            status = payload.get("status")
            
            logger.info(f"Received n8n webhook: {workflow_id} - {status}")
            
            # Process the webhook based on workflow type
            if status == "completed":
                return {"status": "success", "message": "Webhook processed"}
            else:
                return {"status": "pending", "message": "Workflow in progress"}
                
        except Exception as e:
            logger.error(f"Error handling n8n webhook: {e}")
            return {"status": "error", "message": str(e)}
    
    async def get_n8n_workflows(self) -> List[dict]:
        """Get n8n workflows"""
        try:
            n8n_url = os.getenv("N8N_URL", "http://n8n:5678")
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{n8n_url}/api/v1/workflows")
                
                if response.status_code == 200:
                    return response.json()
                else:
                    return []
                    
        except Exception as e:
            logger.error(f"Error getting n8n workflows: {e}")
            return []
    
    # Background tasks
    async def _monitor_agents(self):
        """Monitor agent health"""
        while True:
            try:
                current_time = datetime.utcnow()
                inactive_agents = []
                
                for agent_id, agent_info in self.active_agents.items():
                    last_heartbeat = agent_info["last_heartbeat"]
                    if current_time - last_heartbeat > timedelta(minutes=5):
                        inactive_agents.append(agent_id)
                        logger.warning(f"Agent {agent_id} appears to be inactive")
                
                # Mark inactive agents
                for agent_id in inactive_agents:
                    self.active_agents[agent_id]["status"] = "inactive"
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error in agent monitoring: {e}")
                await asyncio.sleep(60)
    
    async def _cleanup_inactive_agents(self):
        """Clean up inactive agents"""
        while True:
            try:
                current_time = datetime.utcnow()
                agents_to_remove = []
                
                for agent_id, agent_info in self.active_agents.items():
                    last_heartbeat = agent_info["last_heartbeat"]
                    if current_time - last_heartbeat > timedelta(hours=1):
                        agents_to_remove.append(agent_id)
                
                # Remove inactive agents
                for agent_id in agents_to_remove:
                    del self.active_agents[agent_id]
                    logger.info(f"Removed inactive agent: {agent_id}")
                
                await asyncio.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                logger.error(f"Error in cleanup: {e}")
                await asyncio.sleep(300)
    
    async def _log_system_event(self, event_type: str, agent_id: str, payload: dict, db: AsyncSession):
        """Log a system event"""
        try:
            event = SystemEvent(
                event_type=event_type,
                agent_id=agent_id,
                payload=payload
            )
            
            db.add(event)
            await db.commit()
            
        except Exception as e:
            logger.error(f"Error logging system event: {e}")
            await db.rollback() 