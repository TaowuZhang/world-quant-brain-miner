from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
from datetime import datetime
from enum import Enum

class AgentType(str, Enum):
    ALPHA_GENERATOR = "alpha_generator"
    WORKFLOW = "workflow"
    DATA = "data"
    ANALYSIS = "analysis"

class AgentStatus(str, Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    ERROR = "error"

class MessageType(str, Enum):
    REQUEST = "request"
    RESPONSE = "response"
    EVENT = "event"

class WorkflowType(str, Enum):
    N8N = "n8n"
    CUSTOM = "custom"

class WorkflowStatus(str, Enum):
    DRAFT = "draft"
    ACTIVE = "active"
    PAUSED = "paused"
    ERROR = "error"

# Agent schemas
class AgentCreate(BaseModel):
    name: str = Field(..., description="Agent name")
    agent_type: AgentType = Field(..., description="Type of agent")
    capabilities: Dict[str, Any] = Field(default_factory=dict, description="Agent capabilities")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

class AgentResponse(BaseModel):
    id: str
    name: str
    agent_type: AgentType
    status: AgentStatus
    capabilities: Dict[str, Any]
    metadata: Dict[str, Any]
    created_at: datetime
    updated_at: datetime
    last_heartbeat: Optional[datetime] = None

    class Config:
        from_attributes = True

# Message schemas
class MessageCreate(BaseModel):
    sender_id: str = Field(..., description="Sender agent ID")
    recipient_id: str = Field(..., description="Recipient agent ID")
    message_type: MessageType = Field(..., description="Type of message")
    payload: Dict[str, Any] = Field(..., description="Message payload")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

class MessageResponse(BaseModel):
    id: str
    sender_id: str
    recipient_id: str
    message_type: MessageType
    payload: Dict[str, Any]
    metadata: Dict[str, Any]
    created_at: datetime
    processed_at: Optional[datetime] = None
    status: str

    class Config:
        from_attributes = True

# Workflow schemas
class WorkflowCreate(BaseModel):
    name: str = Field(..., description="Workflow name")
    description: Optional[str] = Field(None, description="Workflow description")
    agent_id: str = Field(..., description="Associated agent ID")
    workflow_type: WorkflowType = Field(..., description="Type of workflow")
    configuration: Dict[str, Any] = Field(..., description="Workflow configuration")

class WorkflowResponse(BaseModel):
    id: str
    name: str
    description: Optional[str]
    agent_id: str
    workflow_type: WorkflowType
    configuration: Dict[str, Any]
    status: WorkflowStatus
    created_at: datetime
    updated_at: datetime
    last_executed: Optional[datetime] = None
    execution_count: int

    class Config:
        from_attributes = True

# Alpha generation schemas
class AlphaGenerationRequest(BaseModel):
    batch_size: int = Field(default=5, ge=1, le=20, description="Number of alphas to generate")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Generation parameters")

class AlphaResultResponse(BaseModel):
    id: str
    expression: str
    fitness: Optional[str] = None
    sharpe: Optional[str] = None
    turnover: Optional[str] = None
    returns: Optional[str] = None
    grade: Optional[str] = None
    checks: Optional[Dict[str, Any]] = None
    created_at: datetime
    agent_id: str

    class Config:
        from_attributes = True

# n8n integration schemas
class N8nWebhookPayload(BaseModel):
    workflow_id: str
    execution_id: str
    status: str
    data: Dict[str, Any]
    timestamp: datetime

class N8nWorkflowResponse(BaseModel):
    id: str
    name: str
    active: bool
    nodes: List[Dict[str, Any]]
    connections: Dict[str, Any]
    settings: Dict[str, Any]

# System metrics schemas
class SystemMetrics(BaseModel):
    active_agents: int
    websocket_connections: int
    total_messages: int
    system_uptime: str
    memory_usage: Optional[float] = None
    cpu_usage: Optional[float] = None

# Authentication schemas
class LoginRequest(BaseModel):
    username: str
    password: str

class LoginResponse(BaseModel):
    access_token: str
    token_type: str
    expires_in: int

# WebSocket message schemas
class WebSocketMessage(BaseModel):
    type: str
    payload: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    sender_id: Optional[str] = None
    recipient_id: Optional[str] = None 