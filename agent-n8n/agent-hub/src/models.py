from sqlalchemy import Column, String, DateTime, Text, JSON, Integer, ForeignKey, Boolean
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from .database import Base
import uuid
from datetime import datetime

class Agent(Base):
    __tablename__ = "agents"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String, nullable=False)
    agent_type = Column(String, nullable=False)  # alpha_generator, workflow, data, analysis
    status = Column(String, default="active")  # active, inactive, error
    capabilities = Column(JSON, default=dict)
    metadata = Column(JSON, default=dict)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    last_heartbeat = Column(DateTime)
    
    # Relationships
    messages = relationship("Message", back_populates="agent")
    workflows = relationship("Workflow", back_populates="agent")

class Message(Base):
    __tablename__ = "messages"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    sender_id = Column(String, ForeignKey("agents.id"))
    recipient_id = Column(String, ForeignKey("agents.id"))
    message_type = Column(String, nullable=False)  # request, response, event
    payload = Column(JSON, nullable=False)
    metadata = Column(JSON, default=dict)
    created_at = Column(DateTime, default=func.now())
    processed_at = Column(DateTime)
    status = Column(String, default="pending")  # pending, processed, failed
    
    # Relationships
    agent = relationship("Agent", back_populates="messages")

class Workflow(Base):
    __tablename__ = "workflows"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String, nullable=False)
    description = Column(Text)
    agent_id = Column(String, ForeignKey("agents.id"))
    workflow_type = Column(String, nullable=False)  # n8n, custom
    configuration = Column(JSON, nullable=False)
    status = Column(String, default="draft")  # draft, active, paused, error
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    last_executed = Column(DateTime)
    execution_count = Column(Integer, default=0)
    
    # Relationships
    agent = relationship("Agent", back_populates="workflows")

class AlphaResult(Base):
    __tablename__ = "alpha_results"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    expression = Column(Text, nullable=False)
    fitness = Column(String)
    sharpe = Column(String)
    turnover = Column(String)
    returns = Column(String)
    grade = Column(String)
    checks = Column(JSON)
    created_at = Column(DateTime, default=func.now())
    agent_id = Column(String, ForeignKey("agents.id"))
    
    # Relationships
    agent = relationship("Agent")

class SystemEvent(Base):
    __tablename__ = "system_events"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    event_type = Column(String, nullable=False)  # agent_register, agent_unregister, workflow_execute
    agent_id = Column(String, ForeignKey("agents.id"))
    payload = Column(JSON, nullable=False)
    created_at = Column(DateTime, default=func.now())
    
    # Relationships
    agent = relationship("Agent") 