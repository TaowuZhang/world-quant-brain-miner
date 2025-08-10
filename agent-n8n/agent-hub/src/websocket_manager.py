from fastapi import WebSocket
from typing import Dict, List, Optional
import json
import logging
from datetime import datetime
import asyncio

logger = logging.getLogger(__name__)

class WebSocketManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.connection_metadata: Dict[str, dict] = {}
        
    async def connect(self, websocket: WebSocket, client_id: str):
        """Connect a new WebSocket client"""
        await websocket.accept()
        self.active_connections[client_id] = websocket
        self.connection_metadata[client_id] = {
            "connected_at": datetime.utcnow(),
            "last_activity": datetime.utcnow(),
            "message_count": 0
        }
        logger.info(f"WebSocket client {client_id} connected. Total connections: {len(self.active_connections)}")
        
        # Send welcome message
        await self.send_personal_message(client_id, {
            "type": "connection_established",
            "client_id": client_id,
            "timestamp": datetime.utcnow().isoformat()
        })
    
    def disconnect(self, client_id: str):
        """Disconnect a WebSocket client"""
        if client_id in self.active_connections:
            del self.active_connections[client_id]
        if client_id in self.connection_metadata:
            del self.connection_metadata[client_id]
        logger.info(f"WebSocket client {client_id} disconnected. Total connections: {len(self.active_connections)}")
    
    async def send_personal_message(self, client_id: str, message: dict):
        """Send a message to a specific client"""
        if client_id in self.active_connections:
            try:
                await self.active_connections[client_id].send_text(json.dumps(message))
                self.connection_metadata[client_id]["last_activity"] = datetime.utcnow()
                self.connection_metadata[client_id]["message_count"] += 1
            except Exception as e:
                logger.error(f"Error sending message to {client_id}: {e}")
                self.disconnect(client_id)
    
    async def broadcast_to_agents(self, message: dict):
        """Broadcast message to all agent connections"""
        disconnected_clients = []
        
        for client_id, websocket in self.active_connections.items():
            try:
                await websocket.send_text(json.dumps(message))
                self.connection_metadata[client_id]["last_activity"] = datetime.utcnow()
                self.connection_metadata[client_id]["message_count"] += 1
            except Exception as e:
                logger.error(f"Error broadcasting to {client_id}: {e}")
                disconnected_clients.append(client_id)
        
        # Clean up disconnected clients
        for client_id in disconnected_clients:
            self.disconnect(client_id)
    
    async def broadcast_to_clients(self, message: dict, exclude_client: Optional[str] = None):
        """Broadcast message to all clients except the excluded one"""
        disconnected_clients = []
        
        for client_id, websocket in self.active_connections.items():
            if client_id != exclude_client:
                try:
                    await websocket.send_text(json.dumps(message))
                    self.connection_metadata[client_id]["last_activity"] = datetime.utcnow()
                    self.connection_metadata[client_id]["message_count"] += 1
                except Exception as e:
                    logger.error(f"Error broadcasting to {client_id}: {e}")
                    disconnected_clients.append(client_id)
        
        # Clean up disconnected clients
        for client_id in disconnected_clients:
            self.disconnect(client_id)
    
    def get_connection_info(self, client_id: str) -> Optional[dict]:
        """Get connection information for a client"""
        if client_id in self.connection_metadata:
            return self.connection_metadata[client_id]
        return None
    
    def get_all_connections_info(self) -> List[dict]:
        """Get information about all active connections"""
        connections_info = []
        for client_id, metadata in self.connection_metadata.items():
            connections_info.append({
                "client_id": client_id,
                **metadata
            })
        return connections_info
    
    async def ping_all_connections(self):
        """Ping all connections to check if they're still alive"""
        disconnected_clients = []
        
        for client_id, websocket in self.active_connections.items():
            try:
                await websocket.send_text(json.dumps({
                    "type": "ping",
                    "timestamp": datetime.utcnow().isoformat()
                }))
            except Exception as e:
                logger.error(f"Error pinging {client_id}: {e}")
                disconnected_clients.append(client_id)
        
        # Clean up disconnected clients
        for client_id in disconnected_clients:
            self.disconnect(client_id)
    
    async def start_heartbeat(self, interval: int = 30):
        """Start heartbeat mechanism to keep connections alive"""
        while True:
            try:
                await asyncio.sleep(interval)
                await self.ping_all_connections()
                logger.debug(f"Heartbeat sent to {len(self.active_connections)} connections")
            except Exception as e:
                logger.error(f"Error in heartbeat: {e}")
    
    def get_stats(self) -> dict:
        """Get WebSocket manager statistics"""
        return {
            "total_connections": len(self.active_connections),
            "connections_info": self.get_all_connections_info()
        } 