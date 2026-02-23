"""
WebSocket service for real-time communication in REXI.
"""

import logging
from typing import Dict, List, Optional, Any
import json
import asyncio
from datetime import datetime
import uuid

from fastapi import WebSocket, WebSocketDisconnect
from pydantic import BaseModel

logger = logging.getLogger(__name__)

class WebSocketMessage(BaseModel):
    """WebSocket message model."""
    type: str
    data: Dict[str, Any]
    timestamp: datetime
    message_id: str

class ConnectionManager:
    """Manages WebSocket connections."""
    
    def __init__(self):
        """Initialize connection manager."""
        self.active_connections: Dict[str, WebSocket] = {}
        self.user_connections: Dict[str, List[str]] = {}
    
    async def connect(self, websocket: WebSocket, user_id: str) -> str:
        """Connect a WebSocket client."""
        await websocket.accept()
        connection_id = str(uuid.uuid4())
        self.active_connections[connection_id] = websocket
        
        if user_id not in self.user_connections:
            self.user_connections[user_id] = []
        self.user_connections[user_id].append(connection_id)
        
        logger.info(f"WebSocket connected: {connection_id} for user: {user_id}")
        
        # Send welcome message
        await self.send_message(connection_id, {
            "type": "connection_established",
            "data": {"connection_id": connection_id, "user_id": user_id}
        })
        
        return connection_id
    
    def disconnect(self, connection_id: str, user_id: str):
        """Disconnect a WebSocket client."""
        if connection_id in self.active_connections:
            del self.active_connections[connection_id]
        
        if user_id in self.user_connections:
            if connection_id in self.user_connections[user_id]:
                self.user_connections[user_id].remove(connection_id)
            
            if not self.user_connections[user_id]:
                del self.user_connections[user_id]
        
        logger.info(f"WebSocket disconnected: {connection_id}")
    
    async def send_message(self, connection_id: str, message: Dict[str, Any]) -> bool:
        """Send message to specific connection."""
        if connection_id not in self.active_connections:
            return False
        
        try:
            websocket = self.active_connections[connection_id]
            message_obj = WebSocketMessage(
                type=message.get("type", "message"),
                data=message,
                timestamp=datetime.utcnow(),
                message_id=str(uuid.uuid4())
            )
            
            await websocket.send_text(message_obj.model_dump_json())
            return True
            
        except Exception as e:
            logger.error(f"Failed to send message to {connection_id}: {e}")
            return False
    
    async def broadcast_to_user(self, user_id: str, message: Dict[str, Any]) -> int:
        """Broadcast message to all connections for a user."""
        if user_id not in self.user_connections:
            return 0
        
        sent_count = 0
        for connection_id in self.user_connections[user_id]:
            if await self.send_message(connection_id, message):
                sent_count += 1
        
        return sent_count
    
    async def broadcast_to_all(self, message: Dict[str, Any]) -> int:
        """Broadcast message to all active connections."""
        sent_count = 0
        for connection_id in self.active_connections:
            if await self.send_message(connection_id, message):
                sent_count += 1
        
        return sent_count
    
    def get_connection_count(self) -> int:
        """Get total number of active connections."""
        return len(self.active_connections)
    
    def get_user_connection_count(self, user_id: str) -> int:
        """Get number of connections for a user."""
        return len(self.user_connections.get(user_id, []))

class WebSocketService:
    """WebSocket service for real-time REXI operations."""
    
    def __init__(self):
        """Initialize WebSocket service."""
        self.manager = ConnectionManager()
        self.active_operations: Dict[str, Dict[str, Any]] = {}
    
    async def handle_connection(self, websocket: WebSocket, user_id: str = "default"):
        """Handle WebSocket connection."""
        connection_id = await self.manager.connect(websocket, user_id)
        
        try:
            while True:
                # Receive message from client
                data = await websocket.receive_text()
                message = json.loads(data)
                
                # Handle different message types
                await self.handle_message(connection_id, user_id, message)
                
        except WebSocketDisconnect:
            self.manager.disconnect(connection_id, user_id)
        except Exception as e:
            logger.error(f"WebSocket error for {connection_id}: {e}")
            self.manager.disconnect(connection_id, user_id)
    
    async def handle_message(self, connection_id: str, user_id: str, message: Dict[str, Any]):
        """Handle incoming WebSocket message."""
        message_type = message.get("type", "unknown")
        
        if message_type == "ping":
            await self.manager.send_message(connection_id, {"type": "pong"})
        
        elif message_type == "subscribe":
            # Handle subscription to specific events
            await self.handle_subscription(connection_id, user_id, message)
        
        elif message_type == "query":
            # Handle real-time query
            await self.handle_query(connection_id, user_id, message)
        
        elif message_type == "graph_update":
            # Handle graph update request
            await self.handle_graph_update(connection_id, user_id, message)
        
        else:
            logger.warning(f"Unknown message type: {message_type}")
    
    async def handle_subscription(self, connection_id: str, user_id: str, message: Dict[str, Any]):
        """Handle subscription to events."""
        events = message.get("events", [])
        
        # Store subscription preferences
        # In a real implementation, this would be stored in a database
        
        await self.manager.send_message(connection_id, {
            "type": "subscription_confirmed",
            "data": {"events": events}
        })
    
    async def handle_query(self, connection_id: str, user_id: str, message: Dict[str, Any]):
        """Handle real-time query."""
        query = message.get("query", "")
        query_id = str(uuid.uuid4())
        
        # Start operation tracking
        self.active_operations[query_id] = {
            "type": "query",
            "user_id": user_id,
            "connection_id": connection_id,
            "status": "processing",
            "started_at": datetime.utcnow()
        }
        
        # Send initial response
        await self.manager.send_message(connection_id, {
            "type": "query_started",
            "data": {"query_id": query_id, "query": query}
        })
        
        # In a real implementation, this would trigger the actual query processing
        # For now, simulate processing
        await asyncio.sleep(2)
        
        # Send completion
        await self.manager.send_message(connection_id, {
            "type": "query_completed",
            "data": {
                "query_id": query_id,
                "result": f"Mock result for: {query}",
                "processing_time": 2.0
            }
        })
        
        # Clean up operation tracking
        if query_id in self.active_operations:
            del self.active_operations[query_id]
    
    async def handle_graph_update(self, connection_id: str, user_id: str, message: Dict[str, Any]):
        """Handle graph update."""
        update_type = message.get("update_type", "")
        update_data = message.get("data", {})
        
        # Broadcast update to all user's connections
        await self.manager.broadcast_to_user(user_id, {
            "type": "graph_updated",
            "data": {
                "update_type": update_type,
                "data": update_data,
                "timestamp": datetime.utcnow().isoformat()
            }
        })
    
    async def notify_progress(self, operation_id: str, progress: float, message: str = ""):
        """Notify clients about operation progress."""
        if operation_id not in self.active_operations:
            return
        
        operation = self.active_operations[operation_id]
        connection_id = operation["connection_id"]
        
        await self.manager.send_message(connection_id, {
            "type": "progress_update",
            "data": {
                "operation_id": operation_id,
                "progress": progress,
                "message": message
            }
        })
    
    async def notify_graph_change(self, user_id: str, change_type: str, data: Dict[str, Any]):
        """Notify about graph changes."""
        await self.manager.broadcast_to_user(user_id, {
            "type": "graph_change",
            "data": {
                "change_type": change_type,
                "data": data,
                "timestamp": datetime.utcnow().isoformat()
            }
        })
    
    async def notify_entity_added(self, user_id: str, entity: Dict[str, Any]):
        """Notify about new entity."""
        await self.notify_graph_change(user_id, "entity_added", {"entity": entity})
    
    async def notify_relationship_added(self, user_id: str, relationship: Dict[str, Any]):
        """Notify about new relationship."""
        await self.notify_graph_change(user_id, "relationship_added", {"relationship": relationship})
    
    async def notify_learning_progress(self, user_id: str, progress_data: Dict[str, Any]):
        """Notify about learning progress."""
        await self.manager.broadcast_to_user(user_id, {
            "type": "learning_progress",
            "data": progress_data
        })
    
    async def notify_error(self, connection_id: str, error: str, operation_id: Optional[str] = None):
        """Notify about error."""
        error_data = {
            "type": "error",
            "data": {
                "error": error,
                "timestamp": datetime.utcnow().isoformat()
            }
        }
        
        if operation_id:
            error_data["data"]["operation_id"] = operation_id
        
        await self.manager.send_message(connection_id, error_data)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get WebSocket service statistics."""
        return {
            "active_connections": self.manager.get_connection_count(),
            "active_operations": len(self.active_operations),
            "user_connections": {
                user_id: self.manager.get_user_connection_count(user_id)
                for user_id in self.manager.user_connections
            }
        }

# Global WebSocket service instance
websocket_service = WebSocketService()
