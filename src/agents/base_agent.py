"""
Base Agent Class
Defines the common interface and functionality for all agents
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from datetime import datetime
from dataclasses import dataclass
import json

logger = logging.getLogger(__name__)

@dataclass
class AgentMessage:
    source: str
    target: str
    message_type: str
    payload: Dict[str, Any]
    timestamp: datetime
    priority: int = 1  # 1=high, 2=medium, 3=low

@dataclass
class AgentStatus:
    agent_id: str
    status: str  # 'idle', 'processing', 'error', 'shutdown'
    last_activity: datetime
    current_task: Optional[str] = None
    performance_metrics: Dict[str, Any] = None

class BaseAgent(ABC):
    """Base class for all system agents"""

    def __init__(self, agent_id: str, config: Dict[str, Any]):
        self.agent_id = agent_id
        self.config = config
        self.status = AgentStatus(
            agent_id=agent_id,
            status='idle',
            last_activity=datetime.now()
        )
        self.message_queue = asyncio.Queue()
        self.running = False
        self.logger = logging.getLogger(f"agent.{agent_id}")

    async def start(self):
        """Start the agent"""
        self.running = True
        self.status.status = 'idle'
        self.logger.info(f"Agent {self.agent_id} started")

        # Start message processing loop
        asyncio.create_task(self._message_loop())
        await self.initialize()

    async def stop(self):
        """Stop the agent"""
        self.running = False
        self.status.status = 'shutdown'
        self.logger.info(f"Agent {self.agent_id} stopped")
        await self.cleanup()

    async def send_message(self, target: str, message_type: str, payload: Dict[str, Any], priority: int = 1):
        """Send message to another agent"""
        message = AgentMessage(
            source=self.agent_id,
            target=target,
            message_type=message_type,
            payload=payload,
            timestamp=datetime.now(),
            priority=priority
        )
        # This would typically go through the orchestrator
        self.logger.debug(f"Sending message to {target}: {message_type}")

    async def receive_message(self, message: AgentMessage):
        """Receive message from another agent"""
        await self.message_queue.put(message)

    async def _message_loop(self):
        """Main message processing loop"""
        while self.running:
            try:
                message = await asyncio.wait_for(self.message_queue.get(), timeout=1.0)
                self.status.last_activity = datetime.now()
                self.status.status = 'processing'

                await self.process_message(message)

                self.status.status = 'idle'

            except asyncio.TimeoutError:
                # No message received, continue
                continue
            except Exception as e:
                self.logger.error(f"Error processing message: {e}")
                self.status.status = 'error'

    @abstractmethod
    async def initialize(self):
        """Initialize agent-specific components"""
        pass

    @abstractmethod
    async def process_message(self, message: AgentMessage):
        """Process incoming message"""
        pass

    @abstractmethod
    async def cleanup(self):
        """Cleanup agent resources"""
        pass

    @abstractmethod
    async def get_status(self) -> AgentStatus:
        """Get current agent status"""
        return self.status

    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check"""
        return {
            "agent_id": self.agent_id,
            "status": self.status.status,
            "last_activity": self.status.last_activity.isoformat(),
            "healthy": True
        }