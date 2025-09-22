"""
Cerberus Orchestrator
Main system orchestrator that coordinates all agents and manages the overall system
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import json
from dataclasses import dataclass, asdict

from agents.base_agent import BaseAgent, AgentMessage, AgentStatus
from agents.chief_architect import ChiefArchitectAgent
from agents.quant_modeler import QuantitativeModelerAgent
from agents.greeks_risk_agent import GreeksRiskAgent
from agents.data_infrastructure_agent import DataInfrastructureAgent
from agents.model_validation_agent import ModelValidationAgent
from config import Config

logger = logging.getLogger(__name__)

@dataclass
class SystemStatus:
    timestamp: datetime
    overall_status: str
    agent_statuses: Dict[str, str]
    active_alerts: int
    system_uptime: float

class CerberusOrchestrator:
    """
    Main orchestrator for the Cerberus risk management system
    """

    def __init__(self, config: Config):
        self.config = config
        self.agents: Dict[str, BaseAgent] = {}
        self.message_queue = asyncio.Queue()
        self.running = False
        self.system_start_time = datetime.now()
        self.message_routing = {}
        self.system_metrics = {
            "messages_processed": 0,
            "agents_started": 0,
            "alerts_handled": 0,
            "uptime_seconds": 0
        }

    async def initialize(self):
        """Initialize the orchestrator and all agents"""
        logger.info("Initializing Cerberus Orchestrator...")

        try:
            # Initialize all agents
            await self._initialize_agents()

            # Set up message routing
            await self._setup_message_routing()

            # Start message processing
            asyncio.create_task(self._message_processing_loop())

            # Start system monitoring
            asyncio.create_task(self._system_monitoring_loop())

            logger.info("Cerberus Orchestrator initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize orchestrator: {e}")
            raise

    async def _initialize_agents(self):
        """Initialize all system agents"""
        agent_configs = self.config.get_agent_configs()

        # Initialize Chief Architect
        self.agents["chief_architect"] = ChiefArchitectAgent(
            agent_configs.get("chief_architect", {})
        )

        # Initialize Quantitative Modeler
        self.agents["quant_modeler"] = QuantitativeModelerAgent(
            agent_configs.get("quant_modeler", {})
        )

        # Initialize Greeks & Risk Agent
        self.agents["greeks_risk_agent"] = GreeksRiskAgent(
            agent_configs.get("greeks_risk_agent", {})
        )

        # Initialize Data & Infrastructure Agent
        self.agents["data_infrastructure"] = DataInfrastructureAgent(
            agent_configs.get("data_infrastructure", {})
        )

        # Initialize Model Validation Agent
        self.agents["model_validation"] = ModelValidationAgent(
            agent_configs.get("model_validation", {})
        )

        # Start all agents
        for agent_id, agent in self.agents.items():
            try:
                await agent.start()
                self.system_metrics["agents_started"] += 1
                logger.info(f"Started agent: {agent_id}")
            except Exception as e:
                logger.error(f"Failed to start agent {agent_id}: {e}")
                raise

    async def _setup_message_routing(self):
        """Set up message routing between agents"""
        self.message_routing = {
            "chief_architect": {
                "targets": ["quant_modeler", "greeks_risk_agent", "data_infrastructure", "model_validation"],
                "handles": ["system_health_check", "strategy_request", "risk_alert", "performance_report"]
            },
            "quant_modeler": {
                "targets": ["greeks_risk_agent", "model_validation"],
                "handles": ["price_option", "calculate_greeks", "calculate_var", "calibrate_model"]
            },
            "greeks_risk_agent": {
                "targets": ["chief_architect", "quant_modeler"],
                "handles": ["calculate_portfolio_greeks", "update_position", "check_risk_limits"]
            },
            "data_infrastructure": {
                "targets": ["chief_architect", "quant_modeler", "greeks_risk_agent", "model_validation"],
                "handles": ["get_market_data", "system_health", "data_quality_report"]
            },
            "model_validation": {
                "targets": ["chief_architect", "quant_modeler"],
                "handles": ["validate_model", "run_backtest", "pnl_attribution"]
            }
        }

    async def start_monitoring(self):
        """Start real-time monitoring"""
        logger.info("Starting real-time monitoring...")

        # Start portfolio monitoring
        await self._start_portfolio_monitoring()

        # Start risk monitoring
        await self._start_risk_monitoring()

        # Start system health monitoring
        await self._start_health_monitoring()

    async def _start_portfolio_monitoring(self):
        """Start portfolio monitoring tasks"""
        # Example portfolio for monitoring
        sample_portfolio = [
            {
                "symbol": "MSTR",
                "quantity": 100,
                "type": "stock"
            },
            {
                "symbol": "QQQ",
                "quantity": 200,
                "type": "stock"
            }
        ]

        # Request initial Greeks calculation
        await self.send_message(
            source="orchestrator",
            target="greeks_risk_agent",
            message_type="calculate_portfolio_greeks",
            payload={
                "request_id": f"portfolio_init_{datetime.now().timestamp()}",
                "portfolio": sample_portfolio
            }
        )

    async def _start_risk_monitoring(self):
        """Start risk monitoring tasks"""
        # Request initial risk assessment
        await self.send_message(
            source="orchestrator",
            target="chief_architect",
            message_type="system_health_check",
            payload={
                "request_id": f"health_init_{datetime.now().timestamp()}",
                "full_check": True
            }
        )

    async def _start_health_monitoring(self):
        """Start system health monitoring"""
        # Request system health from infrastructure agent
        await self.send_message(
            source="orchestrator",
            target="data_infrastructure",
            message_type="system_health",
            payload={
                "request_id": f"sys_health_{datetime.now().timestamp()}"
            }
        )

    async def send_message(self, source: str, target: str, message_type: str, payload: Dict[str, Any], priority: int = 1):
        """Send message between agents"""
        message = AgentMessage(
            source=source,
            target=target,
            message_type=message_type,
            payload=payload,
            timestamp=datetime.now(),
            priority=priority
        )

        await self.message_queue.put(message)

    async def _message_processing_loop(self):
        """Main message processing loop"""
        while self.running:
            try:
                # Get message from queue
                message = await asyncio.wait_for(self.message_queue.get(), timeout=1.0)

                # Route message to target agent
                await self._route_message(message)

                self.system_metrics["messages_processed"] += 1

            except asyncio.TimeoutError:
                # No message in queue, continue
                continue
            except Exception as e:
                logger.error(f"Message processing error: {e}")

    async def _route_message(self, message: AgentMessage):
        """Route message to the appropriate agent"""
        target_agent = self.agents.get(message.target)

        if target_agent:
            await target_agent.receive_message(message)
        elif message.target == "orchestrator":
            # Handle messages directed to orchestrator
            await self._handle_orchestrator_message(message)
        else:
            logger.warning(f"Unknown target agent: {message.target}")

    async def _handle_orchestrator_message(self, message: AgentMessage):
        """Handle messages directed to the orchestrator"""
        if message.message_type == "system_status_request":
            status = await self.get_system_status()
            # Send response back to source
        elif message.message_type == "shutdown_request":
            await self.shutdown()
        elif message.message_type == "agent_error":
            await self._handle_agent_error(message)
        else:
            logger.warning(f"Unknown orchestrator message type: {message.message_type}")

    async def _handle_agent_error(self, message: AgentMessage):
        """Handle agent error reports"""
        error_info = message.payload
        agent_id = message.source

        logger.error(f"Agent error from {agent_id}: {error_info}")

        # Attempt to restart failed agent if necessary
        if error_info.get("critical", False):
            await self._restart_agent(agent_id)

    async def _restart_agent(self, agent_id: str):
        """Restart a failed agent"""
        try:
            agent = self.agents.get(agent_id)
            if agent:
                logger.info(f"Restarting agent: {agent_id}")
                await agent.stop()
                await agent.start()
                logger.info(f"Agent {agent_id} restarted successfully")
        except Exception as e:
            logger.error(f"Failed to restart agent {agent_id}: {e}")

    async def _system_monitoring_loop(self):
        """System-wide monitoring loop"""
        while self.running:
            try:
                # Update system metrics
                self.system_metrics["uptime_seconds"] = (
                    datetime.now() - self.system_start_time
                ).total_seconds()

                # Check agent health
                unhealthy_agents = []
                for agent_id, agent in self.agents.items():
                    try:
                        health = await agent.health_check()
                        if not health.get("healthy", False):
                            unhealthy_agents.append(agent_id)
                    except Exception as e:
                        logger.error(f"Health check failed for {agent_id}: {e}")
                        unhealthy_agents.append(agent_id)

                # Handle unhealthy agents
                if unhealthy_agents:
                    logger.warning(f"Unhealthy agents detected: {unhealthy_agents}")
                    # Could implement automatic recovery here

                # Send periodic health checks
                await self._send_periodic_health_checks()

                await asyncio.sleep(30)  # Monitor every 30 seconds

            except Exception as e:
                logger.error(f"System monitoring error: {e}")
                await asyncio.sleep(60)

    async def _send_periodic_health_checks(self):
        """Send periodic health checks to all agents"""
        for agent_id in self.agents.keys():
            await self.send_message(
                source="orchestrator",
                target=agent_id,
                message_type="health_check",
                payload={"timestamp": datetime.now().isoformat()},
                priority=3
            )

    async def get_system_status(self) -> SystemStatus:
        """Get overall system status"""
        agent_statuses = {}
        active_alerts = 0

        for agent_id, agent in self.agents.items():
            try:
                status = await agent.get_status()
                agent_statuses[agent_id] = status.status

                # Count any error statuses as alerts
                if status.status == "error":
                    active_alerts += 1

            except Exception as e:
                logger.error(f"Failed to get status for {agent_id}: {e}")
                agent_statuses[agent_id] = "unknown"
                active_alerts += 1

        # Determine overall status
        if all(status in ["idle", "processing"] for status in agent_statuses.values()):
            overall_status = "healthy"
        elif any(status == "error" for status in agent_statuses.values()):
            overall_status = "degraded"
        else:
            overall_status = "warning"

        uptime = (datetime.now() - self.system_start_time).total_seconds()

        return SystemStatus(
            timestamp=datetime.now(),
            overall_status=overall_status,
            agent_statuses=agent_statuses,
            active_alerts=active_alerts,
            system_uptime=uptime
        )

    async def run(self):
        """Main run loop"""
        self.running = True
        logger.info("Cerberus system is now running...")

        try:
            # Keep the system running
            while self.running:
                await asyncio.sleep(1)

        except KeyboardInterrupt:
            logger.info("Shutdown signal received")
        except Exception as e:
            logger.error(f"System error: {e}")
        finally:
            await self.shutdown()

    async def shutdown(self):
        """Shutdown the entire system"""
        logger.info("Shutting down Cerberus system...")

        self.running = False

        # Stop all agents
        for agent_id, agent in self.agents.items():
            try:
                await agent.stop()
                logger.info(f"Stopped agent: {agent_id}")
            except Exception as e:
                logger.error(f"Error stopping agent {agent_id}: {e}")

        logger.info("Cerberus system shutdown complete")

    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get system performance metrics"""
        agent_metrics = {}

        for agent_id, agent in self.agents.items():
            try:
                health = await agent.health_check()
                agent_metrics[agent_id] = health.get("performance_metrics", {})
            except Exception as e:
                logger.error(f"Failed to get metrics for {agent_id}: {e}")

        return {
            "system_metrics": self.system_metrics,
            "agent_metrics": agent_metrics,
            "timestamp": datetime.now().isoformat()
        }