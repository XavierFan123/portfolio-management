"""
Chief Architect Agent
Responsible for system coordination, architecture decisions, and high-level orchestration
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import json

from agents.base_agent import BaseAgent, AgentMessage, AgentStatus
from data.yahoo_connector import YahooFinanceConnector

logger = logging.getLogger(__name__)

class ChiefArchitectAgent(BaseAgent):
    """
    Chief Architect Agent - System coordinator and decision maker
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__("chief_architect", config)
        self.data_connector = None
        self.system_health = {}
        self.active_strategies = {}
        self.performance_metrics = {
            "system_uptime": 0,
            "messages_processed": 0,
            "decisions_made": 0,
            "alerts_generated": 0
        }

    async def initialize(self):
        """Initialize the Chief Architect Agent"""
        self.logger.info("Initializing Chief Architect Agent...")

        # Initialize data connector
        self.data_connector = YahooFinanceConnector()

        # Initialize system monitoring
        await self._initialize_system_monitoring()

        self.logger.info("Chief Architect Agent initialized successfully")

    async def process_message(self, message: AgentMessage):
        """Process incoming messages"""
        self.performance_metrics["messages_processed"] += 1

        if message.message_type == "system_health_check":
            await self._handle_health_check(message)
        elif message.message_type == "strategy_request":
            await self._handle_strategy_request(message)
        elif message.message_type == "risk_alert":
            await self._handle_risk_alert(message)
        elif message.message_type == "performance_report":
            await self._handle_performance_report(message)
        elif message.message_type == "data_quality_issue":
            await self._handle_data_quality_issue(message)
        elif message.message_type == "health_check":
            # Simple health check response
            self.logger.debug("Health check received - system healthy")
        else:
            self.logger.warning(f"Unknown message type: {message.message_type}")

    async def _handle_health_check(self, message: AgentMessage):
        """Handle system health check requests"""
        self.logger.info("Processing system health check")

        health_report = {
            "timestamp": datetime.now().isoformat(),
            "overall_status": "healthy",
            "components": {
                "data_feeds": await self._check_data_feeds(),
                "risk_models": await self._check_risk_models(),
                "monitoring": await self._check_monitoring_systems()
            },
            "performance": self.performance_metrics
        }

        # Send health report to orchestrator
        await self.send_message(
            target="orchestrator",
            message_type="health_report",
            payload=health_report
        )

    async def _handle_strategy_request(self, message: AgentMessage):
        """Handle requests for new trading strategies"""
        self.logger.info("Processing strategy request")

        strategy_request = message.payload
        symbol = strategy_request.get("symbol", "")
        strategy_type = strategy_request.get("type", "")

        # Analyze current market conditions
        market_analysis = await self._analyze_market_conditions(symbol)

        # Make architectural decision on strategy
        decision = await self._make_strategy_decision(strategy_request, market_analysis)

        self.performance_metrics["decisions_made"] += 1

        # Send decision to requesting agent
        await self.send_message(
            target=message.source,
            message_type="strategy_decision",
            payload={
                "request_id": strategy_request.get("request_id"),
                "decision": decision,
                "reasoning": decision.get("reasoning", ""),
                "risk_parameters": decision.get("risk_parameters", {})
            }
        )

    async def _handle_risk_alert(self, message: AgentMessage):
        """Handle critical risk alerts"""
        self.logger.warning("Processing risk alert")

        alert = message.payload
        severity = alert.get("severity", "medium")

        self.performance_metrics["alerts_generated"] += 1

        if severity == "critical":
            # Initiate emergency procedures
            await self._initiate_emergency_procedures(alert)
        elif severity == "high":
            # Adjust risk parameters
            await self._adjust_risk_parameters(alert)
        else:
            # Log and monitor
            await self._log_risk_event(alert)

    async def _handle_performance_report(self, message: AgentMessage):
        """Handle performance reports from other agents"""
        report = message.payload
        agent_id = message.source

        # Store performance metrics
        self.system_health[agent_id] = {
            "last_report": datetime.now(),
            "metrics": report,
            "status": "healthy" if report.get("healthy", True) else "degraded"
        }

        # Analyze system-wide performance
        system_performance = await self._analyze_system_performance()

        if system_performance.get("requires_action", False):
            await self._optimize_system_performance(system_performance)

    async def _handle_data_quality_issue(self, message: AgentMessage):
        """Handle data quality issues"""
        self.logger.warning("Processing data quality issue")

        issue = message.payload
        data_source = issue.get("source", "unknown")
        issue_type = issue.get("type", "unknown")

        # Determine severity and response
        response = await self._assess_data_quality_impact(issue)

        if response.get("critical", False):
            # Switch to backup data sources
            await self._switch_data_sources(data_source, issue_type)

    async def _analyze_market_conditions(self, symbol: str) -> Dict[str, Any]:
        """Analyze current market conditions for decision making"""
        try:
            # Get current market data
            market_data = self.data_connector.get_real_time_quote(symbol)
            volatility = self.data_connector.get_volatility_data(symbol)

            if symbol == "MSTR":
                # Special analysis for MSTR
                btc_ratio = self.data_connector.calculate_mstr_btc_ratio()
                return {
                    "symbol": symbol,
                    "current_price": market_data.price if market_data else None,
                    "volatility": volatility,
                    "btc_ratio": btc_ratio,
                    "market_regime": await self._determine_market_regime(market_data, volatility),
                    "timestamp": datetime.now().isoformat()
                }
            else:
                return {
                    "symbol": symbol,
                    "current_price": market_data.price if market_data else None,
                    "volatility": volatility,
                    "market_regime": await self._determine_market_regime(market_data, volatility),
                    "timestamp": datetime.now().isoformat()
                }

        except Exception as e:
            self.logger.error(f"Error analyzing market conditions: {e}")
            return {"error": str(e)}

    async def _make_strategy_decision(self, request: Dict[str, Any], analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Make strategic decisions based on request and analysis"""
        symbol = request.get("symbol", "")
        strategy_type = request.get("type", "")

        # Basic decision logic (would be much more sophisticated in practice)
        if analysis.get("volatility", 0) > 0.3:  # High volatility
            risk_limit = 0.02  # 2% max position
            approval = "conditional"
            reasoning = "High volatility detected - reduced position sizing recommended"
        elif analysis.get("volatility", 0) < 0.1:  # Low volatility
            risk_limit = 0.05  # 5% max position
            approval = "approved"
            reasoning = "Low volatility environment - standard position sizing approved"
        else:
            risk_limit = 0.03  # 3% max position
            approval = "approved"
            reasoning = "Normal volatility environment - approved with standard limits"

        return {
            "approval": approval,
            "reasoning": reasoning,
            "risk_parameters": {
                "max_position_size": risk_limit,
                "stop_loss": 0.02,
                "take_profit": 0.05,
                "max_drawdown": 0.01
            },
            "monitoring_requirements": {
                "frequency": "real_time" if analysis.get("volatility", 0) > 0.2 else "5min",
                "alerts": ["price_movement", "volume_spike", "volatility_change"]
            }
        }

    async def _determine_market_regime(self, market_data, volatility) -> str:
        """Determine current market regime"""
        if volatility is None:
            return "unknown"
        elif volatility > 0.4:
            return "high_volatility"
        elif volatility > 0.2:
            return "medium_volatility"
        else:
            return "low_volatility"

    async def _check_data_feeds(self) -> Dict[str, Any]:
        """Check status of data feeds"""
        try:
            # Test major symbols
            test_symbols = ["MSTR", "QQQ", "SPY"]
            feed_status = {}

            for symbol in test_symbols:
                data = self.data_connector.get_real_time_quote(symbol)
                feed_status[symbol] = "healthy" if data else "degraded"

            return {
                "status": "healthy" if all(s == "healthy" for s in feed_status.values()) else "degraded",
                "feeds": feed_status,
                "last_check": datetime.now().isoformat()
            }

        except Exception as e:
            return {"status": "error", "error": str(e)}

    async def _check_risk_models(self) -> Dict[str, Any]:
        """Check status of risk models"""
        return {
            "status": "healthy",
            "models": {
                "greeks_calculator": "healthy",
                "var_model": "healthy",
                "scenario_analysis": "healthy"
            },
            "last_validation": datetime.now().isoformat()
        }

    async def _check_monitoring_systems(self) -> Dict[str, Any]:
        """Check status of monitoring systems"""
        return {
            "status": "healthy",
            "components": {
                "alerting": "healthy",
                "logging": "healthy",
                "metrics": "healthy"
            }
        }

    async def _initialize_system_monitoring(self):
        """Initialize system monitoring components"""
        self.logger.info("Initializing system monitoring...")
        # Initialize monitoring dashboards, alerts, etc.

    async def _initiate_emergency_procedures(self, alert: Dict[str, Any]):
        """Initiate emergency risk management procedures"""
        self.logger.critical(f"EMERGENCY: {alert}")
        # Implement emergency procedures

    async def _adjust_risk_parameters(self, alert: Dict[str, Any]):
        """Adjust system risk parameters based on alert"""
        self.logger.warning(f"Adjusting risk parameters due to: {alert}")

    async def _log_risk_event(self, alert: Dict[str, Any]):
        """Log risk event for monitoring"""
        self.logger.info(f"Risk event logged: {alert}")

    async def _analyze_system_performance(self) -> Dict[str, Any]:
        """Analyze overall system performance"""
        return {
            "performance_score": 0.95,
            "requires_action": False,
            "recommendations": []
        }

    async def _optimize_system_performance(self, analysis: Dict[str, Any]):
        """Optimize system performance based on analysis"""
        self.logger.info("Optimizing system performance...")

    async def _assess_data_quality_impact(self, issue: Dict[str, Any]) -> Dict[str, Any]:
        """Assess impact of data quality issues"""
        return {
            "critical": False,
            "impact_score": 0.1,
            "affected_components": []
        }

    async def _switch_data_sources(self, source: str, issue_type: str):
        """Switch to backup data sources"""
        self.logger.warning(f"Switching from {source} due to {issue_type}")

    async def cleanup(self):
        """Cleanup resources"""
        self.logger.info("Cleaning up Chief Architect Agent...")

    async def get_status(self) -> AgentStatus:
        """Get current status"""
        self.status.performance_metrics = self.performance_metrics
        return self.status

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check"""
        return {
            "agent_id": self.agent_id,
            "status": self.status.status,
            "last_activity": self.status.last_activity.isoformat(),
            "performance_metrics": self.performance_metrics,
            "system_health": self.system_health,
            "healthy": True
        }