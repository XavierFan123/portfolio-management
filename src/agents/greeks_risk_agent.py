"""
Greeks & Real-time Risk Agent
Responsible for real-time Greeks monitoring, risk calculations, and dynamic hedging
"""

import asyncio
import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict

from agents.base_agent import BaseAgent, AgentMessage, AgentStatus
from data.yahoo_connector import YahooFinanceConnector

logger = logging.getLogger(__name__)

@dataclass
class GreeksSnapshot:
    symbol: str
    timestamp: datetime
    spot_price: float
    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float
    total_delta: float
    total_gamma: float
    total_vega: float

@dataclass
class RiskLimit:
    name: str
    current_value: float
    limit_value: float
    utilization: float
    status: str  # 'green', 'yellow', 'red'

@dataclass
class HedgeRecommendation:
    symbol: str
    action: str  # 'buy', 'sell'
    quantity: int
    reasoning: str
    urgency: str  # 'low', 'medium', 'high'
    expected_delta_change: float

class GreeksRiskAgent(BaseAgent):
    """
    Greeks & Real-time Risk Agent - Real-time risk monitoring and hedging
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__("greeks_risk_agent", config)
        self.data_connector = None
        self.current_positions = {}
        self.risk_limits = {}
        self.greeks_history = []
        self.hedging_enabled = True
        self.performance_metrics = {
            "greeks_calculated": 0,
            "hedge_recommendations": 0,
            "risk_alerts": 0,
            "limit_breaches": 0
        }

    async def initialize(self):
        """Initialize the Greeks & Risk Agent"""
        self.logger.info("Initializing Greeks & Risk Agent...")

        # Initialize data connector
        self.data_connector = YahooFinanceConnector()

        # Initialize risk limits
        await self._initialize_risk_limits()

        # Start real-time monitoring
        asyncio.create_task(self._real_time_monitoring_loop())

        self.logger.info("Greeks & Risk Agent initialized successfully")

    async def process_message(self, message: AgentMessage):
        """Process incoming messages"""

        if message.message_type == "calculate_portfolio_greeks":
            await self._handle_portfolio_greeks(message)
        elif message.message_type == "update_position":
            await self._handle_position_update(message)
        elif message.message_type == "check_risk_limits":
            await self._handle_risk_limit_check(message)
        elif message.message_type == "hedge_recommendation":
            await self._handle_hedge_recommendation(message)
        elif message.message_type == "set_risk_limit":
            await self._handle_set_risk_limit(message)
        elif message.message_type == "emergency_hedge":
            await self._handle_emergency_hedge(message)
        elif message.message_type == "health_check":
            # Simple health check response
            self.logger.debug("Health check received - system healthy")
        else:
            self.logger.warning(f"Unknown message type: {message.message_type}")

    async def _handle_position_update(self, message: AgentMessage):
        """Handle position update messages"""
        request = message.payload
        self.logger.info(f"Position update received: {request}")

    async def _handle_risk_limit_check(self, message: AgentMessage):
        """Handle risk limit check requests"""
        await self._check_risk_limits()

    async def _handle_hedge_recommendation(self, message: AgentMessage):
        """Handle hedge recommendation requests"""
        await self._generate_hedge_recommendations()

    async def _handle_set_risk_limit(self, message: AgentMessage):
        """Handle risk limit setting requests"""
        request = message.payload
        limit_name = request.get("limit_name")
        limit_value = request.get("limit_value")
        if limit_name and limit_name in self.risk_limits:
            self.risk_limits[limit_name].limit_value = limit_value
            self.logger.info(f"Updated risk limit {limit_name} to {limit_value}")

    async def _handle_emergency_hedge(self, message: AgentMessage):
        """Handle emergency hedge requests"""
        self.logger.warning("Emergency hedge request received")
        # Implement emergency hedging logic here

    async def _handle_portfolio_greeks(self, message: AgentMessage):
        """Handle portfolio Greeks calculation requests"""
        request = message.payload
        portfolio = request.get("portfolio", [])

        try:
            greeks_summary = await self._calculate_portfolio_greeks(portfolio)

            await self.send_message(
                target=message.source,
                message_type="portfolio_greeks_result",
                payload={
                    "request_id": request.get("request_id"),
                    "greeks": greeks_summary,
                    "timestamp": datetime.now().isoformat()
                }
            )

            self.performance_metrics["greeks_calculated"] += 1

        except Exception as e:
            self.logger.error(f"Portfolio Greeks calculation error: {e}")

    async def _calculate_portfolio_greeks(self, portfolio: List[Dict]) -> Dict[str, Any]:
        """Calculate aggregated Greeks for the entire portfolio"""

        total_delta = 0.0
        total_gamma = 0.0
        total_theta = 0.0
        total_vega = 0.0
        total_rho = 0.0

        position_greeks = []

        for position in portfolio:
            symbol = position["symbol"]
            quantity = position.get("quantity", 0)
            position_type = position.get("type", "stock")  # stock, option

            if position_type == "option":
                # Get option Greeks
                greeks = await self._calculate_option_greeks(position)
                if greeks:
                    # Scale by quantity
                    scaled_greeks = {
                        "delta": greeks["delta"] * quantity,
                        "gamma": greeks["gamma"] * quantity,
                        "theta": greeks["theta"] * quantity,
                        "vega": greeks["vega"] * quantity,
                        "rho": greeks["rho"] * quantity
                    }

                    total_delta += scaled_greeks["delta"]
                    total_gamma += scaled_greeks["gamma"]
                    total_theta += scaled_greeks["theta"]
                    total_vega += scaled_greeks["vega"]
                    total_rho += scaled_greeks["rho"]

                    position_greeks.append({
                        "symbol": symbol,
                        "quantity": quantity,
                        "greeks": scaled_greeks
                    })

            else:  # stock position
                # Stock has delta = 1, others = 0
                stock_delta = quantity
                total_delta += stock_delta

                position_greeks.append({
                    "symbol": symbol,
                    "quantity": quantity,
                    "greeks": {
                        "delta": stock_delta,
                        "gamma": 0,
                        "theta": 0,
                        "vega": 0,
                        "rho": 0
                    }
                })

        # Calculate portfolio-level metrics
        portfolio_value = await self._calculate_portfolio_value(portfolio)

        summary = {
            "total_greeks": {
                "delta": total_delta,
                "gamma": total_gamma,
                "theta": total_theta,
                "vega": total_vega,
                "rho": total_rho
            },
            "normalized_greeks": {  # Normalized by portfolio value
                "delta_pct": total_delta / portfolio_value * 100 if portfolio_value > 0 else 0,
                "gamma_pct": total_gamma / portfolio_value * 100 if portfolio_value > 0 else 0,
                "theta_daily": total_theta,  # Already in daily terms
                "vega_pct": total_vega / portfolio_value * 100 if portfolio_value > 0 else 0
            },
            "position_breakdown": position_greeks,
            "portfolio_value": portfolio_value,
            "risk_assessment": await self._assess_greeks_risk(total_delta, total_gamma, total_vega, portfolio_value)
        }

        return summary

    async def _calculate_option_greeks(self, option_position: Dict) -> Optional[Dict[str, float]]:
        """Calculate Greeks for a single option position"""
        try:
            symbol = option_position["symbol"]
            strike = option_position.get("strike")
            expiry = option_position.get("expiry")
            option_type = option_position.get("option_type", "call")

            # Send pricing request to quant modeler
            await self.send_message(
                target="quant_modeler",
                message_type="price_option",
                payload={
                    "request_id": f"greeks_{symbol}_{datetime.now().timestamp()}",
                    "symbol": symbol,
                    "strike": strike,
                    "expiry": expiry,
                    "option_type": option_type,
                    "model": "black_scholes"
                }
            )

            # For now, return approximate Greeks based on market data
            # In a real system, we'd wait for the response from quant_modeler
            market_data = self.data_connector.get_real_time_quote(symbol)
            if not market_data:
                return None

            # Simple Black-Scholes approximation
            return await self._approximate_greeks(market_data.price, strike, expiry, option_type)

        except Exception as e:
            self.logger.error(f"Error calculating option Greeks: {e}")
            return None

    async def _approximate_greeks(self, spot: float, strike: float, expiry: str, option_type: str) -> Dict[str, float]:
        """Quick Greeks approximation for real-time monitoring"""

        # Calculate time to expiry
        expiry_date = datetime.strptime(expiry, "%Y-%m-%d")
        time_to_expiry = max((expiry_date - datetime.now()).days / 365.0, 0.001)

        # Rough estimates for demonstration
        moneyness = spot / strike

        if option_type == "call":
            delta = 0.5 if abs(moneyness - 1) < 0.1 else (0.8 if moneyness > 1.1 else 0.2)
        else:
            delta = -0.5 if abs(moneyness - 1) < 0.1 else (-0.2 if moneyness > 1.1 else -0.8)

        gamma = 0.1 / (spot * 0.3 * np.sqrt(time_to_expiry))  # Approximate
        theta = -spot * 0.3 / (2 * np.sqrt(time_to_expiry)) / 365  # Daily theta
        vega = spot * np.sqrt(time_to_expiry) / 100  # Per 1% vol change
        rho = strike * time_to_expiry * 0.5 / 100  # Approximate

        return {
            "delta": delta,
            "gamma": gamma,
            "theta": theta,
            "vega": vega,
            "rho": rho
        }

    async def _calculate_portfolio_value(self, portfolio: List[Dict]) -> float:
        """Calculate total portfolio value"""
        total_value = 0.0

        for position in portfolio:
            symbol = position["symbol"]
            quantity = position.get("quantity", 0)

            market_data = self.data_connector.get_real_time_quote(symbol)
            if market_data:
                position_value = market_data.price * abs(quantity)
                total_value += position_value

        return total_value

    async def _assess_greeks_risk(self, delta: float, gamma: float, vega: float, portfolio_value: float) -> Dict[str, Any]:
        """Assess risk based on Greeks levels"""

        risk_score = 0
        alerts = []

        # Delta risk assessment
        delta_pct = abs(delta) / portfolio_value * 100 if portfolio_value > 0 else 0
        if delta_pct > 10:
            risk_score += 3
            alerts.append("High delta exposure detected")
        elif delta_pct > 5:
            risk_score += 1
            alerts.append("Moderate delta exposure")

        # Gamma risk assessment
        gamma_pct = abs(gamma) / portfolio_value * 100 if portfolio_value > 0 else 0
        if gamma_pct > 5:
            risk_score += 2
            alerts.append("High gamma exposure detected")

        # Vega risk assessment
        vega_pct = abs(vega) / portfolio_value * 100 if portfolio_value > 0 else 0
        if vega_pct > 15:
            risk_score += 2
            alerts.append("High vega exposure detected")

        risk_level = "low" if risk_score <= 1 else ("medium" if risk_score <= 3 else "high")

        return {
            "risk_score": risk_score,
            "risk_level": risk_level,
            "alerts": alerts,
            "recommendations": await self._generate_risk_recommendations(delta, gamma, vega, portfolio_value)
        }

    async def _generate_risk_recommendations(self, delta: float, gamma: float, vega: float, portfolio_value: float) -> List[str]:
        """Generate risk management recommendations"""
        recommendations = []

        delta_pct = delta / portfolio_value * 100 if portfolio_value > 0 else 0

        if abs(delta_pct) > 5:
            if delta_pct > 0:
                recommendations.append("Consider selling stock or buying puts to reduce positive delta")
            else:
                recommendations.append("Consider buying stock or selling puts to reduce negative delta")

        if abs(gamma) / portfolio_value * 100 > 3:
            recommendations.append("Monitor gamma exposure closely - consider delta hedging more frequently")

        if abs(vega) / portfolio_value * 100 > 10:
            recommendations.append("High volatility exposure - consider volatility hedging strategies")

        return recommendations

    async def _initialize_risk_limits(self):
        """Initialize risk limits for monitoring"""
        self.risk_limits = {
            "max_delta": RiskLimit("Max Delta", 0, 100000, 0, "green"),
            "max_gamma": RiskLimit("Max Gamma", 0, 10000, 0, "green"),
            "max_vega": RiskLimit("Max Vega", 0, 50000, 0, "green"),
            "max_theta": RiskLimit("Max Theta", 0, 5000, 0, "green"),
            "max_portfolio_var": RiskLimit("Max Portfolio VaR", 0, 100000, 0, "green")
        }

    async def _real_time_monitoring_loop(self):
        """Real-time monitoring loop for Greeks and risk limits"""
        while self.running:
            try:
                await self._monitor_portfolio_greeks()
                await self._check_risk_limits()
                await self._generate_hedge_recommendations()

                await asyncio.sleep(10)  # Monitor every 10 seconds

            except Exception as e:
                self.logger.error(f"Error in real-time monitoring: {e}")
                await asyncio.sleep(30)  # Wait longer on error

    async def _monitor_portfolio_greeks(self):
        """Monitor current portfolio Greeks"""
        if not self.current_positions:
            return

        try:
            portfolio = list(self.current_positions.values())
            greeks_summary = await self._calculate_portfolio_greeks(portfolio)

            # Store snapshot
            snapshot = GreeksSnapshot(
                symbol="PORTFOLIO",
                timestamp=datetime.now(),
                spot_price=0,  # Not applicable for portfolio
                delta=greeks_summary["total_greeks"]["delta"],
                gamma=greeks_summary["total_greeks"]["gamma"],
                theta=greeks_summary["total_greeks"]["theta"],
                vega=greeks_summary["total_greeks"]["vega"],
                rho=greeks_summary["total_greeks"]["rho"],
                total_delta=greeks_summary["total_greeks"]["delta"],
                total_gamma=greeks_summary["total_greeks"]["gamma"],
                total_vega=greeks_summary["total_greeks"]["vega"]
            )

            self.greeks_history.append(snapshot)

            # Keep only last 1000 snapshots
            if len(self.greeks_history) > 1000:
                self.greeks_history = self.greeks_history[-1000:]

        except Exception as e:
            self.logger.error(f"Error monitoring portfolio Greeks: {e}")

    async def _check_risk_limits(self):
        """Check current risk limits"""
        if not self.greeks_history:
            return

        latest_snapshot = self.greeks_history[-1]

        # Update risk limits
        self.risk_limits["max_delta"].current_value = abs(latest_snapshot.total_delta)
        self.risk_limits["max_gamma"].current_value = abs(latest_snapshot.total_gamma)
        self.risk_limits["max_vega"].current_value = abs(latest_snapshot.total_vega)

        # Check for breaches
        for limit_name, limit in self.risk_limits.items():
            utilization = limit.current_value / limit.limit_value if limit.limit_value > 0 else 0
            limit.utilization = utilization

            if utilization > 1.0:
                limit.status = "red"
                await self._send_risk_alert(limit_name, limit)
                self.performance_metrics["limit_breaches"] += 1
            elif utilization > 0.8:
                limit.status = "yellow"
            else:
                limit.status = "green"

    async def _send_risk_alert(self, limit_name: str, limit: RiskLimit):
        """Send risk alert for limit breach"""
        await self.send_message(
            target="chief_architect",
            message_type="risk_alert",
            payload={
                "alert_type": "limit_breach",
                "limit_name": limit_name,
                "current_value": limit.current_value,
                "limit_value": limit.limit_value,
                "utilization": limit.utilization,
                "severity": "critical" if limit.utilization > 1.2 else "high",
                "timestamp": datetime.now().isoformat()
            },
            priority=1
        )

        self.performance_metrics["risk_alerts"] += 1

    async def _generate_hedge_recommendations(self):
        """Generate hedging recommendations based on current Greeks"""
        if not self.hedging_enabled or not self.greeks_history:
            return

        latest_snapshot = self.greeks_history[-1]
        recommendations = []

        # Delta hedging
        if abs(latest_snapshot.total_delta) > 50:  # Threshold for delta hedging
            hedge_quantity = -int(latest_snapshot.total_delta)  # Opposite direction
            recommendations.append(HedgeRecommendation(
                symbol="SPY",  # Use SPY as hedging instrument
                action="buy" if hedge_quantity > 0 else "sell",
                quantity=abs(hedge_quantity),
                reasoning=f"Delta hedge to neutralize {latest_snapshot.total_delta:.2f} delta exposure",
                urgency="medium",
                expected_delta_change=-latest_snapshot.total_delta
            ))

        # Gamma hedging (more complex, simplified here)
        if abs(latest_snapshot.total_gamma) > 10:
            recommendations.append(HedgeRecommendation(
                symbol="SPY_OPTIONS",  # Would specify actual options
                action="buy" if latest_snapshot.total_gamma < 0 else "sell",
                quantity=int(abs(latest_snapshot.total_gamma) / 10),
                reasoning=f"Gamma hedge to manage {latest_snapshot.total_gamma:.2f} gamma exposure",
                urgency="low",
                expected_delta_change=0
            ))

        if recommendations:
            await self.send_message(
                target="chief_architect",
                message_type="hedge_recommendations",
                payload={
                    "recommendations": [asdict(rec) for rec in recommendations],
                    "timestamp": datetime.now().isoformat()
                }
            )

            self.performance_metrics["hedge_recommendations"] += len(recommendations)

    async def cleanup(self):
        """Cleanup resources"""
        self.logger.info("Cleaning up Greeks & Risk Agent...")

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
            "risk_limits": {name: asdict(limit) for name, limit in self.risk_limits.items()},
            "positions_monitored": len(self.current_positions),
            "healthy": True
        }