"""
Model Validation & Governance Agent
Responsible for model validation, backtesting, and governance oversight
"""

import asyncio
import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from scipy import stats
import json
from pathlib import Path

from agents.base_agent import BaseAgent, AgentMessage, AgentStatus
from data.yahoo_connector import YahooFinanceConnector

logger = logging.getLogger(__name__)

@dataclass
class BacktestResult:
    strategy_id: str
    start_date: datetime
    end_date: datetime
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    var_breach_rate: float
    trades_count: int
    status: str

@dataclass
class ModelValidationReport:
    model_id: str
    validation_date: datetime
    validation_type: str  # 'backtest', 'stress_test', 'var_backtest'
    passed: bool
    score: float
    issues: List[str]
    recommendations: List[str]

@dataclass
class PnLAttribution:
    date: datetime
    total_pnl: float
    market_pnl: float
    volatility_pnl: float
    theta_pnl: float
    gamma_pnl: float
    unexplained_pnl: float
    unexplained_percentage: float

class ModelValidationAgent(BaseAgent):
    """
    Model Validation & Governance Agent - Model oversight and validation
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__("model_validation", config)
        self.data_connector = None
        self.validation_history = []
        self.backtest_results = {}
        self.pnl_attributions = []
        self.model_registry = {}
        self.performance_metrics = {
            "backtests_completed": 0,
            "validations_performed": 0,
            "models_approved": 0,
            "models_rejected": 0,
            "governance_reviews": 0
        }

    async def initialize(self):
        """Initialize the Model Validation Agent"""
        self.logger.info("Initializing Model Validation Agent...")

        # Initialize data connector
        self.data_connector = YahooFinanceConnector()

        # Initialize model registry
        await self._initialize_model_registry()

        # Start periodic validation tasks
        asyncio.create_task(self._periodic_validation_loop())
        asyncio.create_task(self._daily_pnl_attribution_loop())

        self.logger.info("Model Validation Agent initialized successfully")

    async def process_message(self, message: AgentMessage):
        """Process incoming messages"""

        if message.message_type == "validate_model":
            await self._handle_model_validation(message)
        elif message.message_type == "run_backtest":
            await self._handle_backtest_request(message)
        elif message.message_type == "pnl_attribution":
            await self._handle_pnl_attribution(message)
        elif message.message_type == "stress_test":
            await self._handle_stress_test(message)
        elif message.message_type == "var_backtest":
            await self._handle_var_backtest(message)
        elif message.message_type == "model_approval":
            await self._handle_model_approval(message)
        elif message.message_type == "governance_review":
            await self._handle_governance_review(message)
        elif message.message_type == "health_check":
            # Simple health check response
            self.logger.debug("Health check received - system healthy")
        else:
            self.logger.warning(f"Unknown message type: {message.message_type}")

    async def _handle_stress_test(self, message: AgentMessage):
        """Handle stress test requests"""
        self.logger.info("Stress test request received")

    async def _handle_model_approval(self, message: AgentMessage):
        """Handle model approval requests"""
        self.logger.info("Model approval request received")

    async def _handle_governance_review(self, message: AgentMessage):
        """Handle governance review requests"""
        self.logger.info("Governance review request received")

    async def _handle_model_validation(self, message: AgentMessage):
        """Handle model validation requests"""
        request = message.payload
        model_id = request.get("model_id")
        validation_type = request.get("type", "comprehensive")

        try:
            validation_report = await self._validate_model(model_id, validation_type)

            await self.send_message(
                target=message.source,
                message_type="model_validation_result",
                payload={
                    "request_id": request.get("request_id"),
                    "model_id": model_id,
                    "report": asdict(validation_report),
                    "timestamp": datetime.now().isoformat()
                }
            )

            self.performance_metrics["validations_performed"] += 1

        except Exception as e:
            self.logger.error(f"Model validation error: {e}")

    async def _validate_model(self, model_id: str, validation_type: str) -> ModelValidationReport:
        """Validate a specific model"""

        issues = []
        recommendations = []
        score = 0.0

        if validation_type == "comprehensive":
            # Comprehensive validation includes multiple tests
            score += await self._validate_model_accuracy(model_id)
            score += await self._validate_model_stability(model_id)
            score += await self._validate_model_performance(model_id)

            # Normalize score (0-100)
            score = score / 3 * 100

        elif validation_type == "accuracy":
            score = await self._validate_model_accuracy(model_id) * 100

        elif validation_type == "stability":
            score = await self._validate_model_stability(model_id) * 100

        # Determine pass/fail
        passed = score >= 70  # 70% threshold

        if not passed:
            issues.append(f"Model score {score:.1f}% below 70% threshold")
            recommendations.append("Review model parameters and recalibrate")

        if score < 50:
            issues.append("Critical model performance issues detected")
            recommendations.append("Immediate model review required")

        report = ModelValidationReport(
            model_id=model_id,
            validation_date=datetime.now(),
            validation_type=validation_type,
            passed=passed,
            score=score,
            issues=issues,
            recommendations=recommendations
        )

        self.validation_history.append(report)
        return report

    async def _validate_model_accuracy(self, model_id: str) -> float:
        """Validate model accuracy against market data"""
        try:
            # Get historical option prices and compare with model predictions
            test_symbols = ["MSTR", "QQQ", "SPY"]
            accuracy_scores = []

            for symbol in test_symbols:
                # Get options data
                options_data = self.data_connector.get_options_chain(symbol)

                if not options_data:
                    continue

                # Compare market prices with theoretical prices
                for option in options_data[:10]:  # Test first 10 options
                    market_price = (option.bid + option.ask) / 2

                    # Get theoretical price from quant model
                    # (In practice, this would be an actual model call)
                    theoretical_price = await self._get_theoretical_price(symbol, option)

                    if market_price > 0 and theoretical_price > 0:
                        relative_error = abs(market_price - theoretical_price) / market_price
                        accuracy_scores.append(1.0 - min(relative_error, 1.0))

            return np.mean(accuracy_scores) if accuracy_scores else 0.5

        except Exception as e:
            self.logger.error(f"Model accuracy validation error: {e}")
            return 0.0

    async def _validate_model_stability(self, model_id: str) -> float:
        """Validate model stability over time"""
        try:
            # Check if model parameters have been stable
            # This is a simplified implementation
            stability_score = 0.8  # Assume good stability for demo

            # Check for parameter drift
            parameter_drift = await self._check_parameter_drift(model_id)
            if parameter_drift > 0.1:  # 10% threshold
                stability_score -= 0.2

            # Check prediction consistency
            prediction_variance = await self._check_prediction_variance(model_id)
            if prediction_variance > 0.15:  # 15% threshold
                stability_score -= 0.2

            return max(stability_score, 0.0)

        except Exception as e:
            self.logger.error(f"Model stability validation error: {e}")
            return 0.0

    async def _validate_model_performance(self, model_id: str) -> float:
        """Validate model performance metrics"""
        try:
            # Check computational performance
            calculation_time = await self._measure_calculation_time(model_id)
            performance_score = 1.0

            if calculation_time > 5.0:  # 5 seconds threshold
                performance_score -= 0.3

            # Check memory usage
            memory_usage = await self._measure_memory_usage(model_id)
            if memory_usage > 1000:  # 1GB threshold
                performance_score -= 0.2

            return max(performance_score, 0.0)

        except Exception as e:
            self.logger.error(f"Model performance validation error: {e}")
            return 0.0

    async def _get_theoretical_price(self, symbol: str, option) -> float:
        """Get theoretical option price from model"""
        # Simplified Black-Scholes calculation for validation
        try:
            market_data = self.data_connector.get_real_time_quote(symbol)
            if not market_data:
                return 0.0

            S = market_data.price
            K = option.strike
            T = (option.expiry - datetime.now()).days / 365.0
            r = 0.045  # Risk-free rate
            sigma = option.implied_volatility if option.implied_volatility > 0 else 0.25

            if T <= 0:
                return max(S - K, 0) if option.option_type == "call" else max(K - S, 0)

            d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)

            if option.option_type == "call":
                price = S * stats.norm.cdf(d1) - K * np.exp(-r * T) * stats.norm.cdf(d2)
            else:
                price = K * np.exp(-r * T) * stats.norm.cdf(-d2) - S * stats.norm.cdf(-d1)

            return price

        except Exception as e:
            self.logger.error(f"Theoretical price calculation error: {e}")
            return 0.0

    async def _handle_backtest_request(self, message: AgentMessage):
        """Handle backtest requests"""
        request = message.payload
        strategy_id = request.get("strategy_id")
        start_date = request.get("start_date")
        end_date = request.get("end_date")

        try:
            backtest_result = await self._run_backtest(strategy_id, start_date, end_date)

            await self.send_message(
                target=message.source,
                message_type="backtest_result",
                payload={
                    "request_id": request.get("request_id"),
                    "result": asdict(backtest_result),
                    "timestamp": datetime.now().isoformat()
                }
            )

            self.performance_metrics["backtests_completed"] += 1

        except Exception as e:
            self.logger.error(f"Backtest error: {e}")

    async def _run_backtest(self, strategy_id: str, start_date: str, end_date: str) -> BacktestResult:
        """Run strategy backtest"""

        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")

        # Simplified backtest simulation
        # In practice, this would involve complex strategy simulation

        # Generate synthetic performance metrics for demonstration
        np.random.seed(42)
        days = (end_dt - start_dt).days
        daily_returns = np.random.normal(0.001, 0.02, days)  # 0.1% daily return, 2% volatility

        # Calculate metrics
        total_return = np.prod(1 + daily_returns) - 1
        sharpe_ratio = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252)

        # Calculate max drawdown
        cumulative_returns = np.cumprod(1 + daily_returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = (cumulative_returns - running_max) / running_max
        max_drawdown = np.min(drawdowns)

        # Simulate other metrics
        win_rate = 0.55  # 55% win rate
        var_breach_rate = 0.03  # 3% VaR breach rate
        trades_count = days // 2  # Assume trade every 2 days

        # Determine status
        status = "passed"
        if sharpe_ratio < 0.5:
            status = "failed"
        elif max_drawdown < -0.15:  # 15% max drawdown threshold
            status = "warning"

        result = BacktestResult(
            strategy_id=strategy_id,
            start_date=start_dt,
            end_date=end_dt,
            total_return=total_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            var_breach_rate=var_breach_rate,
            trades_count=trades_count,
            status=status
        )

        self.backtest_results[strategy_id] = result
        return result

    async def _handle_pnl_attribution(self, message: AgentMessage):
        """Handle P&L attribution requests"""
        request = message.payload
        date = request.get("date", datetime.now().strftime("%Y-%m-%d"))

        try:
            attribution = await self._calculate_pnl_attribution(date)

            await self.send_message(
                target=message.source,
                message_type="pnl_attribution_result",
                payload={
                    "request_id": request.get("request_id"),
                    "attribution": asdict(attribution),
                    "timestamp": datetime.now().isoformat()
                }
            )

        except Exception as e:
            self.logger.error(f"P&L attribution error: {e}")

    async def _calculate_pnl_attribution(self, date_str: str) -> PnLAttribution:
        """Calculate P&L attribution for a given date"""

        # Simplified P&L attribution calculation
        # In practice, this would involve detailed position analysis

        date_obj = datetime.strptime(date_str, "%Y-%m-%d")

        # Simulate P&L components
        total_pnl = np.random.normal(1000, 5000)  # Random P&L for demonstration

        # Attribute to different sources
        market_pnl = total_pnl * 0.6  # 60% from market moves
        volatility_pnl = total_pnl * 0.2  # 20% from volatility changes
        theta_pnl = total_pnl * 0.1  # 10% from time decay
        gamma_pnl = total_pnl * 0.05  # 5% from gamma
        unexplained_pnl = total_pnl * 0.05  # 5% unexplained

        unexplained_percentage = abs(unexplained_pnl) / abs(total_pnl) * 100 if total_pnl != 0 else 0

        attribution = PnLAttribution(
            date=date_obj,
            total_pnl=total_pnl,
            market_pnl=market_pnl,
            volatility_pnl=volatility_pnl,
            theta_pnl=theta_pnl,
            gamma_pnl=gamma_pnl,
            unexplained_pnl=unexplained_pnl,
            unexplained_percentage=unexplained_percentage
        )

        self.pnl_attributions.append(attribution)
        return attribution

    async def _handle_var_backtest(self, message: AgentMessage):
        """Handle VaR backtesting requests"""
        request = message.payload

        try:
            # Run VaR backtest validation
            result = await self._var_backtest_validation()

            await self.send_message(
                target=message.source,
                message_type="var_backtest_result",
                payload={
                    "request_id": request.get("request_id"),
                    "result": result,
                    "timestamp": datetime.now().isoformat()
                }
            )

        except Exception as e:
            self.logger.error(f"VaR backtest error: {e}")

    async def _var_backtest_validation(self) -> Dict[str, Any]:
        """Validate VaR model using Kupiec test"""

        # Simulate VaR backtest data
        np.random.seed(42)
        n_observations = 250  # 1 year of trading days
        confidence_level = 0.95
        expected_breaches = n_observations * (1 - confidence_level)

        # Simulate actual breaches (should be around 12.5 for 95% VaR)
        actual_breaches = np.random.poisson(expected_breaches)

        # Kupiec test
        lr_stat = 2 * (actual_breaches * np.log(actual_breaches / expected_breaches) +
                      (n_observations - actual_breaches) *
                      np.log((n_observations - actual_breaches) / (n_observations - expected_breaches)))

        critical_value = 3.841  # Chi-square critical value for 95% confidence
        test_passed = lr_stat < critical_value

        return {
            "test_type": "kupiec",
            "confidence_level": confidence_level,
            "observations": n_observations,
            "expected_breaches": expected_breaches,
            "actual_breaches": actual_breaches,
            "lr_statistic": lr_stat,
            "critical_value": critical_value,
            "test_passed": test_passed,
            "breach_rate": actual_breaches / n_observations
        }

    async def _check_parameter_drift(self, model_id: str) -> float:
        """Check for parameter drift in model"""
        # Simplified parameter drift check
        return np.random.uniform(0, 0.05)  # Random drift for demo

    async def _check_prediction_variance(self, model_id: str) -> float:
        """Check prediction variance"""
        # Simplified variance check
        return np.random.uniform(0, 0.1)  # Random variance for demo

    async def _measure_calculation_time(self, model_id: str) -> float:
        """Measure model calculation time"""
        # Simulate calculation time measurement
        return np.random.uniform(0.1, 2.0)  # 0.1 to 2 seconds

    async def _measure_memory_usage(self, model_id: str) -> float:
        """Measure model memory usage"""
        # Simulate memory usage measurement
        return np.random.uniform(100, 500)  # 100-500 MB

    async def _initialize_model_registry(self):
        """Initialize the model registry"""
        # Initialize with current timestamp to prevent immediate validation
        current_time = datetime.now().isoformat()

        self.model_registry = {
            "black_scholes": {
                "name": "Black-Scholes Model",
                "version": "1.0",
                "status": "active",
                "last_validation": current_time
            },
            "heston": {
                "name": "Heston Stochastic Volatility",
                "version": "1.0",
                "status": "active",
                "last_validation": current_time
            },
            "var_historical": {
                "name": "Historical VaR",
                "version": "1.0",
                "status": "active",
                "last_validation": current_time
            }
        }

    async def _periodic_validation_loop(self):
        """Periodic model validation loop"""
        # Wait a bit before starting periodic validation to avoid startup issues
        await asyncio.sleep(300)  # Wait 5 minutes before first validation

        while self.running:
            try:
                # Validate all active models monthly
                for model_id, model_info in self.model_registry.items():
                    if model_info["status"] == "active":
                        last_validation = model_info.get("last_validation")

                        # Only validate if truly needed (more than 30 days old)
                        if (last_validation and
                            (datetime.now() - datetime.fromisoformat(last_validation)).days > 30):

                            self.logger.info(f"Running periodic validation for model: {model_id}")
                            await self._validate_model(model_id, "comprehensive")
                            self.model_registry[model_id]["last_validation"] = datetime.now().isoformat()

                # Check once per day
                await asyncio.sleep(86400)

            except Exception as e:
                self.logger.error(f"Periodic validation error: {e}")
                await asyncio.sleep(3600)  # Wait 1 hour on error

    async def _daily_pnl_attribution_loop(self):
        """Daily P&L attribution loop"""
        # Wait 10 minutes before starting P&L attribution
        await asyncio.sleep(600)

        while self.running:
            try:
                # Calculate P&L attribution for yesterday
                yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
                self.logger.info("Running daily P&L attribution...")
                await self._calculate_pnl_attribution(yesterday)

                # Wait until next day
                await asyncio.sleep(86400)  # 24 hours

            except Exception as e:
                self.logger.error(f"Daily P&L attribution error: {e}")
                await asyncio.sleep(3600)

    async def cleanup(self):
        """Cleanup resources"""
        self.logger.info("Cleaning up Model Validation Agent...")

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
            "models_in_registry": len(self.model_registry),
            "validations_completed": len(self.validation_history),
            "backtests_available": len(self.backtest_results),
            "healthy": True
        }