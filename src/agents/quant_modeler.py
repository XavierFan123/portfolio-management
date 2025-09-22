"""
Quantitative Modeling Agent
Responsible for risk models, option pricing, and quantitative analysis
"""

import asyncio
import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from scipy import stats
from scipy.optimize import minimize
import math

from agents.base_agent import BaseAgent, AgentMessage, AgentStatus
from data.yahoo_connector import YahooFinanceConnector

logger = logging.getLogger(__name__)

class QuantitativeModelerAgent(BaseAgent):
    """
    Quantitative Modeler Agent - Advanced risk modeling and option pricing
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__("quant_modeler", config)
        self.data_connector = None
        self.models = {}
        self.calibrated_parameters = {}
        self.performance_metrics = {
            "models_calibrated": 0,
            "pricing_calculations": 0,
            "risk_calculations": 0,
            "backtests_performed": 0
        }

    async def initialize(self):
        """Initialize the Quantitative Modeler Agent"""
        self.logger.info("Initializing Quantitative Modeler Agent...")

        # Initialize data connector
        self.data_connector = YahooFinanceConnector()

        # Initialize models
        await self._initialize_models()

        self.logger.info("Quantitative Modeler Agent initialized successfully")

    async def process_message(self, message: AgentMessage):
        """Process incoming messages"""
        self.performance_metrics["pricing_calculations"] += 1

        if message.message_type == "price_option":
            await self._handle_option_pricing(message)
        elif message.message_type == "calculate_greeks":
            await self._handle_greeks_calculation(message)
        elif message.message_type == "calculate_var":
            await self._handle_var_calculation(message)
        elif message.message_type == "calibrate_model":
            await self._handle_model_calibration(message)
        elif message.message_type == "scenario_analysis":
            await self._handle_scenario_analysis(message)
        elif message.message_type == "volatility_surface":
            await self._handle_volatility_surface(message)
        elif message.message_type == "health_check":
            # Simple health check response
            self.logger.debug("Health check received - system healthy")
        else:
            self.logger.warning(f"Unknown message type: {message.message_type}")

    async def _handle_greeks_calculation(self, message: AgentMessage):
        """Handle Greeks calculation requests"""
        self.logger.info("Greeks calculation request received")

    async def _handle_model_calibration(self, message: AgentMessage):
        """Handle model calibration requests"""
        self.logger.info("Model calibration request received")

    async def _handle_scenario_analysis(self, message: AgentMessage):
        """Handle scenario analysis requests"""
        self.logger.info("Scenario analysis request received")

    async def _handle_volatility_surface(self, message: AgentMessage):
        """Handle volatility surface requests"""
        self.logger.info("Volatility surface request received")

    async def _handle_option_pricing(self, message: AgentMessage):
        """Handle option pricing requests"""
        request = message.payload

        symbol = request.get("symbol")
        strike = request.get("strike")
        expiry = request.get("expiry")
        option_type = request.get("option_type", "call")
        model_type = request.get("model", "black_scholes")

        try:
            # Get market data
            market_data = self.data_connector.get_real_time_quote(symbol)
            if not market_data:
                raise ValueError(f"No market data for {symbol}")

            # Calculate option price
            price_result = await self._price_option(
                spot=market_data.price,
                strike=strike,
                expiry=expiry,
                option_type=option_type,
                model_type=model_type,
                symbol=symbol
            )

            await self.send_message(
                target=message.source,
                message_type="option_price_result",
                payload={
                    "request_id": request.get("request_id"),
                    "symbol": symbol,
                    "price": price_result["price"],
                    "greeks": price_result["greeks"],
                    "model_used": model_type,
                    "timestamp": datetime.now().isoformat()
                }
            )

        except Exception as e:
            self.logger.error(f"Option pricing error: {e}")
            await self.send_message(
                target=message.source,
                message_type="option_price_error",
                payload={"request_id": request.get("request_id"), "error": str(e)}
            )

    async def _price_option(self, spot: float, strike: float, expiry: str,
                           option_type: str, model_type: str, symbol: str) -> Dict[str, Any]:
        """Price an option using specified model"""

        # Calculate time to expiry
        expiry_date = datetime.strptime(expiry, "%Y-%m-%d")
        time_to_expiry = (expiry_date - datetime.now()).days / 365.0

        if time_to_expiry <= 0:
            return {"price": 0.0, "greeks": {}}

        # Get risk-free rate (using 10Y treasury as proxy)
        risk_free_rate = 0.045  # 4.5% default

        # Get volatility
        volatility = await self._get_implied_volatility(symbol, strike, expiry, option_type)
        if volatility is None:
            # Fall back to historical volatility
            volatility = self.data_connector.get_volatility_data(symbol, 30)
            if volatility is None:
                volatility = 0.25  # Default 25%

        if model_type == "black_scholes":
            return self._black_scholes_price(spot, strike, time_to_expiry, risk_free_rate, volatility, option_type)
        elif model_type == "heston":
            return await self._heston_price(spot, strike, time_to_expiry, risk_free_rate, symbol, option_type)
        elif model_type == "jump_diffusion":
            return await self._jump_diffusion_price(spot, strike, time_to_expiry, risk_free_rate, volatility, symbol, option_type)
        else:
            # Default to Black-Scholes
            return self._black_scholes_price(spot, strike, time_to_expiry, risk_free_rate, volatility, option_type)

    def _black_scholes_price(self, S: float, K: float, T: float, r: float, sigma: float, option_type: str) -> Dict[str, Any]:
        """Black-Scholes option pricing"""

        if T <= 0:
            intrinsic = max(S - K, 0) if option_type == "call" else max(K - S, 0)
            return {"price": intrinsic, "greeks": {}}

        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        if option_type == "call":
            price = S * stats.norm.cdf(d1) - K * np.exp(-r * T) * stats.norm.cdf(d2)
            delta = stats.norm.cdf(d1)
        else:  # put
            price = K * np.exp(-r * T) * stats.norm.cdf(-d2) - S * stats.norm.cdf(-d1)
            delta = -stats.norm.cdf(-d1)

        # Calculate Greeks
        gamma = stats.norm.pdf(d1) / (S * sigma * np.sqrt(T))
        theta = (-S * stats.norm.pdf(d1) * sigma / (2 * np.sqrt(T))
                 - r * K * np.exp(-r * T) * stats.norm.cdf(d2 if option_type == "call" else -d2))
        vega = S * stats.norm.pdf(d1) * np.sqrt(T)
        rho = K * T * np.exp(-r * T) * stats.norm.cdf(d2 if option_type == "call" else -d2)

        greeks = {
            "delta": delta,
            "gamma": gamma,
            "theta": theta / 365,  # Per day
            "vega": vega / 100,    # Per 1% volatility change
            "rho": rho / 100       # Per 1% rate change
        }

        return {"price": price, "greeks": greeks}

    async def _heston_price(self, S: float, K: float, T: float, r: float, symbol: str, option_type: str) -> Dict[str, Any]:
        """Heston stochastic volatility model pricing"""

        # Get or calibrate Heston parameters
        params = await self._get_heston_parameters(symbol)

        # Monte Carlo simulation for Heston model
        n_simulations = 10000
        n_steps = int(T * 252)  # Daily steps
        dt = T / n_steps

        # Initialize arrays
        prices = np.zeros((n_simulations, n_steps + 1))
        volatilities = np.zeros((n_simulations, n_steps + 1))

        prices[:, 0] = S
        volatilities[:, 0] = params["v0"]

        # Random numbers
        np.random.seed(42)
        z1 = np.random.normal(0, 1, (n_simulations, n_steps))
        z2 = np.random.normal(0, 1, (n_simulations, n_steps))
        z2_corr = params["rho"] * z1 + np.sqrt(1 - params["rho"]**2) * z2

        # Simulate paths
        for i in range(n_steps):
            volatilities[:, i + 1] = np.maximum(
                volatilities[:, i] + params["kappa"] * (params["theta"] - volatilities[:, i]) * dt +
                params["sigma_v"] * np.sqrt(volatilities[:, i] * dt) * z2_corr[:, i],
                0.001  # Floor volatility
            )

            prices[:, i + 1] = prices[:, i] * np.exp(
                (r - 0.5 * volatilities[:, i]) * dt +
                np.sqrt(volatilities[:, i] * dt) * z1[:, i]
            )

        # Calculate payoffs
        if option_type == "call":
            payoffs = np.maximum(prices[:, -1] - K, 0)
        else:
            payoffs = np.maximum(K - prices[:, -1], 0)

        price = np.exp(-r * T) * np.mean(payoffs)

        # Approximate Greeks using finite differences
        greeks = await self._calculate_heston_greeks(S, K, T, r, params, option_type)

        return {"price": price, "greeks": greeks}

    async def _jump_diffusion_price(self, S: float, K: float, T: float, r: float, sigma: float,
                                  symbol: str, option_type: str) -> Dict[str, Any]:
        """Merton jump diffusion model pricing"""

        # Get jump parameters
        jump_params = await self._get_jump_parameters(symbol)

        lambda_j = jump_params["intensity"]  # Jump intensity
        mu_j = jump_params["mean_jump"]      # Mean jump size
        sigma_j = jump_params["jump_vol"]    # Jump volatility

        # Merton model - sum over possible number of jumps
        max_jumps = 20
        price = 0.0

        for n in range(max_jumps):
            # Probability of n jumps
            poisson_prob = np.exp(-lambda_j * T) * (lambda_j * T)**n / math.factorial(n)

            # Adjusted parameters for n jumps
            sigma_n = np.sqrt(sigma**2 + n * sigma_j**2 / T)
            r_n = r - lambda_j * (np.exp(mu_j + 0.5 * sigma_j**2) - 1) + n * np.log(1 + mu_j) / T

            # Black-Scholes price with adjusted parameters
            bs_price = self._black_scholes_price(S, K, T, r_n, sigma_n, option_type)
            price += poisson_prob * bs_price["price"]

        # Calculate approximate Greeks
        greeks = self._black_scholes_price(S, K, T, r, sigma, option_type)["greeks"]

        return {"price": price, "greeks": greeks}

    async def _get_implied_volatility(self, symbol: str, strike: float, expiry: str, option_type: str) -> Optional[float]:
        """Get implied volatility from options data"""
        try:
            options_data = self.data_connector.get_options_chain(symbol, expiry)

            for option in options_data:
                if (option.strike == strike and
                    option.option_type == option_type and
                    option.implied_volatility > 0):
                    return option.implied_volatility

            return None

        except Exception as e:
            self.logger.error(f"Error getting implied volatility: {e}")
            return None

    async def _get_heston_parameters(self, symbol: str) -> Dict[str, float]:
        """Get or calibrate Heston model parameters"""

        if symbol in self.calibrated_parameters.get("heston", {}):
            return self.calibrated_parameters["heston"][symbol]

        # Default Heston parameters
        default_params = {
            "v0": 0.04,      # Initial variance
            "kappa": 2.0,    # Mean reversion speed
            "theta": 0.04,   # Long-term variance
            "sigma_v": 0.3,  # Volatility of volatility
            "rho": -0.7      # Correlation between price and volatility
        }

        # Store for future use
        if "heston" not in self.calibrated_parameters:
            self.calibrated_parameters["heston"] = {}
        self.calibrated_parameters["heston"][symbol] = default_params

        return default_params

    async def _get_jump_parameters(self, symbol: str) -> Dict[str, float]:
        """Get jump diffusion parameters"""

        # Special parameters for crypto-related stocks like MSTR
        if symbol == "MSTR":
            return {
                "intensity": 2.0,    # 2 jumps per year on average
                "mean_jump": -0.02,  # -2% mean jump size
                "jump_vol": 0.15     # 15% jump volatility
            }
        else:
            return {
                "intensity": 0.5,    # 0.5 jumps per year
                "mean_jump": -0.01,  # -1% mean jump size
                "jump_vol": 0.05     # 5% jump volatility
            }

    async def _calculate_heston_greeks(self, S: float, K: float, T: float, r: float,
                                     params: Dict[str, float], option_type: str) -> Dict[str, float]:
        """Calculate Greeks for Heston model using finite differences"""

        # Small bumps for finite differences
        dS = S * 0.01
        dT = T * 0.01
        dv = params["v0"] * 0.01

        # Base price
        base_price = await self._heston_price(S, K, T, r, "SPY", option_type)  # Use SPY as placeholder

        # Delta (price sensitivity)
        up_price = await self._heston_price(S + dS, K, T, r, "SPY", option_type)
        delta = (up_price["price"] - base_price["price"]) / dS

        # Gamma (delta sensitivity)
        down_price = await self._heston_price(S - dS, K, T, r, "SPY", option_type)
        gamma = (up_price["price"] - 2 * base_price["price"] + down_price["price"]) / (dS**2)

        # Theta (time decay)
        theta_price = await self._heston_price(S, K, T - dT, r, "SPY", option_type)
        theta = (theta_price["price"] - base_price["price"]) / dT

        return {
            "delta": delta,
            "gamma": gamma,
            "theta": theta / 365,
            "vega": 0.0,  # Would need vol bump
            "rho": 0.0    # Would need rate bump
        }

    async def _handle_var_calculation(self, message: AgentMessage):
        """Handle VaR calculation requests"""
        request = message.payload

        portfolio = request.get("portfolio", [])
        confidence_level = request.get("confidence", 0.95)
        time_horizon = request.get("horizon", 1)  # days

        try:
            var_result = await self._calculate_portfolio_var(portfolio, confidence_level, time_horizon)

            await self.send_message(
                target=message.source,
                message_type="var_result",
                payload={
                    "request_id": request.get("request_id"),
                    "var": var_result["var"],
                    "expected_shortfall": var_result["es"],
                    "methodology": var_result["method"],
                    "timestamp": datetime.now().isoformat()
                }
            )

        except Exception as e:
            self.logger.error(f"VaR calculation error: {e}")

    async def _calculate_portfolio_var(self, portfolio: List[Dict], confidence: float, horizon: int) -> Dict[str, Any]:
        """Calculate portfolio Value at Risk"""

        # Get historical data for all symbols
        returns_data = {}
        for position in portfolio:
            symbol = position["symbol"]
            data = self.data_connector.get_stock_data(symbol, period="1y", interval="1d")
            if not data.empty:
                returns = data['Close'].pct_change().dropna()
                returns_data[symbol] = returns

        if not returns_data:
            raise ValueError("No historical data available for VaR calculation")

        # Create portfolio returns
        portfolio_returns = []
        weights = {pos["symbol"]: pos.get("weight", 1.0) for pos in portfolio}

        # Align data and calculate portfolio returns
        common_dates = None
        for symbol, returns in returns_data.items():
            if common_dates is None:
                common_dates = returns.index
            else:
                common_dates = common_dates.intersection(returns.index)

        for date in common_dates:
            portfolio_return = sum(
                weights[symbol] * returns_data[symbol][date]
                for symbol in weights.keys()
                if date in returns_data[symbol].index
            )
            portfolio_returns.append(portfolio_return)

        portfolio_returns = np.array(portfolio_returns)

        # Scale for time horizon
        portfolio_returns = portfolio_returns * np.sqrt(horizon)

        # Calculate VaR and Expected Shortfall
        var = -np.percentile(portfolio_returns, (1 - confidence) * 100)
        es = -np.mean(portfolio_returns[portfolio_returns <= -var])

        return {
            "var": var,
            "es": es,
            "method": "historical_simulation",
            "observations": len(portfolio_returns)
        }

    async def _initialize_models(self):
        """Initialize quantitative models"""
        self.models = {
            "black_scholes": True,
            "heston": True,
            "jump_diffusion": True,
            "var_models": ["historical_simulation", "monte_carlo", "parametric"]
        }

    async def cleanup(self):
        """Cleanup resources"""
        self.logger.info("Cleaning up Quantitative Modeler Agent...")

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
            "models_available": list(self.models.keys()),
            "healthy": True
        }