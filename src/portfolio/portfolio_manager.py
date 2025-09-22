"""
Portfolio Manager
Comprehensive portfolio management with position input, VaR calculation, and risk analysis
"""

import json
import csv
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import logging

from data.yahoo_connector import YahooFinanceConnector

logger = logging.getLogger(__name__)

@dataclass
class Position:
    symbol: str
    quantity: float
    entry_price: float
    position_type: str  # 'stock', 'option', 'crypto'
    entry_date: str
    current_price: Optional[float] = None
    market_value: Optional[float] = None
    unrealized_pnl: Optional[float] = None
    weight: Optional[float] = None

    # Option-specific fields
    strike_price: Optional[float] = None
    expiry_date: Optional[str] = None
    option_type: Optional[str] = None  # 'call', 'put'

@dataclass
class PortfolioSummary:
    total_value: float
    total_positions: int
    cash_balance: float
    daily_pnl: float
    total_return: float
    largest_position: str
    largest_position_weight: float

@dataclass
class VaRAnalysis:
    var_1d_95: float
    var_1d_99: float
    var_10d_95: float
    expected_shortfall_95: float
    max_drawdown: float
    portfolio_volatility: float
    sharpe_ratio: float
    component_var: Dict[str, float]
    methodology: str

class PortfolioManager:
    """Comprehensive portfolio management system"""

    def __init__(self, data_connector: Optional[YahooFinanceConnector] = None):
        self.data_connector = data_connector or YahooFinanceConnector()
        self.positions: Dict[str, Position] = {}
        self.cash_balance = 0.0
        self.portfolio_file = "data/portfolio.json"
        self.ensure_data_directory()

    def ensure_data_directory(self):
        """Ensure data directory exists"""
        Path("data").mkdir(exist_ok=True)

    def add_position(self, symbol: str, quantity: float, entry_price: float,
                    position_type: str = "stock", **kwargs) -> bool:
        """Add a new position to the portfolio"""
        try:
            # Create position
            position = Position(
                symbol=symbol.upper(),
                quantity=quantity,
                entry_price=entry_price,
                position_type=position_type,
                entry_date=datetime.now().strftime("%Y-%m-%d"),
                **kwargs
            )

            # Update current market data
            self._update_position_market_data(position)

            # Add to portfolio
            position_key = self._get_position_key(position)
            self.positions[position_key] = position

            logger.info(f"Added position: {symbol} {quantity} shares at ${entry_price}")
            return True

        except Exception as e:
            logger.error(f"Error adding position {symbol}: {e}")
            return False

    def remove_position(self, symbol: str, option_details: Optional[Dict] = None) -> bool:
        """Remove a position from the portfolio"""
        try:
            position_key = symbol.upper()
            if option_details:
                position_key = f"{symbol.upper()}_{option_details.get('strike')}_{option_details.get('expiry')}_{option_details.get('type')}"

            if position_key in self.positions:
                removed_position = self.positions.pop(position_key)
                logger.info(f"Removed position: {removed_position.symbol}")
                return True
            else:
                logger.warning(f"Position {position_key} not found")
                return False

        except Exception as e:
            logger.error(f"Error removing position {symbol}: {e}")
            return False

    def update_position_quantity(self, symbol: str, new_quantity: float) -> bool:
        """Update the quantity of an existing position"""
        try:
            position_key = symbol.upper()
            if position_key in self.positions:
                old_quantity = self.positions[position_key].quantity
                self.positions[position_key].quantity = new_quantity
                logger.info(f"Updated {symbol} quantity: {old_quantity} -> {new_quantity}")
                return True
            else:
                logger.warning(f"Position {symbol} not found")
                return False

        except Exception as e:
            logger.error(f"Error updating position {symbol}: {e}")
            return False

    def _get_position_key(self, position: Position) -> str:
        """Generate unique key for position"""
        if position.position_type == "option":
            return f"{position.symbol}_{position.strike_price}_{position.expiry_date}_{position.option_type}"
        else:
            return position.symbol

    def _update_position_market_data(self, position: Position):
        """Update position with current market data"""
        try:
            if position.position_type == "crypto":
                market_data = self.data_connector.get_crypto_data(position.symbol.replace('-USD', ''))
            else:
                market_data = self.data_connector.get_real_time_quote(position.symbol)

            if market_data:
                position.current_price = market_data.price
                position.market_value = position.quantity * market_data.price
                position.unrealized_pnl = (market_data.price - position.entry_price) * position.quantity
            else:
                logger.warning(f"No market data available for {position.symbol}")

        except Exception as e:
            logger.error(f"Error updating market data for {position.symbol}: {e}")

    def refresh_portfolio(self):
        """Refresh all positions with current market data"""
        total_value = 0.0

        for position in self.positions.values():
            self._update_position_market_data(position)
            if position.market_value:
                total_value += position.market_value

        # Calculate weights
        for position in self.positions.values():
            if position.market_value and total_value > 0:
                position.weight = position.market_value / total_value

        logger.info(f"Portfolio refreshed. Total value: ${total_value:,.2f}")

    def get_portfolio_summary(self) -> PortfolioSummary:
        """Get comprehensive portfolio summary"""
        self.refresh_portfolio()

        total_value = sum(pos.market_value or 0 for pos in self.positions.values())
        total_pnl = sum(pos.unrealized_pnl or 0 for pos in self.positions.values())

        # Find largest position
        largest_pos = max(self.positions.values(),
                         key=lambda p: p.market_value or 0,
                         default=None)

        return PortfolioSummary(
            total_value=total_value,
            total_positions=len(self.positions),
            cash_balance=self.cash_balance,
            daily_pnl=total_pnl,  # This would be calculated differently in production
            total_return=(total_pnl / (total_value - total_pnl)) * 100 if total_value > total_pnl else 0,
            largest_position=largest_pos.symbol if largest_pos else "",
            largest_position_weight=largest_pos.weight or 0 if largest_pos else 0
        )

    def calculate_var(self, confidence_levels: List[float] = [0.95, 0.99],
                     time_horizons: List[int] = [1, 10],
                     method: str = "historical") -> VaRAnalysis:
        """Calculate comprehensive VaR analysis"""

        if not self.positions:
            raise ValueError("No positions in portfolio for VaR calculation")

        self.refresh_portfolio()

        # Prepare portfolio data for VaR calculation
        portfolio_data = []
        for position in self.positions.values():
            if position.market_value and position.market_value > 0:
                portfolio_data.append({
                    "symbol": position.symbol,
                    "quantity": position.quantity,
                    "weight": position.weight or 0,
                    "market_value": position.market_value,
                    "type": position.position_type
                })

        if method == "historical":
            return self._calculate_historical_var(portfolio_data, confidence_levels, time_horizons)
        elif method == "parametric":
            return self._calculate_parametric_var(portfolio_data, confidence_levels, time_horizons)
        elif method == "monte_carlo":
            return self._calculate_monte_carlo_var(portfolio_data, confidence_levels, time_horizons)
        else:
            raise ValueError(f"Unknown VaR method: {method}")

    def _calculate_historical_var(self, portfolio_data: List[Dict],
                                 confidence_levels: List[float],
                                 time_horizons: List[int]) -> VaRAnalysis:
        """Calculate VaR using historical simulation method"""

        # Get historical returns for all positions
        returns_data = {}
        portfolio_value = sum(pos["market_value"] for pos in portfolio_data)

        for position in portfolio_data:
            symbol = position["symbol"]
            try:
                hist_data = self.data_connector.get_stock_data(symbol, period="2y", interval="1d")
                if not hist_data.empty:
                    returns = hist_data['Close'].pct_change().dropna()
                    returns_data[symbol] = returns
            except Exception as e:
                logger.error(f"Error getting historical data for {symbol}: {e}")

        if not returns_data:
            raise ValueError("No historical data available for VaR calculation")

        # Calculate portfolio returns
        portfolio_returns = self._calculate_portfolio_returns(returns_data, portfolio_data)

        if len(portfolio_returns) < 100:
            raise ValueError("Insufficient historical data for reliable VaR calculation")

        # Calculate VaR metrics
        var_1d_95 = -np.percentile(portfolio_returns, (1 - 0.95) * 100) * portfolio_value
        var_1d_99 = -np.percentile(portfolio_returns, (1 - 0.99) * 100) * portfolio_value
        var_10d_95 = var_1d_95 * np.sqrt(10)

        # Expected Shortfall (Conditional VaR)
        var_95_threshold = -np.percentile(portfolio_returns, 5)
        tail_losses = portfolio_returns[portfolio_returns <= var_95_threshold]
        expected_shortfall = -np.mean(tail_losses) * portfolio_value if len(tail_losses) > 0 else var_1d_95

        # Portfolio volatility and Sharpe ratio
        portfolio_vol = np.std(portfolio_returns) * np.sqrt(252)
        portfolio_return = np.mean(portfolio_returns) * 252
        risk_free_rate = 0.02  # Assume 2% risk-free rate
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_vol if portfolio_vol > 0 else 0

        # Component VaR
        component_var = self._calculate_component_var(returns_data, portfolio_data, var_1d_95)

        # Maximum Drawdown
        cumulative_returns = (1 + pd.Series(portfolio_returns)).cumprod()
        rolling_max = cumulative_returns.expanding().max()
        drawdowns = (cumulative_returns - rolling_max) / rolling_max
        max_drawdown = abs(drawdowns.min()) * portfolio_value

        return VaRAnalysis(
            var_1d_95=var_1d_95,
            var_1d_99=var_1d_99,
            var_10d_95=var_10d_95,
            expected_shortfall_95=expected_shortfall,
            max_drawdown=max_drawdown,
            portfolio_volatility=portfolio_vol,
            sharpe_ratio=sharpe_ratio,
            component_var=component_var,
            methodology="Historical Simulation"
        )

    def _calculate_portfolio_returns(self, returns_data: Dict, portfolio_data: List[Dict]) -> np.array:
        """Calculate historical portfolio returns"""

        # Find common dates
        common_dates = None
        for symbol, returns in returns_data.items():
            if common_dates is None:
                common_dates = returns.index
            else:
                common_dates = common_dates.intersection(returns.index)

        portfolio_returns = []
        weights = {pos["symbol"]: pos["weight"] for pos in portfolio_data}

        for date in common_dates:
            portfolio_return = 0
            for symbol, weight in weights.items():
                if symbol in returns_data and date in returns_data[symbol].index:
                    portfolio_return += weight * returns_data[symbol][date]
            portfolio_returns.append(portfolio_return)

        return np.array(portfolio_returns)

    def _calculate_component_var(self, returns_data: Dict, portfolio_data: List[Dict],
                                total_var: float) -> Dict[str, float]:
        """Calculate component VaR for each position"""
        component_var = {}

        for position in portfolio_data:
            symbol = position["symbol"]
            weight = position["weight"]

            # Simplified component VaR calculation
            # In practice, this would use marginal VaR and correlation analysis
            component_var[symbol] = total_var * weight

        return component_var

    def _calculate_parametric_var(self, portfolio_data: List[Dict],
                                 confidence_levels: List[float],
                                 time_horizons: List[int]) -> VaRAnalysis:
        """Calculate VaR using parametric method (variance-covariance)"""
        # Simplified implementation - would need full covariance matrix in production
        portfolio_value = sum(pos["market_value"] for pos in portfolio_data)

        # Estimate portfolio volatility (simplified)
        avg_volatility = 0.20  # 20% annual volatility assumption
        daily_vol = avg_volatility / np.sqrt(252)

        # Calculate VaR using normal distribution assumption
        from scipy import stats
        var_1d_95 = stats.norm.ppf(0.05) * daily_vol * portfolio_value
        var_1d_99 = stats.norm.ppf(0.01) * daily_vol * portfolio_value

        return VaRAnalysis(
            var_1d_95=abs(var_1d_95),
            var_1d_99=abs(var_1d_99),
            var_10d_95=abs(var_1d_95) * np.sqrt(10),
            expected_shortfall_95=abs(var_1d_95) * 1.2,  # Approximation
            max_drawdown=abs(var_1d_95) * 3,  # Rough estimate
            portfolio_volatility=avg_volatility,
            sharpe_ratio=0.5,  # Default estimate
            component_var={pos["symbol"]: abs(var_1d_95) * pos["weight"] for pos in portfolio_data},
            methodology="Parametric (Variance-Covariance)"
        )

    def _calculate_monte_carlo_var(self, portfolio_data: List[Dict],
                                  confidence_levels: List[float],
                                  time_horizons: List[int]) -> VaRAnalysis:
        """Calculate VaR using Monte Carlo simulation"""
        # Simplified Monte Carlo implementation
        np.random.seed(42)
        n_simulations = 10000
        portfolio_value = sum(pos["market_value"] for pos in portfolio_data)

        # Generate random portfolio returns
        simulated_returns = np.random.normal(0.0005, 0.02, n_simulations)  # Daily returns
        simulated_values = simulated_returns * portfolio_value

        # Calculate VaR from simulations
        var_1d_95 = -np.percentile(simulated_values, 5)
        var_1d_99 = -np.percentile(simulated_values, 1)

        return VaRAnalysis(
            var_1d_95=var_1d_95,
            var_1d_99=var_1d_99,
            var_10d_95=var_1d_95 * np.sqrt(10),
            expected_shortfall_95=var_1d_95 * 1.3,  # Approximation
            max_drawdown=var_1d_95 * 4,  # Rough estimate
            portfolio_volatility=0.20,  # Assumed
            sharpe_ratio=0.6,  # Default estimate
            component_var={pos["symbol"]: var_1d_95 * pos["weight"] for pos in portfolio_data},
            methodology="Monte Carlo Simulation"
        )

    def save_portfolio(self, filename: Optional[str] = None):
        """Save portfolio to JSON file"""
        filename = filename or self.portfolio_file

        portfolio_data = {
            "positions": {key: asdict(pos) for key, pos in self.positions.items()},
            "cash_balance": self.cash_balance,
            "last_updated": datetime.now().isoformat()
        }

        try:
            with open(filename, 'w') as f:
                json.dump(portfolio_data, f, indent=2)
            logger.info(f"Portfolio saved to {filename}")
        except Exception as e:
            logger.error(f"Error saving portfolio: {e}")

    def load_portfolio(self, filename: Optional[str] = None):
        """Load portfolio from JSON file"""
        filename = filename or self.portfolio_file

        try:
            if Path(filename).exists():
                with open(filename, 'r') as f:
                    portfolio_data = json.load(f)

                # Load positions
                self.positions = {}
                for key, pos_data in portfolio_data.get("positions", {}).items():
                    position = Position(**pos_data)
                    self.positions[key] = position

                self.cash_balance = portfolio_data.get("cash_balance", 0.0)
                logger.info(f"Portfolio loaded from {filename}")
                return True
            else:
                logger.info(f"No existing portfolio file found at {filename}")
                return False

        except Exception as e:
            logger.error(f"Error loading portfolio: {e}")
            return False

    def import_from_csv(self, csv_file: str) -> bool:
        """Import portfolio from CSV file"""
        try:
            df = pd.read_csv(csv_file)
            required_columns = ['symbol', 'quantity', 'entry_price']

            if not all(col in df.columns for col in required_columns):
                logger.error(f"CSV must contain columns: {required_columns}")
                return False

            for _, row in df.iterrows():
                self.add_position(
                    symbol=row['symbol'],
                    quantity=float(row['quantity']),
                    entry_price=float(row['entry_price']),
                    position_type=row.get('position_type', 'stock')
                )

            logger.info(f"Imported {len(df)} positions from {csv_file}")
            return True

        except Exception as e:
            logger.error(f"Error importing from CSV: {e}")
            return False

    def export_to_csv(self, csv_file: str) -> bool:
        """Export portfolio to CSV file"""
        try:
            self.refresh_portfolio()

            data = []
            for position in self.positions.values():
                data.append({
                    'symbol': position.symbol,
                    'quantity': position.quantity,
                    'entry_price': position.entry_price,
                    'current_price': position.current_price,
                    'market_value': position.market_value,
                    'unrealized_pnl': position.unrealized_pnl,
                    'weight': position.weight,
                    'position_type': position.position_type,
                    'entry_date': position.entry_date
                })

            df = pd.DataFrame(data)
            df.to_csv(csv_file, index=False)
            logger.info(f"Portfolio exported to {csv_file}")
            return True

        except Exception as e:
            logger.error(f"Error exporting to CSV: {e}")
            return False

    def get_positions_summary(self) -> str:
        """Get formatted string summary of all positions"""
        if not self.positions:
            return "No positions in portfolio"

        self.refresh_portfolio()

        summary = "\n" + "="*80 + "\n"
        summary += "PORTFOLIO POSITIONS SUMMARY\n"
        summary += "="*80 + "\n"

        total_value = 0
        for position in self.positions.values():
            if position.market_value:
                total_value += position.market_value

        for position in sorted(self.positions.values(), key=lambda p: p.market_value or 0, reverse=True):
            pnl_color = "+" if (position.unrealized_pnl or 0) >= 0 else ""

            summary += f"\n{position.symbol:8} | "
            summary += f"Qty: {position.quantity:>8.2f} | "
            summary += f"Entry: ${position.entry_price:>7.2f} | "
            summary += f"Current: ${position.current_price or 0:>7.2f} | "
            summary += f"Value: ${position.market_value or 0:>10,.2f} | "
            summary += f"P&L: {pnl_color}${position.unrealized_pnl or 0:>8,.2f} | "
            summary += f"Weight: {(position.weight or 0)*100:>5.1f}%"

        summary += "\n" + "-"*80
        summary += f"\nTOTAL PORTFOLIO VALUE: ${total_value:>15,.2f}"
        summary += "\n" + "="*80

        return summary