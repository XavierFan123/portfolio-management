"""
Risk Dashboard
Comprehensive risk reporting and visualization for portfolio analysis
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import asdict

from .portfolio_manager import PortfolioManager, VaRAnalysis, PortfolioSummary

logger = logging.getLogger(__name__)

class RiskDashboard:
    """Comprehensive risk dashboard for portfolio analysis"""

    def __init__(self, portfolio_manager: PortfolioManager):
        self.portfolio_manager = portfolio_manager

    def generate_risk_report(self, include_scenarios: bool = True) -> str:
        """Generate comprehensive risk report"""
        try:
            # Get portfolio data
            self.portfolio_manager.refresh_portfolio()
            portfolio_summary = self.portfolio_manager.get_portfolio_summary()

            if not self.portfolio_manager.positions:
                return "No positions in portfolio for risk analysis."

            # Calculate VaR analysis
            var_analysis = self.portfolio_manager.calculate_var(method="historical")

            # Build report
            report = self._build_header(portfolio_summary)
            report += self._build_portfolio_overview(portfolio_summary)
            report += self._build_var_analysis_section(var_analysis, portfolio_summary)
            report += self._build_position_risk_breakdown()
            report += self._build_concentration_analysis()

            if include_scenarios:
                report += self._build_scenario_analysis()

            report += self._build_risk_recommendations(var_analysis, portfolio_summary)
            report += self._build_footer()

            return report

        except Exception as e:
            logger.error(f"Error generating risk report: {e}")
            return f"Error generating risk report: {e}"

    def _build_header(self, portfolio_summary: PortfolioSummary) -> str:
        """Build report header"""
        return f"""
╔═══════════════════════════════════════════════════════════════╗
║                   PROJECT CERBERUS                           ║
║              PORTFOLIO RISK ANALYSIS REPORT                  ║
║                                                              ║
║  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}                        ║
╚═══════════════════════════════════════════════════════════════╝

"""

    def _build_portfolio_overview(self, summary: PortfolioSummary) -> str:
        """Build portfolio overview section"""
        return f"""
{'='*80}
PORTFOLIO OVERVIEW
{'='*80}
Total Portfolio Value:    ${summary.total_value:>15,.2f}
Total Positions:          {summary.total_positions:>15}
Cash Balance:             ${summary.cash_balance:>15,.2f}
Unrealized P&L:           ${summary.daily_pnl:>15,.2f}
Total Return:             {summary.total_return:>14.2f}%
Largest Position:         {summary.largest_position:>15} ({summary.largest_position_weight*100:.1f}%)

"""

    def _build_var_analysis_section(self, var_analysis: VaRAnalysis,
                                  portfolio_summary: PortfolioSummary) -> str:
        """Build VaR analysis section"""
        var_1d_pct = (var_analysis.var_1d_95 / portfolio_summary.total_value * 100) if portfolio_summary.total_value > 0 else 0
        var_99_pct = (var_analysis.var_1d_99 / portfolio_summary.total_value * 100) if portfolio_summary.total_value > 0 else 0

        risk_level = self._assess_risk_level(var_1d_pct)

        return f"""
{'='*80}
VALUE AT RISK (VaR) ANALYSIS
{'='*80}
Methodology:              {var_analysis.methodology}

VaR METRICS:
1-Day VaR (95%):          ${var_analysis.var_1d_95:>12,.2f}  ({var_1d_pct:.2f}% of portfolio)
1-Day VaR (99%):          ${var_analysis.var_1d_99:>12,.2f}  ({var_99_pct:.2f}% of portfolio)
10-Day VaR (95%):         ${var_analysis.var_10d_95:>12,.2f}
Expected Shortfall:       ${var_analysis.expected_shortfall_95:>12,.2f}
Maximum Drawdown:         ${var_analysis.max_drawdown:>12,.2f}

PORTFOLIO RISK METRICS:
Portfolio Volatility:     {var_analysis.portfolio_volatility*100:>11.2f}%
Sharpe Ratio:             {var_analysis.sharpe_ratio:>15.2f}
Risk Level:               {risk_level:>15}

INTERPRETATION:
• There is a 5% probability of losing more than ${var_analysis.var_1d_95:,.2f} in one day
• There is a 1% probability of losing more than ${var_analysis.var_1d_99:,.2f} in one day
• Expected loss in worst 5% of outcomes: ${var_analysis.expected_shortfall_95:,.2f}

"""

    def _build_position_risk_breakdown(self) -> str:
        """Build position-by-position risk breakdown"""
        self.portfolio_manager.refresh_portfolio()

        report = f"""
{'='*80}
POSITION RISK BREAKDOWN
{'='*80}
{'Symbol':>8} | {'Quantity':>10} | {'Entry Price':>12} | {'Current Price':>14} | {'Market Value':>14} | {'Weight':>8} | {'Risk Score':>12}
{'-'*80}
"""

        total_value = sum(pos.market_value or 0 for pos in self.portfolio_manager.positions.values())

        for position in sorted(self.portfolio_manager.positions.values(),
                             key=lambda p: p.market_value or 0, reverse=True):

            weight = (position.weight or 0) * 100
            risk_score = self._calculate_position_risk_score(position, total_value)

            report += f"{position.symbol:>8} | "
            report += f"{position.quantity:>10.2f} | "
            report += f"${position.entry_price:>11.2f} | "
            report += f"${position.current_price or 0:>13.2f} | "
            report += f"${position.market_value or 0:>13,.2f} | "
            report += f"{weight:>7.1f}% | "
            report += f"{risk_score:>11}\n"

        return report + "\n"

    def _build_concentration_analysis(self) -> str:
        """Build concentration risk analysis"""
        if not self.portfolio_manager.positions:
            return ""

        self.portfolio_manager.refresh_portfolio()
        total_value = sum(pos.market_value or 0 for pos in self.portfolio_manager.positions.values())

        # Calculate concentration metrics
        weights = [(pos.symbol, (pos.market_value or 0) / total_value)
                  for pos in self.portfolio_manager.positions.values() if total_value > 0]
        weights.sort(key=lambda x: x[1], reverse=True)

        # Herfindahl-Hirschman Index (concentration measure)
        hhi = sum(weight[1]**2 for weight in weights) * 10000 if weights else 0

        # Top positions
        top_3_weight = sum(weight[1] for weight in weights[:3]) * 100
        top_5_weight = sum(weight[1] for weight in weights[:5]) * 100

        concentration_level = self._assess_concentration_risk(hhi)

        report = f"""
{'='*80}
CONCENTRATION RISK ANALYSIS
{'='*80}
Herfindahl-Hirschman Index:    {hhi:>8.0f}
Concentration Level:           {concentration_level:>15}
Top 3 Positions Weight:        {top_3_weight:>14.1f}%
Top 5 Positions Weight:        {top_5_weight:>14.1f}%

TOP POSITION CONCENTRATIONS:
"""

        for i, (symbol, weight) in enumerate(weights[:5], 1):
            stars = "★" * min(int(weight * 20), 5)  # Visual representation
            report += f"{i:>2}. {symbol:>8}: {weight*100:>6.1f}% {stars}\n"

        return report + "\n"

    def _build_scenario_analysis(self) -> str:
        """Build scenario analysis section"""
        scenarios = self._run_scenario_analysis()

        report = f"""
{'='*80}
SCENARIO ANALYSIS
{'='*80}
Portfolio impact under different market conditions:

{'Scenario':>20} | {'Portfolio Impact':>20} | {'P&L Change':>15}
{'-'*80}
"""

        for scenario_name, impact in scenarios.items():
            impact_pct = impact["percentage"]
            impact_dollar = impact["dollar_amount"]
            color_indicator = "📈" if impact_pct > 0 else "📉" if impact_pct < 0 else "➡️"

            report += f"{scenario_name:>20} | {impact_pct:>17.1f}% | ${impact_dollar:>13,.0f} {color_indicator}\n"

        return report + "\n"

    def _build_risk_recommendations(self, var_analysis: VaRAnalysis,
                                  portfolio_summary: PortfolioSummary) -> str:
        """Build risk management recommendations"""
        recommendations = []

        # VaR-based recommendations
        var_percentage = (var_analysis.var_1d_95 / portfolio_summary.total_value * 100) if portfolio_summary.total_value > 0 else 0

        if var_percentage > 5:
            recommendations.append("HIGH RISK: Consider reducing position sizes or adding hedging instruments")
            recommendations.append("Diversify across more assets to reduce concentration risk")

        if var_percentage > 3:
            recommendations.append("Monitor portfolio daily and consider stop-loss levels")

        # Volatility-based recommendations
        if var_analysis.portfolio_volatility > 0.25:
            recommendations.append("High volatility detected - consider volatility-reducing strategies")

        # Sharpe ratio recommendations
        if var_analysis.sharpe_ratio < 0.5:
            recommendations.append("Low risk-adjusted returns - review asset allocation")

        # Concentration recommendations
        if portfolio_summary.largest_position_weight > 0.2:
            recommendations.append(f"Large concentration in {portfolio_summary.largest_position} - consider rebalancing")

        # General recommendations
        recommendations.append("Regularly review and rebalance portfolio based on risk tolerance")
        recommendations.append("Consider implementing systematic hedging strategies for large positions")

        report = f"""
{'='*80}
RISK MANAGEMENT RECOMMENDATIONS
{'='*80}
"""

        for i, rec in enumerate(recommendations, 1):
            report += f"{i:>2}. {rec}\n"

        return report

    def _build_footer(self) -> str:
        """Build report footer"""
        return f"""
{'='*80}
DISCLAIMER
{'='*80}
This risk analysis is based on historical data and statistical models.
Past performance does not guarantee future results.
Market conditions can change rapidly, affecting risk profiles.
Please consult with financial professionals for investment decisions.

Report generated by Project Cerberus Portfolio Management System
{'='*80}
"""

    def _assess_risk_level(self, var_percentage: float) -> str:
        """Assess overall risk level based on VaR percentage"""
        if var_percentage > 5:
            return "HIGH RISK"
        elif var_percentage > 2:
            return "MODERATE RISK"
        else:
            return "LOW RISK"

    def _assess_concentration_risk(self, hhi: float) -> str:
        """Assess concentration risk based on HHI"""
        if hhi > 2500:
            return "HIGH"
        elif hhi > 1500:
            return "MODERATE"
        else:
            return "LOW"

    def _calculate_position_risk_score(self, position, total_portfolio_value: float) -> str:
        """Calculate risk score for individual position"""
        if not position.market_value or total_portfolio_value == 0:
            return "N/A"

        weight = position.market_value / total_portfolio_value

        # Simple risk scoring based on weight and volatility proxy
        if weight > 0.2:  # More than 20% of portfolio
            return "HIGH"
        elif weight > 0.1:  # More than 10% of portfolio
            return "MODERATE"
        else:
            return "LOW"

    def _run_scenario_analysis(self) -> Dict[str, Dict[str, float]]:
        """Run scenario analysis on portfolio"""
        if not self.portfolio_manager.positions:
            return {}

        portfolio_summary = self.portfolio_manager.get_portfolio_summary()
        base_value = portfolio_summary.total_value

        scenarios = {
            "Market Crash (-20%)": -0.20,
            "Market Correction (-10%)": -0.10,
            "Market Decline (-5%)": -0.05,
            "Normal Market (0%)": 0.00,
            "Market Rally (+5%)": 0.05,
            "Bull Market (+10%)": 0.10,
            "Strong Bull (+20%)": 0.20
        }

        results = {}
        for scenario_name, market_move in scenarios.items():
            # Simplified scenario - assume all positions move with market
            # In practice, this would use individual asset correlations
            portfolio_change = base_value * market_move

            results[scenario_name] = {
                "percentage": market_move * 100,
                "dollar_amount": portfolio_change
            }

        return results

    def get_daily_risk_summary(self) -> str:
        """Get quick daily risk summary"""
        try:
            if not self.portfolio_manager.positions:
                return "No positions for risk analysis"

            self.portfolio_manager.refresh_portfolio()
            portfolio_summary = self.portfolio_manager.get_portfolio_summary()
            var_analysis = self.portfolio_manager.calculate_var(method="historical")

            var_pct = (var_analysis.var_1d_95 / portfolio_summary.total_value * 100) if portfolio_summary.total_value > 0 else 0
            risk_level = self._assess_risk_level(var_pct)

            return f"""
📊 DAILY RISK SUMMARY - {datetime.now().strftime('%Y-%m-%d')}
{'='*60}
Portfolio Value:     ${portfolio_summary.total_value:>12,.2f}
1-Day VaR (95%):     ${var_analysis.var_1d_95:>12,.2f} ({var_pct:.1f}%)
Risk Level:          {risk_level:>15}
Positions:           {portfolio_summary.total_positions:>15}
Largest Position:    {portfolio_summary.largest_position} ({portfolio_summary.largest_position_weight*100:.1f}%)
{'='*60}
"""

        except Exception as e:
            return f"Error generating daily summary: {e}"

    def export_risk_report(self, filename: Optional[str] = None) -> bool:
        """Export risk report to file"""
        filename = filename or f"risk_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

        try:
            report = self.generate_risk_report()
            with open(filename, 'w') as f:
                f.write(report)
            logger.info(f"Risk report exported to {filename}")
            return True

        except Exception as e:
            logger.error(f"Error exporting risk report: {e}")
            return False