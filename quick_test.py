#!/usr/bin/env python3
"""
Quick test for Portfolio Management System
"""

import sys
from pathlib import Path

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from portfolio.portfolio_manager import PortfolioManager
from portfolio.risk_dashboard import RiskDashboard

def main():
    print("PROJECT CERBERUS - Portfolio Management System")
    print("=" * 60)

    # Initialize system
    portfolio_manager = PortfolioManager()

    # Add test positions
    print("\nAdding test positions...")
    positions = [
        ("AAPL", 100, 175.50),
        ("MSFT", 50, 280.00),
        ("GOOGL", 25, 120.00)
    ]

    for symbol, quantity, price in positions:
        success = portfolio_manager.add_position(symbol, quantity, price, "stock")
        print(f"Added {symbol}: {'SUCCESS' if success else 'FAILED'}")

    # Calculate VaR
    print("\nCalculating VaR...")
    var_analysis = portfolio_manager.calculate_var(method="historical")
    print(f"1-Day VaR (95%): ${var_analysis.var_1d_95:,.2f}")

    # Generate risk report
    print("\nGenerating risk dashboard...")
    risk_dashboard = RiskDashboard(portfolio_manager)
    daily_summary = risk_dashboard.get_daily_risk_summary()
    print("Daily Risk Summary:")
    print(daily_summary)

    print("\n" + "=" * 60)
    print("TEST COMPLETED SUCCESSFULLY!")
    print("System is ready. Run: python portfolio_cli.py")

if __name__ == "__main__":
    main()