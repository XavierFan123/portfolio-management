#!/usr/bin/env python3
"""
Portfolio Management CLI
Interactive command-line interface for managing your portfolio and calculating VaR
"""

import sys
import asyncio
import cmd
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Optional

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from portfolio.portfolio_manager import PortfolioManager, VaRAnalysis
from portfolio.risk_dashboard import RiskDashboard
from data.yahoo_connector import YahooFinanceConnector

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class PortfolioCLI(cmd.Cmd):
    """Interactive Portfolio Management CLI"""

    intro = """
╔═══════════════════════════════════════════════════════════════╗
║                 PROJECT CERBERUS                              ║
║               Portfolio Management CLI                        ║
║                                                              ║
║  Interactive Portfolio & VaR Analysis System                 ║
╚═══════════════════════════════════════════════════════════════╝

Welcome to the Portfolio Management CLI!
Type 'help' for available commands or 'help <command>' for specific help.
Type 'quit' to exit.
    """

    prompt = "Cerberus> "

    def __init__(self):
        super().__init__()
        self.portfolio_manager = PortfolioManager()
        self.portfolio_manager.load_portfolio()
        self.risk_dashboard = RiskDashboard(self.portfolio_manager)

    def do_add_position(self, line):
        """
        Add a new position to your portfolio
        Usage: add_position SYMBOL QUANTITY ENTRY_PRICE [POSITION_TYPE]

        Examples:
        add_position AAPL 100 175.50
        add_position TSLA 50 245.00 stock
        add_position BTC-USD 2.5 45000 crypto
        """
        args = line.split()
        if len(args) < 3:
            print("Error: Required arguments: SYMBOL QUANTITY ENTRY_PRICE")
            print("Usage: add_position AAPL 100 175.50")
            return

        try:
            symbol = args[0].upper()
            quantity = float(args[1])
            entry_price = float(args[2])
            position_type = args[3] if len(args) > 3 else "stock"

            success = self.portfolio_manager.add_position(symbol, quantity, entry_price, position_type)
            if success:
                print(f"✅ Added {quantity} shares of {symbol} at ${entry_price:.2f}")
                self.portfolio_manager.save_portfolio()
            else:
                print(f"❌ Failed to add position {symbol}")

        except ValueError as e:
            print(f"Error: Invalid number format - {e}")
        except Exception as e:
            print(f"Error: {e}")

    def do_add_option(self, line):
        """
        Add an option position to your portfolio
        Usage: add_option SYMBOL QUANTITY ENTRY_PRICE STRIKE_PRICE EXPIRY_DATE OPTION_TYPE

        Example:
        add_option SPY 10 5.50 435 2024-01-19 call
        """
        args = line.split()
        if len(args) < 6:
            print("Error: Required arguments: SYMBOL QUANTITY ENTRY_PRICE STRIKE_PRICE EXPIRY_DATE OPTION_TYPE")
            print("Usage: add_option SPY 10 5.50 435 2024-01-19 call")
            return

        try:
            symbol = args[0].upper()
            quantity = float(args[1])
            entry_price = float(args[2])
            strike_price = float(args[3])
            expiry_date = args[4]
            option_type = args[5].lower()

            success = self.portfolio_manager.add_position(
                symbol=symbol,
                quantity=quantity,
                entry_price=entry_price,
                position_type="option",
                strike_price=strike_price,
                expiry_date=expiry_date,
                option_type=option_type
            )

            if success:
                print(f"✅ Added {quantity} {symbol} {option_type} options (${strike_price} strike, {expiry_date} expiry)")
                self.portfolio_manager.save_portfolio()
            else:
                print(f"❌ Failed to add option position")

        except ValueError as e:
            print(f"Error: Invalid number format - {e}")
        except Exception as e:
            print(f"Error: {e}")

    def do_remove_position(self, line):
        """
        Remove a position from your portfolio
        Usage: remove_position SYMBOL

        Example:
        remove_position AAPL
        """
        if not line.strip():
            print("Error: Please specify a symbol")
            print("Usage: remove_position AAPL")
            return

        symbol = line.strip().upper()
        success = self.portfolio_manager.remove_position(symbol)

        if success:
            print(f"✅ Removed position {symbol}")
            self.portfolio_manager.save_portfolio()
        else:
            print(f"❌ Position {symbol} not found")

    def do_update_quantity(self, line):
        """
        Update the quantity of an existing position
        Usage: update_quantity SYMBOL NEW_QUANTITY

        Example:
        update_quantity AAPL 150
        """
        args = line.split()
        if len(args) != 2:
            print("Error: Required arguments: SYMBOL NEW_QUANTITY")
            print("Usage: update_quantity AAPL 150")
            return

        try:
            symbol = args[0].upper()
            new_quantity = float(args[1])

            success = self.portfolio_manager.update_position_quantity(symbol, new_quantity)
            if success:
                print(f"✅ Updated {symbol} quantity to {new_quantity}")
                self.portfolio_manager.save_portfolio()
            else:
                print(f"❌ Position {symbol} not found")

        except ValueError as e:
            print(f"Error: Invalid number format - {e}")

    def do_show_portfolio(self, line):
        """
        Display your complete portfolio with current market values
        Usage: show_portfolio
        """
        try:
            summary = self.portfolio_manager.get_positions_summary()
            print(summary)

            # Show portfolio summary
            portfolio_summary = self.portfolio_manager.get_portfolio_summary()
            print(f"\n📊 PORTFOLIO SUMMARY:")
            print(f"   Total Positions: {portfolio_summary.total_positions}")
            print(f"   Total Value: ${portfolio_summary.total_value:,.2f}")
            print(f"   Unrealized P&L: ${portfolio_summary.daily_pnl:,.2f}")
            print(f"   Total Return: {portfolio_summary.total_return:.2f}%")
            print(f"   Largest Position: {portfolio_summary.largest_position} ({portfolio_summary.largest_position_weight*100:.1f}%)")

        except Exception as e:
            print(f"Error displaying portfolio: {e}")

    def do_calculate_var(self, line):
        """
        Calculate Value at Risk (VaR) for your portfolio
        Usage: calculate_var [METHOD]

        Methods: historical, parametric, monte_carlo
        Default: historical

        Examples:
        calculate_var
        calculate_var historical
        calculate_var monte_carlo
        """
        if not self.portfolio_manager.positions:
            print("❌ No positions in portfolio. Add some positions first.")
            return

        method = line.strip().lower() if line.strip() else "historical"
        valid_methods = ["historical", "parametric", "monte_carlo"]

        if method not in valid_methods:
            print(f"Error: Invalid method '{method}'. Valid methods: {', '.join(valid_methods)}")
            return

        try:
            print(f"🔄 Calculating VaR using {method} method...")
            print("   (This may take a moment to fetch historical data)")

            var_analysis = self.portfolio_manager.calculate_var(method=method)
            self._display_var_analysis(var_analysis)

        except Exception as e:
            print(f"❌ Error calculating VaR: {e}")

    def _display_var_analysis(self, var_analysis: VaRAnalysis):
        """Display comprehensive VaR analysis results"""
        print("\n" + "="*80)
        print("VALUE AT RISK (VaR) ANALYSIS")
        print("="*80)
        print(f"Methodology: {var_analysis.methodology}")
        print(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("-"*80)

        # VaR metrics
        print("📉 VaR METRICS:")
        print(f"   1-Day VaR (95%):     ${var_analysis.var_1d_95:>12,.2f}")
        print(f"   1-Day VaR (99%):     ${var_analysis.var_1d_99:>12,.2f}")
        print(f"   10-Day VaR (95%):    ${var_analysis.var_10d_95:>12,.2f}")
        print(f"   Expected Shortfall:  ${var_analysis.expected_shortfall_95:>12,.2f}")
        print(f"   Maximum Drawdown:    ${var_analysis.max_drawdown:>12,.2f}")

        print("\n📊 PORTFOLIO METRICS:")
        print(f"   Portfolio Volatility: {var_analysis.portfolio_volatility*100:>8.2f}%")
        print(f"   Sharpe Ratio:         {var_analysis.sharpe_ratio:>12.2f}")

        # Component VaR
        if var_analysis.component_var:
            print("\n💼 COMPONENT VaR (Risk Contribution by Position):")
            total_component_var = sum(var_analysis.component_var.values())

            for symbol, component_var in sorted(var_analysis.component_var.items(),
                                              key=lambda x: x[1], reverse=True):
                percentage = (component_var / total_component_var * 100) if total_component_var > 0 else 0
                print(f"   {symbol:8}: ${component_var:>10,.2f} ({percentage:>5.1f}%)")

        print("\n🎯 RISK INTERPRETATION:")
        portfolio_summary = self.portfolio_manager.get_portfolio_summary()
        if portfolio_summary.total_value > 0:
            var_percentage = (var_analysis.var_1d_95 / portfolio_summary.total_value) * 100
            print(f"   • There is a 5% chance of losing more than ${var_analysis.var_1d_95:,.2f}")
            print(f"     in a single day ({var_percentage:.1f}% of portfolio value)")
            print(f"   • There is a 1% chance of losing more than ${var_analysis.var_1d_99:,.2f}")
            print(f"     in a single day")

            if var_percentage > 5:
                print("   ⚠️  HIGH RISK: Consider reducing position sizes or diversifying")
            elif var_percentage > 2:
                print("   ⚡ MODERATE RISK: Monitor positions closely")
            else:
                print("   ✅ LOW RISK: Portfolio appears well-diversified")

        print("="*80)

    def do_import_csv(self, line):
        """
        Import portfolio from CSV file
        Usage: import_csv FILENAME

        CSV format: symbol,quantity,entry_price,position_type
        Example: import_csv my_portfolio.csv
        """
        if not line.strip():
            print("Error: Please specify a CSV filename")
            print("Usage: import_csv my_portfolio.csv")
            return

        filename = line.strip()
        try:
            success = self.portfolio_manager.import_from_csv(filename)
            if success:
                print(f"✅ Portfolio imported from {filename}")
                self.portfolio_manager.save_portfolio()
            else:
                print(f"❌ Failed to import from {filename}")
        except Exception as e:
            print(f"Error: {e}")

    def do_export_csv(self, line):
        """
        Export portfolio to CSV file
        Usage: export_csv FILENAME

        Example: export_csv my_portfolio_export.csv
        """
        filename = line.strip() if line.strip() else "portfolio_export.csv"

        try:
            success = self.portfolio_manager.export_to_csv(filename)
            if success:
                print(f"✅ Portfolio exported to {filename}")
            else:
                print(f"❌ Failed to export to {filename}")
        except Exception as e:
            print(f"Error: {e}")

    def do_refresh(self, line):
        """
        Refresh portfolio with current market prices
        Usage: refresh
        """
        try:
            print("🔄 Refreshing portfolio with current market data...")
            self.portfolio_manager.refresh_portfolio()
            print("✅ Portfolio refreshed successfully")
        except Exception as e:
            print(f"Error refreshing portfolio: {e}")

    def do_cash(self, line):
        """
        Set or view cash balance
        Usage: cash [AMOUNT]

        Examples:
        cash            # View current cash balance
        cash 10000      # Set cash balance to $10,000
        """
        if not line.strip():
            print(f"Current cash balance: ${self.portfolio_manager.cash_balance:,.2f}")
        else:
            try:
                amount = float(line.strip())
                self.portfolio_manager.cash_balance = amount
                self.portfolio_manager.save_portfolio()
                print(f"✅ Cash balance set to ${amount:,.2f}")
            except ValueError:
                print("Error: Please enter a valid number")

    def do_clear(self, line):
        """
        Clear all positions from portfolio
        Usage: clear
        """
        confirm = input("⚠️  Are you sure you want to clear all positions? (yes/no): ")
        if confirm.lower() in ['yes', 'y']:
            self.portfolio_manager.positions.clear()
            self.portfolio_manager.save_portfolio()
            print("✅ All positions cleared")
        else:
            print("❌ Clear operation cancelled")

    def do_status(self, line):
        """
        Show portfolio status and quick summary
        Usage: status
        """
        try:
            portfolio_summary = self.portfolio_manager.get_portfolio_summary()

            print("\n📊 PORTFOLIO STATUS")
            print("="*50)
            print(f"Total Positions:    {portfolio_summary.total_positions}")
            print(f"Total Value:        ${portfolio_summary.total_value:,.2f}")
            print(f"Cash Balance:       ${portfolio_summary.cash_balance:,.2f}")
            print(f"Unrealized P&L:     ${portfolio_summary.daily_pnl:,.2f}")
            print(f"Total Return:       {portfolio_summary.total_return:.2f}%")

            if portfolio_summary.largest_position:
                print(f"Largest Position:   {portfolio_summary.largest_position} ({portfolio_summary.largest_position_weight*100:.1f}%)")

            print("="*50)

        except Exception as e:
            print(f"Error getting status: {e}")

    def do_risk_report(self, line):
        """
        Generate comprehensive risk analysis report
        Usage: risk_report
        """
        try:
            print("🔄 Generating comprehensive risk report...")
            print("   (This may take a moment to analyze portfolio data)")

            report = self.risk_dashboard.generate_risk_report()
            print(report)

        except Exception as e:
            print(f"❌ Error generating risk report: {e}")

    def do_daily_risk(self, line):
        """
        Show quick daily risk summary
        Usage: daily_risk
        """
        try:
            summary = self.risk_dashboard.get_daily_risk_summary()
            print(summary)

        except Exception as e:
            print(f"❌ Error getting daily risk summary: {e}")

    def do_export_risk_report(self, line):
        """
        Export comprehensive risk report to file
        Usage: export_risk_report [FILENAME]

        Example: export_risk_report my_risk_analysis.txt
        """
        filename = line.strip() if line.strip() else None

        try:
            print("🔄 Exporting risk report...")
            success = self.risk_dashboard.export_risk_report(filename)

            if success:
                report_filename = filename or f"risk_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                print(f"✅ Risk report exported to {report_filename}")
            else:
                print("❌ Failed to export risk report")

        except Exception as e:
            print(f"Error exporting risk report: {e}")

    def do_help_getting_started(self, line):
        """Show getting started guide"""
        print("""
🚀 GETTING STARTED WITH PORTFOLIO MANAGEMENT

1. ADD POSITIONS:
   add_position AAPL 100 175.50        # Add 100 shares of Apple
   add_position TSLA 50 245.00         # Add 50 shares of Tesla
   add_position BTC-USD 2.5 45000      # Add Bitcoin

2. ADD OPTIONS:
   add_option SPY 10 5.50 435 2024-01-19 call  # Add SPY call options

3. VIEW PORTFOLIO:
   show_portfolio                       # See all positions
   status                              # Quick summary

4. CALCULATE RISK:
   calculate_var                       # Calculate Value at Risk
   calculate_var monte_carlo           # Use different method
   risk_report                         # Comprehensive risk analysis
   daily_risk                          # Quick daily risk summary

5. MANAGE PORTFOLIO:
   update_quantity AAPL 150            # Change position size
   remove_position TSLA                # Remove position
   cash 10000                          # Set cash balance

6. IMPORT/EXPORT:
   import_csv my_positions.csv         # Import from CSV
   export_csv my_export.csv            # Export to CSV
   export_risk_report risk_analysis.txt # Export risk report

Type 'help <command>' for detailed help on any command.
        """)

    def do_quit(self, line):
        """Exit the portfolio management CLI"""
        print("💾 Saving portfolio...")
        self.portfolio_manager.save_portfolio()
        print("👋 Goodbye! Your portfolio has been saved.")
        return True

    def do_exit(self, line):
        """Exit the portfolio management CLI"""
        return self.do_quit(line)

    def emptyline(self):
        """Do nothing on empty line"""
        pass

    def default(self, line):
        """Handle unknown commands"""
        print(f"Unknown command: {line}")
        print("Type 'help' for available commands or 'help_getting_started' for a guide.")

def main():
    """Main function to start the CLI"""
    try:
        cli = PortfolioCLI()
        cli.cmdloop()
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()