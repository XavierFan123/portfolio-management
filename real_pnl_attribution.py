"""
Real P&L Attribution System
Tracks actual portfolio P&L changes and attributes them to specific risk factors and Greeks
"""
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import sqlite3
from loguru import logger
import warnings
warnings.filterwarnings('ignore')

# Import our real data modules
from real_volatility import real_volatility_calculator
from factor_regression import factor_analyzer

class RealPnLAttributor:
    """Professional P&L attribution using real market data and factor exposures"""

    def __init__(self, db_path: str = 'pnl_attribution.db'):
        self.db_path = db_path
        self.init_database()

    def init_database(self):
        """Initialize P&L attribution database tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Portfolio snapshots table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS portfolio_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date DATE NOT NULL,
                symbol VARCHAR(20) NOT NULL,
                position_size REAL NOT NULL,
                price REAL NOT NULL,
                market_value REAL NOT NULL,
                delta REAL,
                gamma REAL,
                theta REAL,
                vega REAL,
                rho REAL,
                implied_vol REAL,
                time_to_expiry REAL,
                strike REAL,
                option_type VARCHAR(10),
                UNIQUE(date, symbol)
            )
        ''')

        # Daily P&L attribution table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS daily_pnl_attribution (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date DATE NOT NULL,
                symbol VARCHAR(20) NOT NULL,
                total_pnl REAL NOT NULL,
                delta_pnl REAL,
                gamma_pnl REAL,
                theta_pnl REAL,
                vega_pnl REAL,
                rho_pnl REAL,
                market_factor_pnl REAL,
                tech_factor_pnl REAL,
                size_factor_pnl REAL,
                vol_factor_pnl REAL,
                crypto_factor_pnl REAL,
                gold_factor_pnl REAL,
                bond_factor_pnl REAL,
                dollar_factor_pnl REAL,
                unexplained_pnl REAL,
                UNIQUE(date, symbol)
            )
        ''')

        # Factor returns daily table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS daily_factor_returns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date DATE NOT NULL,
                spy_return REAL,
                qqq_return REAL,
                iwm_return REAL,
                vix_return REAL,
                btc_return REAL,
                gld_return REAL,
                tlt_return REAL,
                dxy_return REAL,
                UNIQUE(date)
            )
        ''')

        # Greeks sensitivity analysis table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS greeks_sensitivity (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date DATE NOT NULL,
                symbol VARCHAR(20) NOT NULL,
                underlying_price_change REAL NOT NULL,
                vol_change REAL,
                time_decay REAL,
                rate_change REAL,
                predicted_delta_pnl REAL,
                predicted_gamma_pnl REAL,
                predicted_theta_pnl REAL,
                predicted_vega_pnl REAL,
                predicted_rho_pnl REAL,
                actual_pnl REAL,
                greeks_accuracy REAL,
                UNIQUE(date, symbol)
            )
        ''')

        conn.commit()
        conn.close()
        logger.info("Real P&L attribution database initialized")

    def capture_portfolio_snapshot(self, positions: List[Dict]) -> bool:
        """Capture current portfolio state for P&L tracking"""
        try:
            today = datetime.now().date()
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            for position in positions:
                symbol = position.get('symbol', '')
                position_size = float(position.get('quantity', 0))
                current_price = float(position.get('current_price', 0))
                market_value = float(position.get('market_value', 0))

                # Get Greeks from position if available
                greeks = position.get('greeks', {})
                delta = greeks.get('delta', 0)
                gamma = greeks.get('gamma', 0)
                theta = greeks.get('theta', 0)
                vega = greeks.get('vega', 0)
                rho = greeks.get('rho', 0)

                # Get option details if it's an option
                strike = position.get('strike')
                option_type = position.get('option_type')
                time_to_expiry = position.get('time_to_expiry')

                # Get current implied volatility
                implied_vol = None
                if strike and option_type:
                    implied_vol = real_volatility_calculator.get_current_volatility(
                        symbol.split('_')[0] if '_' in symbol else symbol, 30, 'implied'
                    )

                cursor.execute('''
                    INSERT OR REPLACE INTO portfolio_snapshots
                    (date, symbol, position_size, price, market_value, delta, gamma,
                     theta, vega, rho, implied_vol, time_to_expiry, strike, option_type)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    today, symbol, position_size, current_price, market_value,
                    delta, gamma, theta, vega, rho, implied_vol, time_to_expiry,
                    strike, option_type
                ))

            conn.commit()
            conn.close()
            logger.info(f"Captured portfolio snapshot for {len(positions)} positions")
            return True

        except Exception as e:
            logger.error(f"Error capturing portfolio snapshot: {e}")
            return False

    def fetch_daily_factor_returns(self, date: datetime.date) -> Dict[str, float]:
        """Fetch daily returns for all risk factors"""
        try:
            factor_symbols = {
                'SPY': 'spy_return',
                'QQQ': 'qqq_return',
                'IWM': 'iwm_return',
                'VIX': 'vix_return',
                'BTC-USD': 'btc_return',
                'GLD': 'gld_return',
                'TLT': 'tlt_return',
                'DXY': 'dxy_return'
            }

            returns = {}
            end_date = date + timedelta(days=1)
            start_date = date - timedelta(days=2)

            for symbol, col_name in factor_symbols.items():
                try:
                    # Handle special ticker symbols
                    if symbol == 'VIX':
                        ticker_symbol = '^VIX'
                    elif symbol == 'DXY':
                        ticker_symbol = 'DX-Y.NYB'
                    else:
                        ticker_symbol = symbol

                    ticker = yf.Ticker(ticker_symbol)
                    hist_data = ticker.history(start=start_date, end=end_date)

                    if len(hist_data) >= 2:
                        daily_return = hist_data['Close'].pct_change().iloc[-1]
                        if not pd.isna(daily_return):
                            returns[col_name] = float(daily_return)
                        else:
                            returns[col_name] = 0.0
                    else:
                        returns[col_name] = 0.0

                except Exception as e:
                    logger.warning(f"Error fetching return for {symbol}: {e}")
                    returns[col_name] = 0.0

            # Store in database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute('''
                INSERT OR REPLACE INTO daily_factor_returns
                (date, spy_return, qqq_return, iwm_return, vix_return,
                 btc_return, gld_return, tlt_return, dxy_return)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                date, returns.get('spy_return', 0), returns.get('qqq_return', 0),
                returns.get('iwm_return', 0), returns.get('vix_return', 0),
                returns.get('btc_return', 0), returns.get('gld_return', 0),
                returns.get('tlt_return', 0), returns.get('dxy_return', 0)
            ))

            conn.commit()
            conn.close()

            logger.info(f"Fetched factor returns for {date}: {len(returns)} factors")
            return returns

        except Exception as e:
            logger.error(f"Error fetching factor returns for {date}: {e}")
            return {}

    def calculate_daily_pnl_attribution(self, date: datetime.date) -> Dict:
        """Calculate P&L attribution for a specific date"""
        try:
            conn = sqlite3.connect(self.db_path)

            # Get portfolio snapshots for current and previous day
            prev_date = date - timedelta(days=1)

            current_query = '''
                SELECT * FROM portfolio_snapshots WHERE date = ?
            '''
            current_positions = pd.read_sql_query(current_query, conn, params=(date,))

            prev_query = '''
                SELECT * FROM portfolio_snapshots WHERE date = ?
            '''
            prev_positions = pd.read_sql_query(prev_query, conn, params=(prev_date,))

            if current_positions.empty or prev_positions.empty:
                logger.warning(f"Insufficient snapshot data for P&L attribution on {date}")
                conn.close()
                return {}

            # Get factor returns for the day
            factor_returns = self.fetch_daily_factor_returns(date)

            attribution_results = {}

            for _, current_pos in current_positions.iterrows():
                symbol = current_pos['symbol']

                # Find matching previous position
                prev_pos = prev_positions[prev_positions['symbol'] == symbol]
                if prev_pos.empty:
                    continue

                prev_pos = prev_pos.iloc[0]

                # Calculate actual P&L
                actual_pnl = current_pos['market_value'] - prev_pos['market_value']

                # Calculate Greeks-based P&L attribution
                price_change = current_pos['price'] - prev_pos['price']
                price_change_pct = price_change / prev_pos['price'] if prev_pos['price'] != 0 else 0

                # Delta P&L
                delta_pnl = prev_pos['delta'] * price_change * prev_pos['position_size'] if prev_pos['delta'] else 0

                # Gamma P&L (1/2 * Gamma * (Price Change)^2)
                gamma_pnl = 0.5 * prev_pos['gamma'] * (price_change ** 2) * prev_pos['position_size'] if prev_pos['gamma'] else 0

                # Theta P&L (time decay for 1 day)
                theta_pnl = prev_pos['theta'] * prev_pos['position_size'] if prev_pos['theta'] else 0

                # Vega P&L (volatility change impact)
                vega_pnl = 0
                if prev_pos['vega'] and current_pos['implied_vol'] and prev_pos['implied_vol']:
                    vol_change = current_pos['implied_vol'] - prev_pos['implied_vol']
                    vega_pnl = prev_pos['vega'] * vol_change * prev_pos['position_size']

                # Rho P&L (assume small rate change)
                rho_pnl = 0  # Would need interest rate data

                # Factor-based P&L attribution
                underlying_symbol = symbol.split('_')[0] if '_' in symbol else symbol
                factor_exposures = factor_analyzer.calculate_factor_exposures(underlying_symbol)

                factor_pnl = {}
                if factor_exposures and factor_returns:
                    # Map factor returns to P&L impact
                    factor_mapping = {
                        'market_beta': 'spy_return',
                        'tech_beta': 'qqq_return',
                        'size_beta': 'iwm_return',
                        'vol_beta': 'vix_return',
                        'crypto_beta': 'btc_return',
                        'gold_beta': 'gld_return',
                        'bond_beta': 'tlt_return',
                        'dollar_beta': 'dxy_return'
                    }

                    for beta_key, return_key in factor_mapping.items():
                        beta = factor_exposures.get(beta_key, 0)
                        factor_return = factor_returns.get(return_key, 0)
                        factor_pnl[beta_key.replace('_beta', '_factor_pnl')] = beta * factor_return * prev_pos['market_value']

                # Calculate unexplained P&L
                total_explained_pnl = delta_pnl + gamma_pnl + theta_pnl + vega_pnl + rho_pnl
                for factor_contrib in factor_pnl.values():
                    total_explained_pnl += factor_contrib

                unexplained_pnl = actual_pnl - total_explained_pnl

                # Store attribution results
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR REPLACE INTO daily_pnl_attribution
                    (date, symbol, total_pnl, delta_pnl, gamma_pnl, theta_pnl, vega_pnl, rho_pnl,
                     market_factor_pnl, tech_factor_pnl, size_factor_pnl, vol_factor_pnl,
                     crypto_factor_pnl, gold_factor_pnl, bond_factor_pnl, dollar_factor_pnl, unexplained_pnl)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    date, symbol, actual_pnl, delta_pnl, gamma_pnl, theta_pnl, vega_pnl, rho_pnl,
                    factor_pnl.get('market_factor_pnl', 0), factor_pnl.get('tech_factor_pnl', 0),
                    factor_pnl.get('size_factor_pnl', 0), factor_pnl.get('vol_factor_pnl', 0),
                    factor_pnl.get('crypto_factor_pnl', 0), factor_pnl.get('gold_factor_pnl', 0),
                    factor_pnl.get('bond_factor_pnl', 0), factor_pnl.get('dollar_factor_pnl', 0),
                    unexplained_pnl
                ))

                attribution_results[symbol] = {
                    'total_pnl': actual_pnl,
                    'delta_pnl': delta_pnl,
                    'gamma_pnl': gamma_pnl,
                    'theta_pnl': theta_pnl,
                    'vega_pnl': vega_pnl,
                    'factor_pnl': factor_pnl,
                    'unexplained_pnl': unexplained_pnl,
                    'explanation_ratio': (total_explained_pnl / actual_pnl) if actual_pnl != 0 else 0
                }

            conn.commit()
            conn.close()

            logger.info(f"Calculated P&L attribution for {len(attribution_results)} positions on {date}")
            return attribution_results

        except Exception as e:
            logger.error(f"Error calculating P&L attribution for {date}: {e}")
            return {}

    def get_portfolio_pnl_summary(self, start_date: datetime.date, end_date: datetime.date) -> Dict:
        """Get comprehensive P&L attribution summary for date range"""
        try:
            conn = sqlite3.connect(self.db_path)

            query = '''
                SELECT
                    SUM(total_pnl) as total_pnl,
                    SUM(delta_pnl) as total_delta_pnl,
                    SUM(gamma_pnl) as total_gamma_pnl,
                    SUM(theta_pnl) as total_theta_pnl,
                    SUM(vega_pnl) as total_vega_pnl,
                    SUM(market_factor_pnl) as total_market_pnl,
                    SUM(tech_factor_pnl) as total_tech_pnl,
                    SUM(crypto_factor_pnl) as total_crypto_pnl,
                    SUM(unexplained_pnl) as total_unexplained_pnl,
                    COUNT(*) as days_analyzed
                FROM daily_pnl_attribution
                WHERE date BETWEEN ? AND ?
            '''

            result = pd.read_sql_query(query, conn, params=(start_date, end_date))
            conn.close()

            if result.empty:
                return {'error': 'No P&L attribution data found for date range'}

            summary = result.iloc[0].to_dict()

            # Calculate attribution percentages
            total_pnl = summary.get('total_pnl', 0)
            if total_pnl and total_pnl != 0:
                summary['delta_contribution_pct'] = (summary.get('total_delta_pnl', 0) or 0) / total_pnl * 100
                summary['gamma_contribution_pct'] = (summary.get('total_gamma_pnl', 0) or 0) / total_pnl * 100
                summary['theta_contribution_pct'] = (summary.get('total_theta_pnl', 0) or 0) / total_pnl * 100
                summary['vega_contribution_pct'] = (summary.get('total_vega_pnl', 0) or 0) / total_pnl * 100
                summary['market_contribution_pct'] = (summary.get('total_market_pnl', 0) or 0) / total_pnl * 100
                summary['crypto_contribution_pct'] = (summary.get('total_crypto_pnl', 0) or 0) / total_pnl * 100
                summary['unexplained_pct'] = (summary.get('total_unexplained_pnl', 0) or 0) / total_pnl * 100
            else:
                # No P&L data available
                summary['delta_contribution_pct'] = 0
                summary['gamma_contribution_pct'] = 0
                summary['theta_contribution_pct'] = 0
                summary['vega_contribution_pct'] = 0
                summary['market_contribution_pct'] = 0
                summary['crypto_contribution_pct'] = 0
                summary['unexplained_pct'] = 0

            summary['start_date'] = start_date.strftime('%Y-%m-%d')
            summary['end_date'] = end_date.strftime('%Y-%m-%d')

            return summary

        except Exception as e:
            logger.error(f"Error getting P&L summary: {e}")
            return {'error': str(e)}

    def analyze_greeks_accuracy(self, date: datetime.date) -> Dict:
        """Analyze how accurate Greeks predictions were vs actual P&L"""
        try:
            conn = sqlite3.connect(self.db_path)

            # Get P&L attribution data
            query = '''
                SELECT symbol, total_pnl, delta_pnl, gamma_pnl, theta_pnl, vega_pnl
                FROM daily_pnl_attribution
                WHERE date = ?
            '''
            attribution_data = pd.read_sql_query(query, conn, params=(date,))

            if attribution_data.empty:
                return {'error': f'No attribution data for {date}'}

            accuracy_results = {}

            for _, row in attribution_data.iterrows():
                symbol = row['symbol']
                actual_pnl = row['total_pnl']
                predicted_pnl = row['delta_pnl'] + row['gamma_pnl'] + row['theta_pnl'] + row['vega_pnl']

                accuracy = (predicted_pnl / actual_pnl) if actual_pnl != 0 else 0

                accuracy_results[symbol] = {
                    'actual_pnl': actual_pnl,
                    'predicted_pnl': predicted_pnl,
                    'accuracy_ratio': accuracy,
                    'prediction_error': abs(actual_pnl - predicted_pnl),
                    'delta_contribution': row['delta_pnl'] / predicted_pnl if predicted_pnl != 0 else 0,
                    'gamma_contribution': row['gamma_pnl'] / predicted_pnl if predicted_pnl != 0 else 0
                }

            conn.close()
            return accuracy_results

        except Exception as e:
            logger.error(f"Error analyzing Greeks accuracy: {e}")
            return {'error': str(e)}

    def run_daily_attribution(self, positions: List[Dict], date: datetime.date = None) -> Dict:
        """Run complete daily P&L attribution process"""
        if date is None:
            date = datetime.now().date()

        logger.info(f"Running daily P&L attribution for {date}")

        # Step 1: Capture current portfolio snapshot
        snapshot_success = self.capture_portfolio_snapshot(positions)

        # Step 2: Calculate P&L attribution (if we have previous day data)
        attribution_results = self.calculate_daily_pnl_attribution(date)

        # Step 3: Analyze Greeks accuracy
        accuracy_analysis = self.analyze_greeks_accuracy(date)

        return {
            'date': date.strftime('%Y-%m-%d'),
            'snapshot_captured': snapshot_success,
            'attribution_results': attribution_results,
            'accuracy_analysis': accuracy_analysis,
            'total_positions_analyzed': len(attribution_results)
        }


# Global instance
real_pnl_attributor = RealPnLAttributor()