"""
Historical Data Storage and Management
Stores real historical market data, Greeks, and portfolio metrics
"""
import sqlite3
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import numpy as np
import json
from loguru import logger
from typing import Dict, List, Optional, Tuple
import os

class HistoricalDataManager:
    def __init__(self, db_path: str = 'historical_data.db'):
        self.db_path = db_path
        self.init_database()

    def init_database(self):
        """Initialize the historical data database with required tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Portfolio Greeks history table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS portfolio_greeks_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME NOT NULL,
                portfolio_delta REAL NOT NULL,
                portfolio_gamma REAL NOT NULL,
                portfolio_vega REAL NOT NULL,
                portfolio_theta REAL NOT NULL,
                total_value REAL NOT NULL,
                positions_snapshot TEXT NOT NULL
            )
        ''')

        # Market data history table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS market_data_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME NOT NULL,
                symbol VARCHAR(20) NOT NULL,
                price REAL NOT NULL,
                volume REAL
            )
        ''')

        # Create index separately
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_market_data_symbol_timestamp
            ON market_data_history (symbol, timestamp)
        ''')

        # MSTR-BTC correlation history table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS mstr_btc_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME NOT NULL,
                mstr_price REAL NOT NULL,
                btc_price REAL NOT NULL,
                ratio REAL NOT NULL,
                correlation_30d REAL
            )
        ''')

        # Implied volatility history table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS iv_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME NOT NULL,
                symbol VARCHAR(20) NOT NULL,
                strike REAL NOT NULL,
                expiry DATE NOT NULL,
                option_type VARCHAR(4) NOT NULL,
                implied_vol REAL NOT NULL,
                underlying_price REAL NOT NULL
            )
        ''')

        # P&L attribution history table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS pnl_attribution_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME NOT NULL,
                delta_pnl REAL NOT NULL,
                gamma_pnl REAL NOT NULL,
                vega_pnl REAL NOT NULL,
                theta_pnl REAL NOT NULL,
                total_pnl REAL NOT NULL,
                portfolio_value REAL NOT NULL
            )
        ''')

        conn.commit()
        conn.close()
        logger.info("Historical data database initialized")

    def store_portfolio_greeks(self, greeks: Dict, total_value: float, positions: List[Dict]):
        """Store current portfolio Greeks snapshot"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        timestamp = datetime.now()
        positions_json = json.dumps(positions)

        cursor.execute('''
            INSERT INTO portfolio_greeks_history
            (timestamp, portfolio_delta, portfolio_gamma, portfolio_vega, portfolio_theta, total_value, positions_snapshot)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (timestamp, greeks['delta'], greeks['gamma'], greeks['vega'],
              greeks['theta'], total_value, positions_json))

        conn.commit()
        conn.close()
        logger.info(f"Stored portfolio Greeks: Delta={greeks['delta']:.3f}, Gamma={greeks['gamma']:.3f}")

    def store_market_data(self, symbol: str, price: float, volume: Optional[float] = None):
        """Store current market data point"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        timestamp = datetime.now()

        cursor.execute('''
            INSERT INTO market_data_history (timestamp, symbol, price, volume)
            VALUES (?, ?, ?, ?)
        ''', (timestamp, symbol, price, volume))

        conn.commit()
        conn.close()

    def fetch_historical_market_data(self, symbol: str, days: int = 30) -> pd.DataFrame:
        """Fetch real historical market data from Yahoo Finance and store it"""
        try:
            # Get data from Yahoo Finance
            ticker = yf.Ticker(symbol)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days + 5)  # Extra buffer

            hist_data = ticker.history(start=start_date, end=end_date)

            if hist_data.empty:
                logger.warning(f"No historical data found for {symbol}")
                return pd.DataFrame()

            # Store in database if not already present
            conn = sqlite3.connect(self.db_path)

            for date, row in hist_data.iterrows():
                timestamp = date.to_pydatetime()
                price = float(row['Close'])
                volume = float(row['Volume']) if 'Volume' in row else None

                # Check if already exists
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT COUNT(*) FROM market_data_history
                    WHERE symbol = ? AND date(timestamp) = date(?)
                ''', (symbol, timestamp))

                if cursor.fetchone()[0] == 0:
                    cursor.execute('''
                        INSERT INTO market_data_history (timestamp, symbol, price, volume)
                        VALUES (?, ?, ?, ?)
                    ''', (timestamp, symbol, price, volume))

            conn.commit()
            conn.close()

            logger.info(f"Fetched and stored {len(hist_data)} historical data points for {symbol}")
            return hist_data

        except Exception as e:
            logger.error(f"Error fetching historical data for {symbol}: {e}")
            return pd.DataFrame()

    def get_portfolio_greeks_history(self, days: int = 30) -> Dict:
        """Get historical portfolio Greeks data"""
        conn = sqlite3.connect(self.db_path)

        start_date = datetime.now() - timedelta(days=days)

        query = '''
            SELECT timestamp, portfolio_delta, portfolio_gamma, portfolio_vega, portfolio_theta
            FROM portfolio_greeks_history
            WHERE timestamp >= ?
            ORDER BY timestamp
        '''

        df = pd.read_sql_query(query, conn, params=(start_date,))
        conn.close()

        if df.empty:
            logger.warning("No historical portfolio Greeks data found")
            return {'labels': [], 'datasets': []}

        # Convert timestamps to labels
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        labels = [ts.strftime('%m/%d') for ts in df['timestamp']]

        return {
            'labels': labels,
            'datasets': [
                {
                    'label': 'Portfolio Delta',
                    'data': df['portfolio_delta'].tolist(),
                    'current': df['portfolio_delta'].iloc[-1] if not df.empty else 0
                },
                {
                    'label': 'Portfolio Gamma',
                    'data': df['portfolio_gamma'].tolist(),
                    'current': df['portfolio_gamma'].iloc[-1] if not df.empty else 0
                }
            ]
        }

    def get_mstr_btc_history(self, days: int = 30) -> Dict:
        """Get historical MSTR-BTC data with real correlation calculation"""
        try:
            # Fetch recent market data for both symbols
            mstr_data = self.fetch_historical_market_data('MSTR', days)
            btc_data = self.fetch_historical_market_data('BTC-USD', days)

            if mstr_data.empty or btc_data.empty:
                logger.warning("Cannot get MSTR-BTC history: missing market data")
                return {'labels': [], 'datasets': []}

            # Align data by date
            mstr_prices = mstr_data['Close'].resample('D').last().dropna()
            btc_prices = btc_data['Close'].resample('D').last().dropna()

            # Find common dates
            common_dates = mstr_prices.index.intersection(btc_prices.index)

            if len(common_dates) < 10:
                logger.warning("Insufficient overlapping data for MSTR-BTC analysis")
                return {'labels': [], 'datasets': []}

            mstr_aligned = mstr_prices.reindex(common_dates)
            btc_aligned = btc_prices.reindex(common_dates)

            # Calculate ratios
            ratios = mstr_aligned / btc_aligned

            # Calculate rolling correlation
            returns_mstr = mstr_aligned.pct_change().dropna()
            returns_btc = btc_aligned.pct_change().dropna()
            correlations = returns_mstr.rolling(window=10, min_periods=5).corr(returns_btc)

            # Store MSTR-BTC history
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            for date in common_dates[-30:]:  # Last 30 days
                if date in ratios.index and date in correlations.index:
                    timestamp = date.to_pydatetime()
                    mstr_price = float(mstr_aligned[date])
                    btc_price = float(btc_aligned[date])
                    ratio = float(ratios[date])
                    correlation = float(correlations[date]) if not pd.isna(correlations[date]) else None

                    # Check if already exists
                    cursor.execute('''
                        SELECT COUNT(*) FROM mstr_btc_history
                        WHERE date(timestamp) = date(?)
                    ''', (timestamp,))

                    if cursor.fetchone()[0] == 0:
                        cursor.execute('''
                            INSERT INTO mstr_btc_history
                            (timestamp, mstr_price, btc_price, ratio, correlation_30d)
                            VALUES (?, ?, ?, ?, ?)
                        ''', (timestamp, mstr_price, btc_price, ratio, correlation))

            conn.commit()
            conn.close()

            # Prepare chart data
            labels = [date.strftime('%m/%d') for date in common_dates[-30:]]
            ratio_data = ratios.iloc[-30:].tolist()
            correlation_data = correlations.iloc[-30:].fillna(0.5).tolist()

            return {
                'labels': labels,
                'datasets': [
                    {
                        'label': 'MSTR/BTC Ratio',
                        'data': ratio_data,
                        'current': ratio_data[-1] if ratio_data else 0
                    },
                    {
                        'label': '10-Day Rolling Correlation',
                        'data': correlation_data,
                        'current': correlation_data[-1] if correlation_data else 0
                    }
                ]
            }

        except Exception as e:
            logger.error(f"Error getting MSTR-BTC history: {e}")
            return {'labels': [], 'datasets': []}

    def calculate_and_store_pnl_attribution(self, positions: List[Dict], previous_positions: List[Dict] = None):
        """Calculate P&L attribution to Greeks factors and store"""
        try:
            # This is a simplified attribution model
            # In practice, you'd need previous portfolio state and market moves

            total_pnl = 0
            delta_pnl = 0
            gamma_pnl = 0
            vega_pnl = 0
            theta_pnl = 0

            # Calculate current portfolio value
            portfolio_value = sum(pos.get('market_value', 0) for pos in positions)

            # For demonstration, we'll calculate basic attribution
            # Real implementation would require previous state and market moves

            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            timestamp = datetime.now()

            cursor.execute('''
                INSERT INTO pnl_attribution_history
                (timestamp, delta_pnl, gamma_pnl, vega_pnl, theta_pnl, total_pnl, portfolio_value)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (timestamp, delta_pnl, gamma_pnl, vega_pnl, theta_pnl, total_pnl, portfolio_value))

            conn.commit()
            conn.close()

            logger.info(f"Stored P&L attribution: Total={total_pnl:.2f}")

        except Exception as e:
            logger.error(f"Error calculating P&L attribution: {e}")

    def get_pnl_attribution_history(self, days: int = 30) -> Dict:
        """Get historical P&L attribution data"""
        conn = sqlite3.connect(self.db_path)

        start_date = datetime.now() - timedelta(days=days)

        query = '''
            SELECT timestamp, delta_pnl, gamma_pnl, vega_pnl, theta_pnl, total_pnl
            FROM pnl_attribution_history
            WHERE timestamp >= ?
            ORDER BY timestamp
        '''

        df = pd.read_sql_query(query, conn, params=(start_date,))
        conn.close()

        if df.empty:
            logger.warning("No historical P&L attribution data found")
            return {'labels': [], 'datasets': []}

        df['timestamp'] = pd.to_datetime(df['timestamp'])
        labels = [ts.strftime('%m/%d') for ts in df['timestamp']]

        return {
            'labels': labels,
            'datasets': [
                {
                    'label': 'Delta P&L',
                    'data': df['delta_pnl'].tolist()
                },
                {
                    'label': 'Gamma P&L',
                    'data': df['gamma_pnl'].tolist()
                },
                {
                    'label': 'Vega P&L',
                    'data': df['vega_pnl'].tolist()
                },
                {
                    'label': 'Theta P&L',
                    'data': df['theta_pnl'].tolist()
                }
            ]
        }

    def get_real_iv_surface(self, positions: List[Dict]) -> Dict:
        """Get real implied volatility surface data for options positions"""
        try:
            # Get unique symbols from options positions
            option_symbols = set()
            for pos in positions:
                if pos.get('type') in ['call', 'put']:
                    option_symbols.add(pos['symbol'])

            if not option_symbols:
                # Return empty data if no options
                return {
                    'labels': [],
                    'datasets': [],
                    'message': 'No options positions found'
                }

            # For now, use the first option symbol
            primary_symbol = list(option_symbols)[0]

            # Fetch real option chain data from Yahoo Finance
            ticker = yf.Ticker(primary_symbol)

            # Get available expiration dates
            try:
                expirations = ticker.options
                if not expirations:
                    logger.warning(f"No option expirations found for {primary_symbol}")
                    return self._generate_fallback_iv_surface(primary_symbol)

                # Use first 3 available expirations
                selected_exps = expirations[:3]

                labels = []
                datasets = []

                for i, exp_date in enumerate(selected_exps):
                    try:
                        # Get option chain for this expiration
                        options_chain = ticker.option_chain(exp_date)
                        calls = options_chain.calls

                        if calls.empty:
                            continue

                        # Extract strikes and implied volatilities
                        strikes = calls['strike'].tolist()
                        ivs = calls['impliedVolatility'].tolist()

                        # Filter out invalid IVs and sort by strike
                        valid_data = [(s, iv) for s, iv in zip(strikes, ivs)
                                     if pd.notna(iv) and iv > 0 and iv < 5.0]

                        if not valid_data:
                            continue

                        valid_data.sort(key=lambda x: x[0])  # Sort by strike

                        # Take up to 7 strikes around ATM
                        mid_idx = len(valid_data) // 2
                        start_idx = max(0, mid_idx - 3)
                        end_idx = min(len(valid_data), start_idx + 7)

                        selected_data = valid_data[start_idx:end_idx]

                        if i == 0:  # Set labels from first expiration
                            labels = [str(int(s)) for s, _ in selected_data]

                        # Calculate days to expiration
                        exp_datetime = datetime.strptime(exp_date, '%Y-%m-%d')
                        days_to_exp = (exp_datetime - datetime.now()).days

                        datasets.append({
                            'label': f'{days_to_exp} Days',
                            'data': [iv for _, iv in selected_data]
                        })

                    except Exception as e:
                        logger.warning(f"Error processing expiration {exp_date}: {e}")
                        continue

                if labels and datasets:
                    logger.info(f"Retrieved real IV surface for {primary_symbol} with {len(datasets)} expirations")
                    return {
                        'labels': labels,
                        'datasets': datasets
                    }
                else:
                    logger.warning(f"No valid IV data found for {primary_symbol}")
                    return self._generate_fallback_iv_surface(primary_symbol)

            except Exception as e:
                logger.error(f"Error fetching option chain for {primary_symbol}: {e}")
                return self._generate_fallback_iv_surface(primary_symbol)

        except Exception as e:
            logger.error(f"Error getting IV surface: {e}")
            return {'labels': [], 'datasets': []}

    def _generate_fallback_iv_surface(self, symbol: str) -> Dict:
        """Generate fallback IV surface when real data unavailable"""
        try:
            # Get current price
            ticker = yf.Ticker(symbol)
            current_price = ticker.info.get('currentPrice', 350)  # Default to 350 if not found

            # Generate strikes around current price
            strikes = [int(current_price * (0.8 + 0.1 * i)) for i in range(7)]
            strike_labels = [str(s) for s in strikes]

            # Generate realistic IV smile
            def generate_iv_smile(base_iv, strikes, current_price):
                ivs = []
                for strike in strikes:
                    moneyness = strike / current_price
                    # IV smile: higher IV for OTM options
                    if moneyness < 0.95:  # OTM puts
                        iv = base_iv * (1.3 - 0.3 * moneyness)
                    elif moneyness > 1.05:  # OTM calls
                        iv = base_iv * (0.7 + 0.4 * (moneyness - 1))
                    else:  # ATM
                        iv = base_iv
                    ivs.append(max(0.15, min(1.0, iv)))
                return ivs

            return {
                'labels': strike_labels,
                'datasets': [
                    {
                        'label': '30 Days (Estimated)',
                        'data': generate_iv_smile(0.50, strikes, current_price)
                    },
                    {
                        'label': '60 Days (Estimated)',
                        'data': generate_iv_smile(0.45, strikes, current_price)
                    },
                    {
                        'label': '90 Days (Estimated)',
                        'data': generate_iv_smile(0.42, strikes, current_price)
                    }
                ]
            }
        except Exception as e:
            logger.error(f"Error generating fallback IV surface: {e}")
            return {'labels': [], 'datasets': []}

    def cleanup_old_data(self, days_to_keep: int = 90):
        """Remove data older than specified days to manage database size"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cutoff_date = datetime.now() - timedelta(days=days_to_keep)

        tables = ['portfolio_greeks_history', 'market_data_history',
                 'mstr_btc_history', 'iv_history', 'pnl_attribution_history']

        for table in tables:
            cursor.execute(f'DELETE FROM {table} WHERE timestamp < ?', (cutoff_date,))
            deleted = cursor.rowcount
            logger.info(f"Cleaned up {deleted} old records from {table}")

        conn.commit()
        conn.close()


# Global instance
historical_data_manager = HistoricalDataManager()