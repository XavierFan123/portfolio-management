"""
Factor Regression Model for Real Market Risk Analysis
Calculates portfolio exposures to systematic risk factors using real market data
"""
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from loguru import logger
import warnings
warnings.filterwarnings('ignore')

class FactorRegressionAnalyzer:
    """Professional factor regression analysis using real market data"""

    def __init__(self, db_path: str = 'factor_data.db'):
        self.db_path = db_path
        self.init_database()

        # Define systematic risk factors
        self.risk_factors = {
            'SPY': 'Market Factor (S&P 500)',
            'QQQ': 'Technology Factor (Nasdaq 100)',
            'IWM': 'Small Cap Factor (Russell 2000)',
            'VIX': 'Volatility Factor',
            'BTC-USD': 'Cryptocurrency Factor',
            'GLD': 'Gold/Commodity Factor',
            'TLT': 'Treasury/Interest Rate Factor',
            'DXY': 'US Dollar Factor'
        }

    def init_database(self):
        """Initialize factor analysis database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Factor returns table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS factor_returns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date DATE NOT NULL,
                factor_symbol VARCHAR(20) NOT NULL,
                return_1d REAL NOT NULL,
                return_5d REAL,
                return_21d REAL,
                price REAL NOT NULL,
                volume REAL,
                UNIQUE(date, factor_symbol)
            )
        ''')

        # Portfolio factor exposures table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS portfolio_exposures (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date DATE NOT NULL,
                symbol VARCHAR(20) NOT NULL,
                market_beta REAL,
                tech_beta REAL,
                size_beta REAL,
                vol_beta REAL,
                crypto_beta REAL,
                gold_beta REAL,
                bond_beta REAL,
                dollar_beta REAL,
                alpha REAL,
                r_squared REAL,
                tracking_error REAL,
                window_days INTEGER DEFAULT 60,
                UNIQUE(date, symbol, window_days)
            )
        ''')

        # Risk factor correlations table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS factor_correlations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date DATE NOT NULL,
                factor1 VARCHAR(20) NOT NULL,
                factor2 VARCHAR(20) NOT NULL,
                correlation REAL NOT NULL,
                window_days INTEGER DEFAULT 30,
                UNIQUE(date, factor1, factor2, window_days)
            )
        ''')

        # Portfolio factor attribution table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS factor_attribution (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date DATE NOT NULL,
                portfolio_return REAL NOT NULL,
                market_attribution REAL,
                tech_attribution REAL,
                size_attribution REAL,
                vol_attribution REAL,
                crypto_attribution REAL,
                alpha_attribution REAL,
                unexplained_return REAL,
                total_explained REAL
            )
        ''')

        conn.commit()
        conn.close()
        logger.info("Factor regression database initialized")

    def fetch_factor_data(self, days: int = 252) -> Dict[str, bool]:
        """Fetch historical data for all risk factors"""
        results = {}
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days + 30)

        logger.info(f"Fetching factor data for {len(self.risk_factors)} factors over {days} days")

        for factor_symbol in self.risk_factors.keys():
            try:
                # Handle special cases
                if factor_symbol == 'VIX':
                    ticker_symbol = '^VIX'
                elif factor_symbol == 'DXY':
                    ticker_symbol = 'DX-Y.NYB'
                else:
                    ticker_symbol = factor_symbol

                ticker = yf.Ticker(ticker_symbol)
                hist_data = ticker.history(start=start_date, end=end_date)

                if hist_data.empty:
                    logger.warning(f"No data found for factor {factor_symbol}")
                    results[factor_symbol] = False
                    continue

                # Calculate returns
                hist_data['Return_1D'] = hist_data['Close'].pct_change()
                hist_data['Return_5D'] = hist_data['Close'].pct_change(5)
                hist_data['Return_21D'] = hist_data['Close'].pct_change(21)

                # Store in database
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()

                stored_count = 0
                for date, row in hist_data.iterrows():
                    if pd.notna(row['Return_1D']):
                        cursor.execute('''
                            INSERT OR REPLACE INTO factor_returns
                            (date, factor_symbol, return_1d, return_5d, return_21d, price, volume)
                            VALUES (?, ?, ?, ?, ?, ?, ?)
                        ''', (
                            date.date(), factor_symbol, float(row['Return_1D']),
                            float(row['Return_5D']) if pd.notna(row['Return_5D']) else None,
                            float(row['Return_21D']) if pd.notna(row['Return_21D']) else None,
                            float(row['Close']), float(row['Volume']) if 'Volume' in row else None
                        ))
                        stored_count += 1

                conn.commit()
                conn.close()

                logger.info(f"Stored {stored_count} return records for {factor_symbol}")
                results[factor_symbol] = True

            except Exception as e:
                logger.error(f"Error fetching factor data for {factor_symbol}: {e}")
                results[factor_symbol] = False

        return results

    def calculate_factor_exposures(self, symbol: str, window_days: int = 60) -> Optional[Dict]:
        """Calculate factor exposures (betas) for a symbol using regression"""
        try:
            conn = sqlite3.connect(self.db_path)

            # Get symbol returns (assuming we have this from previous modules)
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=window_days + 30)

            # Fetch symbol data
            ticker = yf.Ticker(symbol)
            symbol_data = ticker.history(start=start_date, end=end_date)

            if symbol_data.empty:
                logger.warning(f"No price data for {symbol}")
                return None

            # Remove timezone information to avoid conflicts
            symbol_data.index = symbol_data.index.tz_localize(None)
            symbol_data['Return'] = symbol_data['Close'].pct_change()
            symbol_returns = symbol_data['Return'].dropna()

            # Get factor returns for the same period
            factor_data = {}
            for factor_symbol in self.risk_factors.keys():
                query = '''
                    SELECT date, return_1d FROM factor_returns
                    WHERE factor_symbol = ? AND date >= ? AND date <= ?
                    ORDER BY date
                '''
                df = pd.read_sql_query(query, conn, params=(factor_symbol, start_date, end_date))

                if not df.empty:
                    df['date'] = pd.to_datetime(df['date'])
                    df.set_index('date', inplace=True)
                    factor_data[factor_symbol] = df['return_1d']

            conn.close()

            if len(factor_data) < 3:
                logger.warning(f"Insufficient factor data for {symbol}")
                return None

            # Align all data by date
            all_data = pd.DataFrame(symbol_returns)
            all_data.columns = ['symbol_return']

            for factor_name, factor_returns in factor_data.items():
                # Resample to daily and align dates
                factor_daily = factor_returns.resample('D').last().dropna()
                all_data = all_data.join(factor_daily.rename(factor_name), how='inner')

            # Remove rows with any NaN values
            all_data = all_data.dropna()

            if len(all_data) < window_days // 2:
                logger.warning(f"Insufficient aligned data for {symbol}: {len(all_data)} days")
                return None

            # Take only the most recent window_days
            recent_data = all_data.tail(window_days)

            # Prepare regression
            y = recent_data['symbol_return'].values
            X = recent_data.drop('symbol_return', axis=1).values

            # Fit multiple regression
            model = LinearRegression()
            model.fit(X, y)

            # Calculate metrics
            y_pred = model.predict(X)
            r2 = r2_score(y, y_pred)
            residuals = y - y_pred
            tracking_error = np.std(residuals) * np.sqrt(252)  # Annualized

            # Extract factor exposures (betas)
            factor_names = recent_data.drop('symbol_return', axis=1).columns
            exposures = {
                'symbol': symbol,
                'date': end_date,
                'alpha': float(model.intercept_ * 252),  # Annualized alpha
                'r_squared': float(r2),
                'tracking_error': float(tracking_error),
                'window_days': window_days
            }

            # Map factor betas
            factor_mapping = {
                'SPY': 'market_beta',
                'QQQ': 'tech_beta',
                'IWM': 'size_beta',
                'VIX': 'vol_beta',
                'BTC-USD': 'crypto_beta',
                'GLD': 'gold_beta',
                'TLT': 'bond_beta',
                'DXY': 'dollar_beta'
            }

            for i, factor_name in enumerate(factor_names):
                beta_key = factor_mapping.get(factor_name, f'{factor_name.lower()}_beta')
                exposures[beta_key] = float(model.coef_[i])

            # Store in database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute('''
                INSERT OR REPLACE INTO portfolio_exposures
                (date, symbol, market_beta, tech_beta, size_beta, vol_beta,
                 crypto_beta, gold_beta, bond_beta, dollar_beta, alpha, r_squared,
                 tracking_error, window_days)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                end_date, symbol,
                exposures.get('market_beta'), exposures.get('tech_beta'),
                exposures.get('size_beta'), exposures.get('vol_beta'),
                exposures.get('crypto_beta'), exposures.get('gold_beta'),
                exposures.get('bond_beta'), exposures.get('dollar_beta'),
                exposures['alpha'], exposures['r_squared'],
                exposures['tracking_error'], window_days
            ))

            conn.commit()
            conn.close()

            logger.info(f"Calculated factor exposures for {symbol}: R² = {r2:.3f}, Alpha = {exposures['alpha']:.1%}")
            return exposures

        except Exception as e:
            logger.error(f"Error calculating factor exposures for {symbol}: {e}")
            return None

    def calculate_portfolio_exposures(self, positions: List[Dict]) -> Dict:
        """Calculate aggregate portfolio factor exposures"""
        try:
            portfolio_exposures = {
                'market_beta': 0,
                'tech_beta': 0,
                'size_beta': 0,
                'vol_beta': 0,
                'crypto_beta': 0,
                'gold_beta': 0,
                'bond_beta': 0,
                'dollar_beta': 0,
                'alpha': 0,
                'avg_r_squared': 0,
                'total_value': 0
            }

            total_value = 0
            weighted_r2 = 0

            for pos in positions:
                try:
                    symbol = pos.get('symbol', '')
                    market_value = abs(pos.get('market_value', 0))

                    if market_value == 0:
                        continue

                    # Get factor exposures for this symbol
                    exposures = self.calculate_factor_exposures(symbol)

                    if exposures:
                        weight = market_value
                        total_value += weight

                        # Weight by market value
                        for key in ['market_beta', 'tech_beta', 'size_beta', 'vol_beta',
                                   'crypto_beta', 'gold_beta', 'bond_beta', 'dollar_beta', 'alpha']:
                            if key in exposures and exposures[key] is not None:
                                portfolio_exposures[key] += exposures[key] * weight

                        # Track R-squared
                        if exposures.get('r_squared'):
                            weighted_r2 += exposures['r_squared'] * weight

                except Exception as e:
                    logger.warning(f"Error processing position {pos.get('symbol', 'unknown')}: {e}")

            # Normalize by total value
            if total_value > 0:
                for key in portfolio_exposures:
                    if key not in ['total_value', 'avg_r_squared']:
                        portfolio_exposures[key] /= total_value

                portfolio_exposures['avg_r_squared'] = weighted_r2 / total_value
                portfolio_exposures['total_value'] = total_value

            return portfolio_exposures

        except Exception as e:
            logger.error(f"Error calculating portfolio exposures: {e}")
            return {}

    def get_factor_summary(self, symbol: str = None) -> Dict:
        """Get factor analysis summary"""
        try:
            conn = sqlite3.connect(self.db_path)

            if symbol:
                # Get latest exposures for specific symbol
                query = '''
                    SELECT * FROM portfolio_exposures
                    WHERE symbol = ?
                    ORDER BY date DESC
                    LIMIT 1
                '''
                df = pd.read_sql_query(query, conn, params=(symbol,))
            else:
                # Get latest factor returns
                query = '''
                    SELECT factor_symbol, return_1d, return_5d, return_21d
                    FROM factor_returns
                    WHERE date = (SELECT MAX(date) FROM factor_returns)
                '''
                df = pd.read_sql_query(query, conn)

            conn.close()

            if df.empty:
                return {'error': 'No factor data available'}

            return df.to_dict('records')[0] if symbol else df.to_dict('records')

        except Exception as e:
            logger.error(f"Error getting factor summary: {e}")
            return {'error': str(e)}

    def update_all_factors(self, symbols: List[str] = None):
        """Update factor data and calculate exposures for all symbols"""
        logger.info("Starting comprehensive factor analysis update")

        # Step 1: Update factor data
        factor_results = self.fetch_factor_data()
        successful_factors = [f for f, success in factor_results.items() if success]
        logger.info(f"Successfully updated {len(successful_factors)} factors: {successful_factors}")

        # Step 2: Calculate exposures for provided symbols
        if symbols:
            for symbol in symbols:
                exposures = self.calculate_factor_exposures(symbol)
                if exposures:
                    logger.info(f"Updated exposures for {symbol}")

        logger.info("Factor analysis update completed")


# Global instance
factor_analyzer = FactorRegressionAnalyzer()