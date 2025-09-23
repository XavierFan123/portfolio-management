"""
Real Historical Volatility Calculation System
Replaces all hardcoded volatility with true market-based calculations
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

class RealVolatilityCalculator:
    """Professional volatility calculation using real market data"""

    def __init__(self, db_path: str = 'volatility_data.db'):
        self.db_path = db_path
        self.init_database()

    def init_database(self):
        """Initialize volatility database tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Historical prices table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS historical_prices (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol VARCHAR(20) NOT NULL,
                date DATE NOT NULL,
                open_price REAL NOT NULL,
                high_price REAL NOT NULL,
                low_price REAL NOT NULL,
                close_price REAL NOT NULL,
                adj_close REAL NOT NULL,
                volume REAL,
                returns REAL,
                UNIQUE(symbol, date)
            )
        ''')

        # Realized volatility table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS realized_volatility (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol VARCHAR(20) NOT NULL,
                date DATE NOT NULL,
                window_days INTEGER NOT NULL,
                volatility REAL NOT NULL,
                annualized_vol REAL NOT NULL,
                vol_type VARCHAR(20) DEFAULT 'close_to_close',
                UNIQUE(symbol, date, window_days, vol_type)
            )
        ''')

        # Implied volatility surface table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS iv_surface (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol VARCHAR(20) NOT NULL,
                date DATE NOT NULL,
                strike REAL NOT NULL,
                expiry DATE NOT NULL,
                option_type VARCHAR(4) NOT NULL,
                implied_vol REAL NOT NULL,
                bid REAL,
                ask REAL,
                last_price REAL,
                volume REAL,
                open_interest REAL,
                delta REAL,
                gamma REAL,
                theta REAL,
                vega REAL,
                UNIQUE(symbol, date, strike, expiry, option_type)
            )
        ''')

        # Volatility term structure table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS vol_term_structure (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol VARCHAR(20) NOT NULL,
                date DATE NOT NULL,
                days_to_expiry INTEGER NOT NULL,
                atm_implied_vol REAL NOT NULL,
                skew_25delta REAL,
                convexity REAL,
                UNIQUE(symbol, date, days_to_expiry)
            )
        ''')

        conn.commit()
        conn.close()
        logger.info("Real volatility database initialized")

    def fetch_and_store_historical_data(self, symbols: List[str], days: int = 252) -> Dict[str, bool]:
        """Fetch historical price data and calculate returns"""
        results = {}
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days + 50)  # Extra buffer for calculations

        for symbol in symbols:
            try:
                # Fetch data from Yahoo Finance
                ticker = yf.Ticker(symbol)
                hist_data = ticker.history(start=start_date, end=end_date)

                if hist_data.empty:
                    logger.warning(f"No historical data found for {symbol}")
                    results[symbol] = False
                    continue

                # Calculate returns
                hist_data['Returns'] = hist_data['Close'].pct_change()

                # Store in database
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()

                for date, row in hist_data.iterrows():
                    cursor.execute('''
                        INSERT OR REPLACE INTO historical_prices
                        (symbol, date, open_price, high_price, low_price, close_price,
                         adj_close, volume, returns)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        symbol, date.date(), float(row['Open']), float(row['High']),
                        float(row['Low']), float(row['Close']), float(row['Close']),
                        float(row['Volume']), float(row['Returns']) if not pd.isna(row['Returns']) else None
                    ))

                conn.commit()
                conn.close()

                logger.info(f"Stored {len(hist_data)} price records for {symbol}")
                results[symbol] = True

            except Exception as e:
                logger.error(f"Error fetching data for {symbol}: {e}")
                results[symbol] = False

        return results

    def calculate_realized_volatility(self, symbol: str, windows: List[int] = [30, 60, 90, 252]) -> Dict[int, float]:
        """Calculate rolling realized volatility for different windows"""
        try:
            conn = sqlite3.connect(self.db_path)

            # Get historical returns
            query = '''
                SELECT date, returns FROM historical_prices
                WHERE symbol = ? AND returns IS NOT NULL
                ORDER BY date
            '''
            df = pd.read_sql_query(query, conn, params=(symbol,))
            conn.close()

            if df.empty:
                logger.warning(f"No returns data for {symbol}")
                return {}

            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)

            volatilities = {}

            for window in windows:
                if len(df) < window:
                    logger.warning(f"Insufficient data for {window}-day volatility of {symbol}")
                    continue

                # Calculate rolling volatility
                rolling_vol = df['returns'].rolling(window=window, min_periods=window//2).std()

                # Annualize (assuming 252 trading days)
                annualized_vol = rolling_vol * np.sqrt(252)

                # Get current volatility (most recent value)
                current_vol = annualized_vol.dropna().iloc[-1] if not annualized_vol.dropna().empty else None

                if current_vol is not None:
                    volatilities[window] = float(current_vol)

                    # Store in database
                    conn = sqlite3.connect(self.db_path)
                    cursor = conn.cursor()

                    for date, vol in annualized_vol.dropna().items():
                        cursor.execute('''
                            INSERT OR REPLACE INTO realized_volatility
                            (symbol, date, window_days, volatility, annualized_vol, vol_type)
                            VALUES (?, ?, ?, ?, ?, ?)
                        ''', (symbol, date.date(), window, float(rolling_vol.loc[date]),
                              float(vol), 'close_to_close'))

                    conn.commit()
                    conn.close()

                    logger.info(f"Calculated {window}-day volatility for {symbol}: {current_vol:.1%}")

            return volatilities

        except Exception as e:
            logger.error(f"Error calculating volatility for {symbol}: {e}")
            return {}

    def fetch_real_implied_volatility(self, symbol: str) -> Dict:
        """Fetch real implied volatility from option chains"""
        try:
            ticker = yf.Ticker(symbol)

            # Get available expirations
            expirations = ticker.options
            if not expirations:
                logger.warning(f"No option data available for {symbol}")
                return {}

            today = datetime.now().date()
            iv_data = {}

            # Process first 5 expirations to avoid too much data
            for exp_date in expirations[:5]:
                try:
                    # Get option chain
                    option_chain = ticker.option_chain(exp_date)

                    # Process calls
                    calls = option_chain.calls
                    if not calls.empty:
                        self._store_option_data(symbol, calls, exp_date, 'call', today)

                    # Process puts
                    puts = option_chain.puts
                    if not puts.empty:
                        self._store_option_data(symbol, puts, exp_date, 'put', today)

                    # Calculate days to expiry
                    exp_datetime = datetime.strptime(exp_date, '%Y-%m-%d').date()
                    days_to_expiry = (exp_datetime - today).days

                    # Get ATM implied volatility
                    current_price = ticker.info.get('currentPrice', 0)
                    if current_price > 0:
                        atm_iv = self._get_atm_implied_vol(calls, puts, current_price)
                        if atm_iv:
                            iv_data[days_to_expiry] = atm_iv

                            # Store term structure
                            conn = sqlite3.connect(self.db_path)
                            cursor = conn.cursor()

                            cursor.execute('''
                                INSERT OR REPLACE INTO vol_term_structure
                                (symbol, date, days_to_expiry, atm_implied_vol)
                                VALUES (?, ?, ?, ?)
                            ''', (symbol, today, days_to_expiry, atm_iv))

                            conn.commit()
                            conn.close()

                except Exception as e:
                    logger.warning(f"Error processing {exp_date} for {symbol}: {e}")
                    continue

            logger.info(f"Fetched implied volatility data for {symbol}: {len(iv_data)} expirations")
            return iv_data

        except Exception as e:
            logger.error(f"Error fetching implied volatility for {symbol}: {e}")
            return {}

    def _store_option_data(self, symbol: str, options_df: pd.DataFrame, exp_date: str, option_type: str, date: datetime.date):
        """Store individual option data in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        for idx, row in options_df.iterrows():
            try:
                cursor.execute('''
                    INSERT OR REPLACE INTO iv_surface
                    (symbol, date, strike, expiry, option_type, implied_vol,
                     bid, ask, last_price, volume, open_interest)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    symbol, date, float(row['strike']), exp_date, option_type,
                    float(row.get('impliedVolatility', 0)) if pd.notna(row.get('impliedVolatility')) else None,
                    float(row.get('bid', 0)) if pd.notna(row.get('bid')) else None,
                    float(row.get('ask', 0)) if pd.notna(row.get('ask')) else None,
                    float(row.get('lastPrice', 0)) if pd.notna(row.get('lastPrice')) else None,
                    float(row.get('volume', 0)) if pd.notna(row.get('volume')) else None,
                    float(row.get('openInterest', 0)) if pd.notna(row.get('openInterest')) else None
                ))
            except Exception as e:
                logger.warning(f"Error storing option data: {e}")
                continue

        conn.commit()
        conn.close()

    def _get_atm_implied_vol(self, calls: pd.DataFrame, puts: pd.DataFrame, current_price: float) -> Optional[float]:
        """Get ATM implied volatility from option chain"""
        try:
            # Find closest strikes to current price
            all_strikes = pd.concat([calls['strike'], puts['strike']]).unique()
            closest_strike = min(all_strikes, key=lambda x: abs(x - current_price))

            # Get IV for closest strike from both calls and puts
            call_iv = calls[calls['strike'] == closest_strike]['impliedVolatility']
            put_iv = puts[puts['strike'] == closest_strike]['impliedVolatility']

            ivs = []
            if not call_iv.empty and pd.notna(call_iv.iloc[0]):
                ivs.append(float(call_iv.iloc[0]))
            if not put_iv.empty and pd.notna(put_iv.iloc[0]):
                ivs.append(float(put_iv.iloc[0]))

            if ivs:
                return np.mean(ivs)

            return None

        except Exception as e:
            logger.warning(f"Error calculating ATM IV: {e}")
            return None

    def get_current_volatility(self, symbol: str, window_days: int = 30, vol_type: str = 'implied') -> Optional[float]:
        """Get current volatility for a symbol"""
        try:
            conn = sqlite3.connect(self.db_path)

            if vol_type == 'implied':
                # Get most recent implied volatility
                query = '''
                    SELECT atm_implied_vol FROM vol_term_structure
                    WHERE symbol = ? AND days_to_expiry >= ?
                    ORDER BY date DESC, days_to_expiry ASC
                    LIMIT 1
                '''
                cursor = conn.cursor()
                cursor.execute(query, (symbol, window_days))
                result = cursor.fetchone()

                if result:
                    conn.close()
                    return float(result[0])

            # Fallback to realized volatility
            query = '''
                SELECT annualized_vol FROM realized_volatility
                WHERE symbol = ? AND window_days = ?
                ORDER BY date DESC
                LIMIT 1
            '''
            cursor = conn.cursor()
            cursor.execute(query, (symbol, window_days))
            result = cursor.fetchone()

            conn.close()

            if result:
                return float(result[0])

            return None

        except Exception as e:
            logger.error(f"Error getting volatility for {symbol}: {e}")
            return None

    def update_all_volatilities(self, symbols: List[str]):
        """Update all volatility data for given symbols"""
        logger.info(f"Starting volatility update for {len(symbols)} symbols")

        # Step 1: Fetch historical price data
        price_results = self.fetch_and_store_historical_data(symbols)

        # Step 2: Calculate realized volatilities
        for symbol in symbols:
            if price_results.get(symbol, False):
                self.calculate_realized_volatility(symbol)

        # Step 3: Fetch implied volatilities for options symbols
        option_symbols = [s for s in symbols if s in ['MSTR', 'TSLA', 'NVDA', 'AAPL', 'SPY', 'QQQ']]
        for symbol in option_symbols:
            self.fetch_real_implied_volatility(symbol)

        logger.info("Volatility update completed")

    def get_volatility_summary(self, symbol: str) -> Dict:
        """Get comprehensive volatility summary for a symbol"""
        try:
            summary = {
                'symbol': symbol,
                'realized_30d': self.get_current_volatility(symbol, 30, 'realized'),
                'realized_60d': self.get_current_volatility(symbol, 60, 'realized'),
                'realized_90d': self.get_current_volatility(symbol, 90, 'realized'),
                'implied_30d': self.get_current_volatility(symbol, 30, 'implied'),
                'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }

            # Calculate volatility of volatility
            conn = sqlite3.connect(self.db_path)
            query = '''
                SELECT annualized_vol FROM realized_volatility
                WHERE symbol = ? AND window_days = 30
                ORDER BY date DESC
                LIMIT 20
            '''
            df = pd.read_sql_query(query, conn, params=(symbol,))
            conn.close()

            if not df.empty:
                vol_of_vol = df['annualized_vol'].std()
                summary['vol_of_vol'] = float(vol_of_vol)

            return summary

        except Exception as e:
            logger.error(f"Error getting volatility summary for {symbol}: {e}")
            return {'symbol': symbol, 'error': str(e)}


# Global instance
real_volatility_calculator = RealVolatilityCalculator()