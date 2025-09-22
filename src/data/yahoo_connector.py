"""
Yahoo Finance API Data Connector
Handles real-time and historical market data retrieval
"""

import yfinance as yf
import pandas as pd
import numpy as np
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import aiohttp

logger = logging.getLogger(__name__)

@dataclass
class MarketData:
    symbol: str
    timestamp: datetime
    price: float
    volume: int
    bid: float
    ask: float
    high: float
    low: float
    open: float

@dataclass
class OptionData:
    symbol: str
    strike: float
    expiry: datetime
    option_type: str  # 'call' or 'put'
    bid: float
    ask: float
    last_price: float
    volume: int
    open_interest: int
    implied_volatility: float

class YahooFinanceConnector:
    """Enhanced Yahoo Finance data connector with real-time capabilities"""

    def __init__(self):
        self.cache = {}
        self.session = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    def get_stock_data(self, symbol: str, period: str = "1d", interval: str = "1m") -> pd.DataFrame:
        """Get historical stock data"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval=interval)

            if data.empty:
                logger.warning(f"No data found for symbol: {symbol}")
                return pd.DataFrame()

            data['symbol'] = symbol
            logger.info(f"Retrieved {len(data)} records for {symbol}")
            return data

        except Exception as e:
            logger.error(f"Error fetching stock data for {symbol}: {e}")
            return pd.DataFrame()

    def get_real_time_quote(self, symbol: str) -> Optional[MarketData]:
        """Get real-time quote for a symbol"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info

            if not info:
                logger.warning(f"No real-time data for {symbol}")
                return None

            # Try multiple price fields for better data availability
            current_price = (info.get('currentPrice') or
                           info.get('regularMarketPrice') or
                           info.get('previousClose') or 0.0)

            return MarketData(
                symbol=symbol,
                timestamp=datetime.now(),
                price=current_price,
                volume=info.get('volume', 0),
                bid=info.get('bid', 0.0),
                ask=info.get('ask', 0.0),
                high=info.get('dayHigh', 0.0),
                low=info.get('dayLow', 0.0),
                open=info.get('open', 0.0)
            )

        except Exception as e:
            logger.error(f"Error fetching real-time quote for {symbol}: {e}")
            return None

    def get_options_chain(self, symbol: str, expiry_date: Optional[str] = None) -> List[OptionData]:
        """Get options chain for a symbol"""
        try:
            ticker = yf.Ticker(symbol)

            if expiry_date:
                options = ticker.option_chain(expiry_date)
            else:
                options = ticker.option_chain()

            option_data = []

            # Process calls
            for _, row in options.calls.iterrows():
                option_data.append(OptionData(
                    symbol=symbol,
                    strike=row['strike'],
                    expiry=datetime.strptime(expiry_date or ticker.options[0], '%Y-%m-%d'),
                    option_type='call',
                    bid=row.get('bid', 0.0),
                    ask=row.get('ask', 0.0),
                    last_price=row.get('lastPrice', 0.0),
                    volume=row.get('volume', 0),
                    open_interest=row.get('openInterest', 0),
                    implied_volatility=row.get('impliedVolatility', 0.0)
                ))

            # Process puts
            for _, row in options.puts.iterrows():
                option_data.append(OptionData(
                    symbol=symbol,
                    strike=row['strike'],
                    expiry=datetime.strptime(expiry_date or ticker.options[0], '%Y-%m-%d'),
                    option_type='put',
                    bid=row.get('bid', 0.0),
                    ask=row.get('ask', 0.0),
                    last_price=row.get('lastPrice', 0.0),
                    volume=row.get('volume', 0),
                    open_interest=row.get('openInterest', 0),
                    implied_volatility=row.get('impliedVolatility', 0.0)
                ))

            logger.info(f"Retrieved {len(option_data)} options for {symbol}")
            return option_data

        except Exception as e:
            logger.error(f"Error fetching options chain for {symbol}: {e}")
            return []

    def get_crypto_data(self, symbol: str) -> Optional[MarketData]:
        """Get cryptocurrency data (BTC, ETH, etc.)"""
        try:
            # Yahoo Finance uses specific format for crypto
            crypto_symbol = f"{symbol}-USD"
            ticker = yf.Ticker(crypto_symbol)
            info = ticker.info

            if not info:
                logger.warning(f"No crypto data for {symbol}")
                return None

            return MarketData(
                symbol=symbol,
                timestamp=datetime.now(),
                price=info.get('regularMarketPrice', 0.0),
                volume=info.get('regularMarketVolume', 0),
                bid=info.get('bid', 0.0),
                ask=info.get('ask', 0.0),
                high=info.get('regularMarketDayHigh', 0.0),
                low=info.get('regularMarketDayLow', 0.0),
                open=info.get('regularMarketOpen', 0.0)
            )

        except Exception as e:
            logger.error(f"Error fetching crypto data for {symbol}: {e}")
            return None

    def calculate_mstr_btc_ratio(self) -> Optional[float]:
        """Calculate MSTR to BTC price ratio for basis risk monitoring"""
        try:
            mstr_data = self.get_real_time_quote("MSTR")
            btc_data = self.get_crypto_data("BTC")

            if mstr_data and btc_data and btc_data.price > 0:
                ratio = mstr_data.price / btc_data.price
                logger.info(f"MSTR/BTC ratio: {ratio:.6f}")
                return ratio

            return None

        except Exception as e:
            logger.error(f"Error calculating MSTR/BTC ratio: {e}")
            return None

    def get_volatility_data(self, symbol: str, days: int = 30) -> Optional[float]:
        """Calculate historical volatility"""
        try:
            data = self.get_stock_data(symbol, period=f"{days}d", interval="1d")

            if data.empty or len(data) < 2:
                return None

            # Calculate daily returns
            data['returns'] = data['Close'].pct_change().dropna()

            # Annualized volatility
            volatility = data['returns'].std() * np.sqrt(252)

            logger.info(f"{days}-day volatility for {symbol}: {volatility:.4f}")
            return volatility

        except Exception as e:
            logger.error(f"Error calculating volatility for {symbol}: {e}")
            return None

    async def stream_real_time_data(self, symbols: List[str], callback):
        """Stream real-time data for multiple symbols"""
        while True:
            try:
                for symbol in symbols:
                    data = self.get_real_time_quote(symbol)
                    if data:
                        await callback(data)

                await asyncio.sleep(1)  # Update every second

            except Exception as e:
                logger.error(f"Error in real-time streaming: {e}")
                await asyncio.sleep(5)  # Wait before retrying