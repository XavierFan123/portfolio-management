#!/usr/bin/env python3
"""
Flask Web Application for Portfolio Management
"""

import os
import logging
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import json
from datetime import datetime
from typing import Dict, List, Any

# Initialize Flask app
app = Flask(__name__, static_folder='static')
CORS(app)

# Add security headers
@app.after_request
def after_request(response):
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    return response

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PortfolioData:
    """Simple portfolio data manager with file persistence"""

    def __init__(self):
        self.data_file = 'portfolio_data.json'
        self.positions = []
        self.load_data()

    def load_data(self):
        """Load portfolio data from file"""
        try:
            if os.path.exists(self.data_file):
                with open(self.data_file, 'r') as f:
                    data = json.load(f)
                    self.positions = data.get('positions', [])
            else:
                # Initialize with demo data
                self.positions = [
                    {
                        'id': '1',
                        'symbol': 'MSTR',
                        'type': 'stock',
                        'quantity': 100,
                        'avg_cost': 800.0,
                        'delta': 1.0,
                        'gamma': 0.0,
                        'vega': 0.0,
                        'theta': 0.0,
                        'timestamp': datetime.now().isoformat()
                    },
                    {
                        'id': '2',
                        'symbol': 'BTC-USD',
                        'type': 'crypto',
                        'quantity': 0.5,
                        'avg_cost': 60000.0,
                        'delta': 1.0,
                        'gamma': 0.0,
                        'vega': 0.0,
                        'theta': 0.0,
                        'timestamp': datetime.now().isoformat()
                    }
                ]
                self.save_data()
        except Exception as e:
            logger.error(f"Error loading portfolio data: {e}")
            self.positions = []

    def save_data(self):
        """Save portfolio data to file"""
        try:
            data = {
                'positions': self.positions,
                'last_updated': datetime.now().isoformat()
            }
            with open(self.data_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving portfolio data: {e}")

    def add_position(self, position: Dict[str, Any]) -> str:
        """Add a new position"""
        position_id = str(len(self.positions) + 1)
        position['id'] = position_id
        position['timestamp'] = datetime.now().isoformat()

        # Calculate Greeks based on position type
        if position['type'] == 'stock' or position['type'] == 'crypto':
            position['delta'] = 1.0
            position['gamma'] = 0.0
            position['vega'] = 0.0
            position['theta'] = 0.0
        elif position['type'] == 'call':
            # Simplified option Greeks
            position['delta'] = 0.6  # Typical ATM call delta
            position['gamma'] = 0.1
            position['vega'] = 20.0
            position['theta'] = -5.0
        elif position['type'] == 'put':
            position['delta'] = -0.4  # Typical ATM put delta
            position['gamma'] = 0.1
            position['vega'] = 20.0
            position['theta'] = -5.0

        self.positions.append(position)
        self.save_data()
        return position_id

    def remove_position(self, position_id: str) -> bool:
        """Remove a position"""
        original_length = len(self.positions)
        self.positions = [pos for pos in self.positions if pos['id'] != position_id]
        if len(self.positions) < original_length:
            self.save_data()
            return True
        return False

    def get_positions(self) -> List[Dict[str, Any]]:
        """Get all positions with current market data"""
        positions_with_market_data = []

        for pos in self.positions:
            try:
                # Simulate current market prices
                current_price = self.get_mock_price(pos['symbol'], pos['type'])
                market_value = pos['quantity'] * current_price
                pnl = market_value - (pos['quantity'] * pos['avg_cost'])

                position_data = {
                    'id': pos['id'],
                    'symbol': pos['symbol'],
                    'type': pos['type'].title(),
                    'quantity': pos['quantity'],
                    'currentPrice': current_price,
                    'marketValue': market_value,
                    'pnl': pnl,
                    'delta': pos.get('delta', 0),
                    'gamma': pos.get('gamma', 0),
                    'vega': pos.get('vega', 0),
                    'theta': pos.get('theta', 0)
                }
                positions_with_market_data.append(position_data)
            except Exception as e:
                logger.error(f"Error processing position {pos.get('symbol', 'unknown')}: {e}")
                continue

        return positions_with_market_data

    def get_mock_price(self, symbol: str, position_type: str) -> float:
        """Get mock current price for symbols"""
        # Simplified price simulation
        base_prices = {
            'MSTR': 850.0,
            'BTC-USD': 65000.0,
            'TSLA': 250.0,
            'AAPL': 175.0,
            'NVDA': 450.0,
            'SPY': 420.0
        }

        base_price = base_prices.get(symbol, 100.0)

        # Add some random variation (±5%)
        import random
        variation = random.uniform(0.95, 1.05)

        if position_type in ['call', 'put']:
            # Options are much cheaper than underlying
            return base_price * 0.1 * variation
        else:
            return base_price * variation

    def calculate_portfolio_value(self) -> float:
        """Calculate total portfolio value"""
        total_value = 0
        for pos in self.positions:
            try:
                current_price = self.get_mock_price(pos['symbol'], pos['type'])
                total_value += pos['quantity'] * current_price
            except Exception as e:
                logger.error(f"Error calculating value for {pos.get('symbol', 'unknown')}: {e}")
        return total_value

    def calculate_portfolio_greeks(self) -> Dict[str, float]:
        """Calculate portfolio-level Greeks"""
        greeks = {'delta': 0, 'gamma': 0, 'vega': 0, 'theta': 0}

        for pos in self.positions:
            try:
                # Weight Greeks by position size
                weight = abs(pos['quantity'])
                greeks['delta'] += pos.get('delta', 0) * weight
                greeks['gamma'] += pos.get('gamma', 0) * weight
                greeks['vega'] += pos.get('vega', 0) * weight
                greeks['theta'] += pos.get('theta', 0) * weight
            except Exception as e:
                logger.error(f"Error calculating Greeks for {pos.get('symbol', 'unknown')}: {e}")

        return greeks

    def calculate_var(self, confidence_level: float = 0.95, time_horizon: int = 1) -> float:
        """Calculate Value at Risk"""
        portfolio_value = self.calculate_portfolio_value()

        # Simplified VaR calculation (normally distributed returns)
        if confidence_level == 0.95:
            z_score = 1.645
        elif confidence_level == 0.99:
            z_score = 2.326
        else:
            z_score = 1.645

        # Assume 2% daily volatility for simplicity
        daily_volatility = 0.02
        time_adjusted_volatility = daily_volatility * (time_horizon ** 0.5)

        var = portfolio_value * z_score * time_adjusted_volatility
        return var

# Initialize portfolio data
portfolio_data = PortfolioData()
logger.info("Portfolio data manager initialized successfully")

@app.route('/')
def index():
    """Serve the main HTML page"""
    return send_from_directory('.', 'index.html')

@app.route('/api/dashboard')
def get_dashboard_data():
    """Get dashboard metrics"""
    try:
        portfolio_value = portfolio_data.calculate_portfolio_value()
        greeks = portfolio_data.calculate_portfolio_greeks()
        var_95 = portfolio_data.calculate_var(confidence_level=0.95)

        return jsonify({
            'portfolioValue': portfolio_value,
            'var': var_95,
            'delta': greeks['delta'],
            'gamma': greeks['gamma'],
            'status': 'success'
        })
    except Exception as e:
        logger.error(f"Error in dashboard data: {e}")
        return jsonify({
            'portfolioValue': 125420.50,
            'var': 3245.80,
            'delta': 0.42,
            'gamma': 0.12,
            'status': 'error',
            'message': str(e)
        })

@app.route('/api/positions')
def get_positions():
    """Get all portfolio positions"""
    try:
        positions = portfolio_data.get_positions()
        return jsonify({'positions': positions, 'status': 'success'})
    except Exception as e:
        logger.error(f"Error in positions endpoint: {e}")
        return jsonify({
            'positions': [],
            'status': 'error',
            'message': str(e)
        })

@app.route('/api/positions', methods=['POST'])
def add_position():
    """Add a new position"""
    try:
        data = request.get_json()
        logger.info(f"Received position data: {data}")

        if not data:
            return jsonify({'error': 'No data provided', 'status': 'error'}), 400

        # Validate required fields
        required_fields = ['symbol', 'type', 'quantity']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing field: {field}', 'status': 'error'}), 400

        # Prepare position data
        position = {
            'symbol': data['symbol'].upper(),
            'type': data['type'].lower(),
            'quantity': float(data['quantity'])
        }

        # Handle different position types
        if data['type'].lower() in ['call', 'put']:
            # For options, require strike price
            if 'strike_price' not in data:
                return jsonify({'error': 'Strike price required for options', 'status': 'error'}), 400
            position['strike_price'] = float(data['strike_price'])
            position['avg_cost'] = float(data.get('premium', 10.0))  # Default premium
        else:
            # For stocks/crypto, use current market price if no cost provided
            position['avg_cost'] = float(data.get('avg_cost', portfolio_data.get_mock_price(position['symbol'], position['type'])))

        position_id = portfolio_data.add_position(position)

        return jsonify({
            'message': 'Position added successfully',
            'position_id': position_id,
            'status': 'success'
        })

    except ValueError as e:
        logger.error(f"Value error in add position: {e}")
        return jsonify({'error': 'Invalid numeric value provided', 'status': 'error'}), 400
    except Exception as e:
        logger.error(f"Error adding position: {e}")
        return jsonify({'error': f'Failed to add position: {str(e)}', 'status': 'error'}), 500

@app.route('/api/positions/<position_id>', methods=['DELETE'])
def delete_position(position_id):
    """Delete a position"""
    try:
        success = portfolio_data.remove_position(position_id)
        if success:
            return jsonify({'message': 'Position deleted successfully', 'status': 'success'})
        else:
            return jsonify({'error': 'Position not found', 'status': 'error'}), 404
    except Exception as e:
        logger.error(f"Error deleting position: {e}")
        return jsonify({'error': f'Failed to delete position: {str(e)}', 'status': 'error'}), 500

@app.route('/api/risk')
def get_risk_data():
    """Get risk analysis data"""
    try:
        greeks = portfolio_data.calculate_portfolio_greeks()

        return jsonify({
            'var95_1d': portfolio_data.calculate_var(confidence_level=0.95, time_horizon=1),
            'var99_1d': portfolio_data.calculate_var(confidence_level=0.99, time_horizon=1),
            'var95_10d': portfolio_data.calculate_var(confidence_level=0.95, time_horizon=10),
            'delta': greeks['delta'],
            'gamma': greeks['gamma'],
            'vega': greeks['vega'],
            'theta': greeks['theta'],
            'status': 'success'
        })
    except Exception as e:
        logger.error(f"Error in risk data: {e}")
        return jsonify({
            'var95_1d': 3245.80,
            'var99_1d': 4892.15,
            'var95_10d': 10267.34,
            'delta': 0.42,
            'gamma': 0.12,
            'vega': 45.8,
            'theta': -12.3,
            'status': 'error',
            'message': str(e)
        })

@app.route('/api/monitoring/health')
def get_system_health():
    """Get system health status"""
    try:
        return jsonify({
            'dataFeed': 'online',
            'riskEngine': 'online',
            'modelValidation': 'online',
            'status': 'success'
        })
    except Exception as e:
        logger.error(f"Error in health check: {e}")
        return jsonify({
            'dataFeed': 'offline',
            'riskEngine': 'offline',
            'modelValidation': 'offline',
            'status': 'error',
            'message': str(e)
        })

@app.route('/api/monitoring/alerts')
def get_alerts():
    """Get current system alerts"""
    try:
        alerts = []
        greeks = portfolio_data.calculate_portfolio_greeks()

        # Check delta limits
        if abs(greeks['delta']) > 5.0:
            alerts.append({
                'type': 'warning',
                'message': f'Portfolio delta ({greeks["delta"]:.2f}) approaching limit',
                'timestamp': datetime.now().isoformat()
            })

        # Check VaR limits
        portfolio_value = portfolio_data.calculate_portfolio_value()
        var_95 = portfolio_data.calculate_var()

        if portfolio_value > 0 and var_95 / portfolio_value > 0.05:
            alerts.append({
                'type': 'high',
                'message': f'VaR ({var_95:.0f}) exceeds 5% of portfolio value',
                'timestamp': datetime.now().isoformat()
            })

        if not alerts:
            alerts.append({
                'type': 'info',
                'message': 'All systems operational',
                'timestamp': datetime.now().isoformat()
            })

        return jsonify({'alerts': alerts, 'status': 'success'})
    except Exception as e:
        logger.error(f"Error in alerts endpoint: {e}")
        return jsonify({
            'alerts': [
                {
                    'type': 'error',
                    'message': f'Error generating alerts: {str(e)}',
                    'timestamp': datetime.now().isoformat()
                }
            ],
            'status': 'error'
        })

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found', 'status': 'error'}), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {error}")
    return jsonify({'error': 'Internal server error', 'status': 'error'}), 500

if __name__ == '__main__':
    # Get port from environment variable or default to 5000
    port = int(os.environ.get('PORT', 5000))

    # Run the Flask application
    app.run(
        host='0.0.0.0',
        port=port,
        debug=os.environ.get('FLASK_ENV') == 'development'
    )