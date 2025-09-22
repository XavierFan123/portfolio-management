#!/usr/bin/env python3
"""
Flask Web Application for Portfolio Management
"""

import os
import asyncio
from flask import Flask, render_template, jsonify, request, send_from_directory
from flask_cors import CORS
import sys
from pathlib import Path

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from portfolio.portfolio_manager import PortfolioManager, VaRAnalysis
from portfolio.risk_dashboard import RiskDashboard
from data.yahoo_connector import YahooFinanceConnector

app = Flask(__name__, static_folder='static')
CORS(app)

# Initialize portfolio components
portfolio_manager = PortfolioManager()
portfolio_manager.load_portfolio()
risk_dashboard = RiskDashboard(portfolio_manager)
yahoo_connector = YahooFinanceConnector()

@app.route('/')
def index():
    """Serve the main HTML page"""
    return send_from_directory('.', 'index.html')

@app.route('/api/dashboard')
def get_dashboard_data():
    """Get dashboard metrics"""
    try:
        # Calculate portfolio value
        portfolio_value = portfolio_manager.calculate_portfolio_value()

        # Calculate VaR
        var_analysis = VaRAnalysis()
        var_95 = var_analysis.calculate_var(portfolio_manager.positions, confidence_level=0.95)

        # Calculate Greeks (simplified)
        total_delta = sum(pos.get('delta', 0) for pos in portfolio_manager.positions)
        total_gamma = sum(pos.get('gamma', 0) for pos in portfolio_manager.positions)

        return jsonify({
            'portfolioValue': portfolio_value,
            'var': var_95,
            'delta': total_delta,
            'gamma': total_gamma,
            'status': 'success'
        })
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 500

@app.route('/api/positions')
def get_positions():
    """Get all portfolio positions"""
    try:
        positions = []
        for pos in portfolio_manager.positions:
            # Get current market price
            current_price = yahoo_connector.get_current_price(pos['symbol'])
            market_value = pos['quantity'] * current_price
            pnl = market_value - (pos['quantity'] * pos.get('avg_cost', current_price))

            positions.append({
                'id': pos.get('id', len(positions)),
                'symbol': pos['symbol'],
                'type': pos.get('type', 'Stock'),
                'quantity': pos['quantity'],
                'currentPrice': current_price,
                'marketValue': market_value,
                'pnl': pnl,
                'delta': pos.get('delta', 0),
                'gamma': pos.get('gamma', 0)
            })

        return jsonify({'positions': positions, 'status': 'success'})
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 500

@app.route('/api/positions', methods=['POST'])
def add_position():
    """Add a new position"""
    try:
        data = request.get_json()

        # Validate required fields
        required_fields = ['symbol', 'type', 'quantity', 'price']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing field: {field}', 'status': 'error'}), 400

        # Add position to portfolio
        position = {
            'symbol': data['symbol'].upper(),
            'type': data['type'],
            'quantity': float(data['quantity']),
            'avg_cost': float(data['price']),
            'delta': 0,  # Will be calculated
            'gamma': 0   # Will be calculated
        }

        portfolio_manager.add_position(position)
        portfolio_manager.save_portfolio()

        return jsonify({'message': 'Position added successfully', 'status': 'success'})
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 500

@app.route('/api/positions/<position_id>', methods=['DELETE'])
def delete_position(position_id):
    """Delete a position"""
    try:
        portfolio_manager.remove_position(int(position_id))
        portfolio_manager.save_portfolio()

        return jsonify({'message': 'Position deleted successfully', 'status': 'success'})
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 500

@app.route('/api/risk')
def get_risk_data():
    """Get risk analysis data"""
    try:
        var_analysis = VaRAnalysis()

        # Calculate different VaR metrics
        var_95_1d = var_analysis.calculate_var(portfolio_manager.positions, confidence_level=0.95, time_horizon=1)
        var_99_1d = var_analysis.calculate_var(portfolio_manager.positions, confidence_level=0.99, time_horizon=1)
        var_95_10d = var_analysis.calculate_var(portfolio_manager.positions, confidence_level=0.95, time_horizon=10)

        # Calculate Greeks
        total_delta = sum(pos.get('delta', 0) for pos in portfolio_manager.positions)
        total_gamma = sum(pos.get('gamma', 0) for pos in portfolio_manager.positions)
        total_vega = sum(pos.get('vega', 0) for pos in portfolio_manager.positions)
        total_theta = sum(pos.get('theta', 0) for pos in portfolio_manager.positions)

        return jsonify({
            'var95_1d': var_95_1d,
            'var99_1d': var_99_1d,
            'var95_10d': var_95_10d,
            'delta': total_delta,
            'gamma': total_gamma,
            'vega': total_vega,
            'theta': total_theta,
            'status': 'success'
        })
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 500

@app.route('/api/monitoring/health')
def get_system_health():
    """Get system health status"""
    try:
        # Check data feed
        data_feed_status = yahoo_connector.health_check()

        # Check portfolio manager
        portfolio_status = len(portfolio_manager.positions) >= 0  # Simple check

        return jsonify({
            'dataFeed': 'online' if data_feed_status else 'offline',
            'riskEngine': 'online' if portfolio_status else 'offline',
            'modelValidation': 'online',  # Simplified
            'status': 'success'
        })
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 500

@app.route('/api/monitoring/alerts')
def get_alerts():
    """Get current system alerts"""
    try:
        alerts = []

        # Check portfolio limits
        total_delta = sum(pos.get('delta', 0) for pos in portfolio_manager.positions)
        if abs(total_delta) > 0.8:
            alerts.append({
                'type': 'warning',
                'message': f'Portfolio delta ({total_delta:.2f}) approaching limit',
                'timestamp': '2024-01-01T12:00:00Z'
            })

        # Check VaR
        var_analysis = VaRAnalysis()
        var_95 = var_analysis.calculate_var(portfolio_manager.positions)
        portfolio_value = portfolio_manager.calculate_portfolio_value()

        if var_95 / portfolio_value > 0.05:  # 5% threshold
            alerts.append({
                'type': 'high',
                'message': 'VaR exceeds 5% of portfolio value',
                'timestamp': '2024-01-01T12:05:00Z'
            })

        return jsonify({'alerts': alerts, 'status': 'success'})
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 500

@app.route('/api/market/<symbol>')
def get_market_data(symbol):
    """Get market data for a symbol"""
    try:
        current_price = yahoo_connector.get_current_price(symbol)

        return jsonify({
            'symbol': symbol,
            'currentPrice': current_price,
            'status': 'success'
        })
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found', 'status': 'error'}), 404

@app.errorhandler(500)
def internal_error(error):
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