# Unispark Z2 System - Advanced Portfolio Management

A professional portfolio management system with **real market data integration** - completely eliminating simulated data patterns with authentic financial analysis.

## 🚀 Real Market Data Integration

**All volatility, P&L attribution, and risk calculations now use REAL market data:**

### ✅ Real Volatility System
- **Historical Volatility**: Real market calculations from Yahoo Finance price data
- **Implied Volatility**: Extracted from actual option chain data
- **Rolling Windows**: 30/60/90/252 day realized volatility calculations
- **No More Hardcoded Values**: Dynamic volatility based on market conditions

### ✅ Factor Regression Model
- **8-Factor Systematic Risk Model**: SPY, QQQ, IWM, VIX, BTC-USD, GLD, TLT, DXY
- **Real Beta Calculations**: Linear regression on actual market factor returns
- **Factor Attribution**: Track portfolio exposures to systematic risk sources
- **Alpha & Tracking Error**: Professional risk-adjusted performance metrics

### ✅ Real P&L Attribution
- **Daily P&L Breakdown**: Greeks-based attribution (Delta, Gamma, Theta, Vega)
- **Factor P&L Impact**: Systematic factor contribution to portfolio P&L
- **Greeks Accuracy Analysis**: Compare predicted vs actual P&L
- **Portfolio Snapshots**: Track day-over-day position changes

### 🔧 Professional Features

- **Real-time Options Greeks Monitoring** - Calculated with market-based volatility
- **Market-Based Option Pricing** - Uses real implied volatility and market prices
- **Professional Risk Management** - VaR with real correlation matrices
- **MSTR-BTC Factor Analysis** - Real correlation tracking and factor exposures
- **Historical Data Accumulation** - Professional time series data management
- **Advanced Portfolio Analytics** - Institutional-grade risk analytics

## Quick Start

### Prerequisites

- Python 3.11+
- Internet connection for Yahoo Finance API
- All dependencies included in requirements.txt

### Installation

1. Clone the repository:
```bash
git clone https://github.com/XavierFan123/portfolio-management.git
cd portfolio-management
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the Flask application:
```bash
python app.py
```

4. Open your browser and navigate to:
```
http://localhost:5000
```

### 🔥 Real Data System Initialization

After starting the app, initialize the real market data systems:

1. **Initialize Volatility Data**:
```bash
POST http://localhost:5000/api/volatility/initialize
```

2. **Update Factor Analysis**:
```bash
POST http://localhost:5000/api/factor/update
```

3. **Run P&L Attribution**:
```bash
POST http://localhost:5000/api/pnl/run-attribution
```

### 📊 Key API Endpoints

- `GET /api/portfolio` - Portfolio overview with real Greeks
- `GET /api/volatility/summary` - Real volatility analysis
- `GET /api/factor/exposures` - Factor exposures
- `GET /api/pnl/summary` - P&L attribution summary
- `GET /api/charts/pnl-attribution` - Real P&L charts

## Configuration

The system uses a hierarchical configuration system:

1. Default configuration in `src/config.py`
2. Environment variables (see `.env.example`)
3. Optional configuration file (`config.json`)

### Key Configuration Areas

- **Risk Limits**: Portfolio delta, gamma, VaR limits
- **Model Parameters**: Option pricing model selection and parameters
- **Monitoring**: Update frequencies and alert thresholds
- **Data Sources**: Market data API configuration

## Usage Examples

### Basic Portfolio Monitoring

```python
from src.orchestrator import CerberusOrchestrator
from src.config import Config

# Initialize system
config = Config()
orchestrator = CerberusOrchestrator(config)

# Start monitoring
await orchestrator.initialize()
await orchestrator.start_monitoring()
await orchestrator.run()
```

### Calculate Portfolio Greeks

The system automatically monitors portfolio Greeks, but you can also request calculations:

```python
# Portfolio Greeks are calculated automatically
# Check logs for real-time Greeks updates
```

### Risk Limit Monitoring

Risk limits are monitored continuously:
- Delta exposure limits
- Gamma exposure limits
- VaR limits
- Position concentration limits

## System Components

### Data Layer
- Yahoo Finance API connector
- Redis caching (optional)
- Real-time data streaming
- Historical data management

### Computational Layer
- Black-Scholes option pricing
- Heston stochastic volatility model
- Jump diffusion models
- Monte Carlo simulation
- Greeks calculation engine

### Risk Management Layer
- Real-time risk monitoring
- Automated hedge recommendations
- Limit breach detection
- P&L attribution

### Validation Layer
- Model backtesting
- VaR backtesting (Kupiec test)
- Performance validation
- Governance oversight

## Monitoring and Alerts

The system provides multiple monitoring capabilities:

### Real-time Monitoring
- Portfolio Greeks every 10 seconds
- Risk limit checks every 30 seconds
- System health checks every 30 seconds

### Alert Types
- Risk limit breaches
- Model validation failures
- System performance issues
- Data quality problems

### Performance Metrics
- Message processing rates
- Model calculation times
- Cache hit rates
- System resource usage

## MSTR-BTC Specific Features

Special handling for MSTR (MicroStrategy) and Bitcoin correlation:

- **Basis Risk Monitoring**: Track MSTR/BTC price ratio
- **Jump Risk Analysis**: Monitor weekend gap risk
- **Volatility Surface**: MSTR-specific implied volatility analysis
- **Leverage Analysis**: Track MSTR's effective Bitcoin leverage

## Architecture Details

### Agent Communication
- Asynchronous message passing
- Priority-based message queuing
- Error handling and recovery
- Health monitoring

### Data Flow
1. Market data ingestion (Yahoo Finance)
2. Real-time processing and caching
3. Risk calculation and Greeks computation
4. Monitoring and alerting
5. Model validation and governance

### Scalability
- Modular agent architecture
- Async/await for concurrency
- Configurable update frequencies
- Optional Redis caching for performance

## Development

### Adding New Models
1. Implement model in `src/agents/quant_modeler.py`
2. Add model validation in `src/agents/model_validation_agent.py`
3. Update configuration in `src/config.py`

### Adding New Risk Metrics
1. Implement calculation in `src/agents/greeks_risk_agent.py`
2. Add monitoring logic
3. Update risk limits configuration

### Extending Data Sources
1. Create new connector in `src/data/`
2. Update `src/agents/data_infrastructure_agent.py`
3. Add configuration options

## Security Considerations

- API key management through environment variables
- Optional data encryption
- Rate limiting for external API calls
- Secure configuration management

## Performance Optimization

- Redis caching for frequently accessed data
- Parallel computation using asyncio
- Efficient data structures
- Configurable update frequencies

## Troubleshooting

### Common Issues

1. **Redis Connection Failed**: Redis is optional; system will run without it
2. **Market Data Errors**: Check Yahoo Finance API availability
3. **High Memory Usage**: Adjust cache settings and data retention periods
4. **Agent Communication Timeout**: Check system load and adjust timeouts

### Logging

Check the following log files:
- `cerberus.log` - Main system log
- Console output for real-time monitoring

### Health Checks

The system provides comprehensive health monitoring:
- Agent status monitoring
- Data feed quality checks
- Model validation status
- System resource monitoring

## License

[Add your license information here]

## Contributing

[Add contribution guidelines here]

## Support

[Add support contact information here]