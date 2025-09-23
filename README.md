# Unispark Z2 System - Advanced Portfolio Management

A sophisticated multi-agent portfolio management and risk control system designed for options trading and real-time risk monitoring.

## System Architecture

Project Cerberus consists of five specialized agents working together under a central orchestrator:

### Agents

1. **Chief Architect Agent** - System coordination and high-level decision making
2. **Quantitative Modeler Agent** - Advanced risk models and option pricing
3. **Greeks & Real-time Risk Agent** - Real-time Greeks monitoring and dynamic hedging
4. **Data & Infrastructure Agent** - Data management and system performance
5. **Model Validation & Governance Agent** - Model validation and compliance oversight

### Features

- **Real-time Options Greeks Monitoring** - Delta, Gamma, Vega, Theta, Rho tracking
- **Advanced Option Pricing Models** - Black-Scholes, Heston, Jump Diffusion
- **Risk Management** - VaR calculation, stress testing, limit monitoring
- **MSTR-BTC Basis Risk Analysis** - Specialized monitoring for MSTR/Bitcoin correlation
- **Automated Backtesting** - Model validation and performance analysis
- **Real-time Data Feeds** - Yahoo Finance API integration
- **Performance Monitoring** - System health and performance metrics

## Quick Start

### Prerequisites

- Python 3.11+
- Optional: Redis for caching
- Market data access (Yahoo Finance API included)

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd portfoliomanagement
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

4. Run the system:
```bash
python main.py
```

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