"""
Configuration Management for Project Cerberus
Centralized configuration for all system components
"""

import os
import json
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

class Config:
    """
    Centralized configuration management for Cerberus system
    """

    def __init__(self, config_file: Optional[str] = None):
        # Load environment variables
        load_dotenv()

        # Default configuration
        self._config = {
            "system": {
                "name": "Project Cerberus",
                "version": "1.0.0",
                "debug": self._get_env_bool("DEBUG", False),
                "log_level": os.getenv("LOG_LEVEL", "INFO")
            },
            "data": {
                "yahoo_finance": {
                    "enabled": True,
                    "timeout": 30,
                    "retry_attempts": 3
                },
                "redis": {
                    "host": os.getenv("REDIS_HOST", "localhost"),
                    "port": int(os.getenv("REDIS_PORT", "6379")),
                    "db": int(os.getenv("REDIS_DB", "0")),
                    "password": os.getenv("REDIS_PASSWORD"),
                    "enabled": self._get_env_bool("REDIS_ENABLED", False)
                },
                "cache": {
                    "default_ttl": 300,  # 5 minutes
                    "market_data_ttl": 60,  # 1 minute
                    "options_data_ttl": 300  # 5 minutes
                }
            },
            "risk": {
                "limits": {
                    "max_portfolio_delta": 100000,
                    "max_portfolio_gamma": 10000,
                    "max_portfolio_vega": 50000,
                    "max_portfolio_theta": 5000,
                    "max_portfolio_var": 100000,
                    "max_single_position": 0.1,  # 10% of portfolio
                    "max_sector_concentration": 0.3  # 30% of portfolio
                },
                "monitoring": {
                    "update_frequency": 10,  # seconds
                    "alert_thresholds": {
                        "delta_warning": 0.8,  # 80% of limit
                        "delta_critical": 0.95,  # 95% of limit
                        "gamma_warning": 0.8,
                        "gamma_critical": 0.95
                    }
                },
                "var": {
                    "confidence_level": 0.95,
                    "time_horizon": 1,  # days
                    "lookback_period": 252  # trading days
                }
            },
            "models": {
                "option_pricing": {
                    "default_model": "black_scholes",
                    "available_models": ["black_scholes", "heston", "jump_diffusion"],
                    "risk_free_rate": 0.045,  # 4.5%
                    "default_volatility": 0.25  # 25%
                },
                "validation": {
                    "backtest_frequency": "monthly",
                    "var_backtest_frequency": "daily",
                    "model_approval_threshold": 70,  # percentage
                    "max_unexplained_pnl": 0.05  # 5%
                }
            },
            "agents": {
                "chief_architect": {
                    "monitoring_frequency": 30,  # seconds
                    "decision_timeout": 60,  # seconds
                    "max_concurrent_decisions": 10
                },
                "quant_modeler": {
                    "calculation_timeout": 30,  # seconds
                    "monte_carlo_simulations": 10000,
                    "numerical_precision": 1e-6
                },
                "greeks_risk_agent": {
                    "update_frequency": 10,  # seconds
                    "hedge_threshold": 50,  # delta units
                    "max_hedge_size": 1000
                },
                "data_infrastructure": {
                    "health_check_frequency": 30,  # seconds
                    "data_quality_threshold": 0.95,  # 95%
                    "max_latency_ms": 1000
                },
                "model_validation": {
                    "validation_frequency": 86400,  # daily
                    "backtest_lookback": 252,  # trading days
                    "stress_test_scenarios": 100
                }
            },
            "trading": {
                "symbols": {
                    "primary": ["MSTR", "QQQ"],
                    "benchmark": ["SPY"],
                    "crypto": ["BTC-USD", "ETH-USD"]
                },
                "market_hours": {
                    "start": "09:30",
                    "end": "16:00",
                    "timezone": "US/Eastern"
                },
                "position_limits": {
                    "max_notional": 10000000,  # $10M
                    "max_leverage": 3.0,
                    "max_concentration": 0.2  # 20% per symbol
                }
            },
            "monitoring": {
                "dashboard": {
                    "enabled": True,
                    "port": 8080,
                    "host": "localhost"
                },
                "alerts": {
                    "email_enabled": self._get_env_bool("EMAIL_ALERTS", False),
                    "email_smtp": os.getenv("SMTP_SERVER"),
                    "email_port": int(os.getenv("SMTP_PORT", "587")),
                    "email_username": os.getenv("EMAIL_USERNAME"),
                    "email_password": os.getenv("EMAIL_PASSWORD"),
                    "webhook_url": os.getenv("WEBHOOK_URL")
                },
                "logging": {
                    "file_enabled": True,
                    "file_path": "logs/cerberus.log",
                    "rotation": "daily",
                    "retention": 30  # days
                }
            },
            "security": {
                "api_key_required": self._get_env_bool("API_KEY_REQUIRED", True),
                "api_key": os.getenv("API_KEY"),
                "rate_limiting": {
                    "enabled": True,
                    "requests_per_minute": 100
                },
                "data_encryption": {
                    "enabled": self._get_env_bool("ENCRYPTION_ENABLED", False),
                    "key": os.getenv("ENCRYPTION_KEY")
                }
            }
        }

        # Load custom configuration if provided
        if config_file:
            self._load_config_file(config_file)

        # Validate configuration
        self._validate_config()

    def _get_env_bool(self, key: str, default: bool = False) -> bool:
        """Get boolean value from environment variable"""
        value = os.getenv(key, str(default)).lower()
        return value in ("true", "1", "yes", "on")

    def _load_config_file(self, config_file: str):
        """Load configuration from file"""
        try:
            config_path = Path(config_file)
            if config_path.exists():
                with open(config_path, 'r') as f:
                    file_config = json.load(f)
                    self._deep_update(self._config, file_config)
                logger.info(f"Loaded configuration from {config_file}")
            else:
                logger.warning(f"Configuration file not found: {config_file}")
        except Exception as e:
            logger.error(f"Failed to load configuration file {config_file}: {e}")

    def _deep_update(self, base_dict: Dict, update_dict: Dict):
        """Deep update of nested dictionary"""
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value

    def _validate_config(self):
        """Validate configuration values"""
        # Validate required fields
        required_fields = [
            "system.name",
            "risk.limits.max_portfolio_delta",
            "models.option_pricing.default_model"
        ]

        for field in required_fields:
            if not self._get_nested_value(field):
                raise ValueError(f"Required configuration field missing: {field}")

        # Validate numeric ranges
        if self.get("risk.var.confidence_level", 0) <= 0 or self.get("risk.var.confidence_level", 0) >= 1:
            raise ValueError("VaR confidence level must be between 0 and 1")

        if self.get("models.validation.model_approval_threshold", 0) <= 0:
            raise ValueError("Model approval threshold must be positive")

        logger.info("Configuration validation passed")

    def _get_nested_value(self, key_path: str) -> Any:
        """Get nested value using dot notation"""
        keys = key_path.split('.')
        value = self._config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return None
        return value

    def get(self, key_path: str, default: Any = None) -> Any:
        """Get configuration value using dot notation"""
        value = self._get_nested_value(key_path)
        return value if value is not None else default

    def set(self, key_path: str, value: Any):
        """Set configuration value using dot notation"""
        keys = key_path.split('.')
        config = self._config
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        config[keys[-1]] = value

    def get_agent_configs(self) -> Dict[str, Dict[str, Any]]:
        """Get configuration for all agents"""
        return self.get("agents", {})

    def get_risk_limits(self) -> Dict[str, Any]:
        """Get risk limit configuration"""
        return self.get("risk.limits", {})

    def get_model_config(self) -> Dict[str, Any]:
        """Get model configuration"""
        return self.get("models", {})

    def get_data_config(self) -> Dict[str, Any]:
        """Get data configuration"""
        return self.get("data", {})

    def get_monitoring_config(self) -> Dict[str, Any]:
        """Get monitoring configuration"""
        return self.get("monitoring", {})

    def get_trading_config(self) -> Dict[str, Any]:
        """Get trading configuration"""
        return self.get("trading", {})

    def is_debug_mode(self) -> bool:
        """Check if debug mode is enabled"""
        return self.get("system.debug", False)

    def get_log_level(self) -> str:
        """Get logging level"""
        return self.get("system.log_level", "INFO")

    def to_dict(self) -> Dict[str, Any]:
        """Get full configuration as dictionary"""
        return self._config.copy()

    def save_to_file(self, file_path: str):
        """Save current configuration to file"""
        try:
            config_path = Path(file_path)
            config_path.parent.mkdir(parents=True, exist_ok=True)

            with open(config_path, 'w') as f:
                json.dump(self._config, f, indent=2, default=str)

            logger.info(f"Configuration saved to {file_path}")
        except Exception as e:
            logger.error(f"Failed to save configuration to {file_path}: {e}")

    def get_symbols_to_monitor(self) -> List[str]:
        """Get list of symbols to monitor"""
        primary = self.get("trading.symbols.primary", [])
        benchmark = self.get("trading.symbols.benchmark", [])
        return primary + benchmark

    def get_market_hours(self) -> Dict[str, str]:
        """Get market hours configuration"""
        return self.get("trading.market_hours", {})

    def is_redis_enabled(self) -> bool:
        """Check if Redis caching is enabled"""
        return self.get("data.redis.enabled", False)

    def get_redis_config(self) -> Dict[str, Any]:
        """Get Redis configuration"""
        return self.get("data.redis", {})

    def get_alert_config(self) -> Dict[str, Any]:
        """Get alert configuration"""
        return self.get("monitoring.alerts", {})

    def validate_runtime_config(self) -> bool:
        """Validate runtime configuration for system startup"""
        try:
            # Check if required services are available
            symbols = self.get_symbols_to_monitor()
            if not symbols:
                logger.warning("No symbols configured for monitoring")

            # Check if risk limits are reasonable
            max_delta = self.get("risk.limits.max_portfolio_delta")
            if max_delta <= 0:
                logger.error("Invalid max portfolio delta limit")
                return False

            # Check model configuration
            default_model = self.get("models.option_pricing.default_model")
            available_models = self.get("models.option_pricing.available_models", [])
            if default_model not in available_models:
                logger.error(f"Default model {default_model} not in available models")
                return False

            return True

        except Exception as e:
            logger.error(f"Runtime configuration validation failed: {e}")
            return False