"""
Data & Infrastructure Agent
Responsible for data management, system performance, and infrastructure monitoring
"""

import asyncio
import logging
import psutil
import json
import time
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import aiofiles
import redis
from pathlib import Path

from agents.base_agent import BaseAgent, AgentMessage, AgentStatus
from data.yahoo_connector import YahooFinanceConnector

logger = logging.getLogger(__name__)

@dataclass
class SystemMetrics:
    timestamp: datetime
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    network_io: Dict[str, int]
    active_connections: int

@dataclass
class DataQualityMetric:
    source: str
    metric_type: str  # 'latency', 'completeness', 'accuracy'
    value: float
    threshold: float
    status: str  # 'good', 'warning', 'critical'
    timestamp: datetime

class DataInfrastructureAgent(BaseAgent):
    """
    Data & Infrastructure Agent - System performance and data quality management
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__("data_infrastructure", config)
        self.data_connector = None
        self.redis_client = None
        self.system_metrics_history = []
        self.data_quality_metrics = {}
        self.cache_stats = {}
        self.performance_metrics = {
            "data_requests_processed": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "data_quality_checks": 0,
            "system_health_checks": 0
        }

    async def initialize(self):
        """Initialize the Data & Infrastructure Agent"""
        self.logger.info("Initializing Data & Infrastructure Agent...")

        # Initialize data connector
        self.data_connector = YahooFinanceConnector()

        # Initialize Redis cache (optional, will handle gracefully if not available)
        try:
            self.redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
            self.redis_client.ping()
            self.logger.info("Redis cache connected successfully")
        except Exception as e:
            self.logger.info("Redis cache not available (optional) - continuing without caching")
            self.redis_client = None

        # Start monitoring loops
        asyncio.create_task(self._system_monitoring_loop())
        asyncio.create_task(self._data_quality_monitoring_loop())
        asyncio.create_task(self._cache_cleanup_loop())

        self.logger.info("Data & Infrastructure Agent initialized successfully")

    async def process_message(self, message: AgentMessage):
        """Process incoming messages"""

        if message.message_type == "get_market_data":
            await self._handle_market_data_request(message)
        elif message.message_type == "cache_data":
            await self._handle_cache_request(message)
        elif message.message_type == "system_health":
            await self._handle_system_health_request(message)
        elif message.message_type == "data_quality_report":
            await self._handle_data_quality_request(message)
        elif message.message_type == "performance_optimization":
            await self._handle_performance_optimization(message)
        elif message.message_type == "backup_data":
            await self._handle_backup_request(message)
        elif message.message_type == "health_check":
            # Simple health check response
            self.logger.debug("Health check received - system healthy")
        else:
            self.logger.warning(f"Unknown message type: {message.message_type}")

    async def _handle_cache_request(self, message: AgentMessage):
        """Handle cache requests"""
        self.logger.info("Cache request received")

    async def _handle_performance_optimization(self, message: AgentMessage):
        """Handle performance optimization requests"""
        self.logger.info("Performance optimization request received")

    async def _handle_market_data_request(self, message: AgentMessage):
        """Handle market data requests with caching"""
        request = message.payload
        symbol = request.get("symbol")
        data_type = request.get("type", "quote")  # quote, historical, options

        try:
            # Check cache first
            cache_key = f"market_data:{symbol}:{data_type}:{datetime.now().strftime('%Y-%m-%d-%H-%M')}"
            cached_data = await self._get_from_cache(cache_key)

            if cached_data:
                self.performance_metrics["cache_hits"] += 1
                await self.send_message(
                    target=message.source,
                    message_type="market_data_response",
                    payload={
                        "request_id": request.get("request_id"),
                        "data": cached_data,
                        "source": "cache",
                        "timestamp": datetime.now().isoformat()
                    }
                )
            else:
                self.performance_metrics["cache_misses"] += 1
                # Fetch fresh data
                data = await self._fetch_market_data(symbol, data_type)

                if data:
                    # Cache the data
                    await self._store_in_cache(cache_key, data, ttl=60)  # 1 minute TTL

                    await self.send_message(
                        target=message.source,
                        message_type="market_data_response",
                        payload={
                            "request_id": request.get("request_id"),
                            "data": data,
                            "source": "live",
                            "timestamp": datetime.now().isoformat()
                        }
                    )
                else:
                    raise ValueError(f"No data available for {symbol}")

            self.performance_metrics["data_requests_processed"] += 1

        except Exception as e:
            self.logger.error(f"Market data request error: {e}")
            await self.send_message(
                target=message.source,
                message_type="market_data_error",
                payload={
                    "request_id": request.get("request_id"),
                    "error": str(e)
                }
            )

    async def _fetch_market_data(self, symbol: str, data_type: str) -> Optional[Dict[str, Any]]:
        """Fetch market data from external sources"""
        try:
            if data_type == "quote":
                market_data = self.data_connector.get_real_time_quote(symbol)
                return asdict(market_data) if market_data else None

            elif data_type == "historical":
                historical_data = self.data_connector.get_stock_data(symbol, period="1d", interval="1m")
                if not historical_data.empty:
                    return historical_data.to_dict('records')
                return None

            elif data_type == "options":
                options_data = self.data_connector.get_options_chain(symbol)
                return [asdict(option) for option in options_data]

            elif data_type == "volatility":
                volatility = self.data_connector.get_volatility_data(symbol, 30)
                return {"symbol": symbol, "volatility": volatility, "period": 30}

            else:
                self.logger.warning(f"Unknown data type: {data_type}")
                return None

        except Exception as e:
            self.logger.error(f"Error fetching market data: {e}")
            return None

    async def _get_from_cache(self, key: str) -> Optional[Dict[str, Any]]:
        """Get data from cache"""
        if not self.redis_client:
            return None

        try:
            cached_data = self.redis_client.get(key)
            if cached_data:
                return json.loads(cached_data)
            return None

        except Exception as e:
            self.logger.error(f"Cache read error: {e}")
            return None

    async def _store_in_cache(self, key: str, data: Any, ttl: int = 300):
        """Store data in cache"""
        if not self.redis_client:
            return

        try:
            serialized_data = json.dumps(data, default=str)
            self.redis_client.setex(key, ttl, serialized_data)

        except Exception as e:
            self.logger.error(f"Cache write error: {e}")

    async def _handle_system_health_request(self, message: AgentMessage):
        """Handle system health check requests"""
        try:
            health_data = await self._collect_system_metrics()

            await self.send_message(
                target=message.source,
                message_type="system_health_response",
                payload={
                    "request_id": message.payload.get("request_id"),
                    "metrics": asdict(health_data),
                    "status": await self._assess_system_health(health_data),
                    "timestamp": datetime.now().isoformat()
                }
            )

            self.performance_metrics["system_health_checks"] += 1

        except Exception as e:
            self.logger.error(f"System health check error: {e}")

    async def _collect_system_metrics(self) -> SystemMetrics:
        """Collect current system metrics"""
        # CPU usage
        cpu_usage = psutil.cpu_percent(interval=1)

        # Memory usage
        memory = psutil.virtual_memory()
        memory_usage = memory.percent

        # Disk usage
        disk = psutil.disk_usage('/')
        disk_usage = disk.percent

        # Network I/O
        network_io = psutil.net_io_counters()
        network_stats = {
            "bytes_sent": network_io.bytes_sent,
            "bytes_recv": network_io.bytes_recv,
            "packets_sent": network_io.packets_sent,
            "packets_recv": network_io.packets_recv
        }

        # Active connections (approximate)
        connections = len(psutil.net_connections())

        metrics = SystemMetrics(
            timestamp=datetime.now(),
            cpu_usage=cpu_usage,
            memory_usage=memory_usage,
            disk_usage=disk_usage,
            network_io=network_stats,
            active_connections=connections
        )

        # Store in history
        self.system_metrics_history.append(metrics)
        if len(self.system_metrics_history) > 1000:
            self.system_metrics_history = self.system_metrics_history[-1000:]

        return metrics

    async def _assess_system_health(self, metrics: SystemMetrics) -> str:
        """Assess overall system health"""
        score = 0

        # CPU assessment
        if metrics.cpu_usage > 80:
            score += 3
        elif metrics.cpu_usage > 60:
            score += 1

        # Memory assessment
        if metrics.memory_usage > 85:
            score += 3
        elif metrics.memory_usage > 70:
            score += 1

        # Disk assessment
        if metrics.disk_usage > 90:
            score += 2
        elif metrics.disk_usage > 80:
            score += 1

        if score == 0:
            return "excellent"
        elif score <= 2:
            return "good"
        elif score <= 4:
            return "warning"
        else:
            return "critical"

    async def _handle_data_quality_request(self, message: AgentMessage):
        """Handle data quality assessment requests"""
        try:
            quality_report = await self._assess_data_quality()

            await self.send_message(
                target=message.source,
                message_type="data_quality_response",
                payload={
                    "request_id": message.payload.get("request_id"),
                    "quality_metrics": quality_report,
                    "timestamp": datetime.now().isoformat()
                }
            )

            self.performance_metrics["data_quality_checks"] += 1

        except Exception as e:
            self.logger.error(f"Data quality assessment error: {e}")

    async def _assess_data_quality(self) -> Dict[str, Any]:
        """Assess data quality across sources"""
        quality_metrics = {}

        # Test major symbols for data availability and latency
        test_symbols = ["MSTR", "QQQ", "SPY", "AAPL"]

        for symbol in test_symbols:
            start_time = time.time()

            # Test data availability
            market_data = self.data_connector.get_real_time_quote(symbol)
            latency = (time.time() - start_time) * 1000  # milliseconds

            if market_data:
                # Check data freshness (assuming market hours)
                data_age = (datetime.now() - market_data.timestamp).total_seconds()

                quality_metrics[symbol] = {
                    "availability": "good",
                    "latency_ms": latency,
                    "data_age_seconds": data_age,
                    "price_valid": market_data.price > 0,
                    "volume_valid": market_data.volume >= 0
                }
            else:
                quality_metrics[symbol] = {
                    "availability": "poor",
                    "latency_ms": latency,
                    "data_age_seconds": None,
                    "price_valid": False,
                    "volume_valid": False
                }

        # Overall assessment
        available_count = sum(1 for metrics in quality_metrics.values() if metrics["availability"] == "good")
        avg_latency = sum(metrics["latency_ms"] for metrics in quality_metrics.values()) / len(quality_metrics)

        overall_assessment = {
            "symbols_tested": len(test_symbols),
            "symbols_available": available_count,
            "availability_rate": available_count / len(test_symbols),
            "average_latency_ms": avg_latency,
            "overall_status": "good" if available_count >= len(test_symbols) * 0.8 else "degraded"
        }

        return {
            "symbol_metrics": quality_metrics,
            "overall": overall_assessment
        }

    async def _system_monitoring_loop(self):
        """Continuous system monitoring loop"""
        while self.running:
            try:
                metrics = await self._collect_system_metrics()
                health_status = await self._assess_system_health(metrics)

                # Send alerts if system health is poor
                if health_status in ["warning", "critical"]:
                    await self.send_message(
                        target="chief_architect",
                        message_type="system_health_alert",
                        payload={
                            "metrics": asdict(metrics),
                            "status": health_status,
                            "timestamp": datetime.now().isoformat()
                        },
                        priority=1 if health_status == "critical" else 2
                    )

                await asyncio.sleep(30)  # Check every 30 seconds

            except Exception as e:
                self.logger.error(f"System monitoring error: {e}")
                await asyncio.sleep(60)

    async def _data_quality_monitoring_loop(self):
        """Continuous data quality monitoring loop"""
        while self.running:
            try:
                quality_report = await self._assess_data_quality()

                # Check for data quality issues
                if quality_report["overall"]["overall_status"] == "degraded":
                    await self.send_message(
                        target="chief_architect",
                        message_type="data_quality_issue",
                        payload={
                            "source": "market_data",
                            "type": "availability_degraded",
                            "report": quality_report,
                            "timestamp": datetime.now().isoformat()
                        },
                        priority=2
                    )

                await asyncio.sleep(600)  # Check every 10 minutes

            except Exception as e:
                self.logger.error(f"Data quality monitoring error: {e}")
                await asyncio.sleep(300)

    async def _cache_cleanup_loop(self):
        """Cache cleanup and statistics collection loop"""
        while self.running:
            try:
                if self.redis_client:
                    # Collect cache statistics
                    info = self.redis_client.info()
                    self.cache_stats = {
                        "used_memory": info.get("used_memory", 0),
                        "connected_clients": info.get("connected_clients", 0),
                        "total_commands_processed": info.get("total_commands_processed", 0),
                        "keyspace_hits": info.get("keyspace_hits", 0),
                        "keyspace_misses": info.get("keyspace_misses", 0)
                    }

                    # Calculate hit rate
                    hits = self.cache_stats["keyspace_hits"]
                    misses = self.cache_stats["keyspace_misses"]
                    if hits + misses > 0:
                        hit_rate = hits / (hits + misses)
                        self.cache_stats["hit_rate"] = hit_rate

                await asyncio.sleep(300)  # Every 5 minutes

            except Exception as e:
                self.logger.error(f"Cache cleanup error: {e}")
                await asyncio.sleep(600)

    async def _handle_backup_request(self, message: AgentMessage):
        """Handle data backup requests"""
        request = message.payload
        backup_type = request.get("type", "system_state")

        try:
            backup_path = await self._create_backup(backup_type)

            await self.send_message(
                target=message.source,
                message_type="backup_response",
                payload={
                    "request_id": request.get("request_id"),
                    "backup_path": str(backup_path),
                    "backup_type": backup_type,
                    "timestamp": datetime.now().isoformat()
                }
            )

        except Exception as e:
            self.logger.error(f"Backup creation error: {e}")

    async def _create_backup(self, backup_type: str) -> Path:
        """Create system backup"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = Path("backups")
        backup_dir.mkdir(exist_ok=True)

        if backup_type == "system_state":
            backup_file = backup_dir / f"system_state_{timestamp}.json"

            backup_data = {
                "timestamp": datetime.now().isoformat(),
                "system_metrics": [asdict(m) for m in self.system_metrics_history[-100:]],
                "performance_metrics": self.performance_metrics,
                "cache_stats": self.cache_stats,
                "data_quality_metrics": self.data_quality_metrics
            }

            async with aiofiles.open(backup_file, 'w') as f:
                await f.write(json.dumps(backup_data, indent=2, default=str))

            return backup_file

        else:
            raise ValueError(f"Unknown backup type: {backup_type}")

    async def cleanup(self):
        """Cleanup resources"""
        self.logger.info("Cleaning up Data & Infrastructure Agent...")
        if self.redis_client:
            self.redis_client.close()

    async def get_status(self) -> AgentStatus:
        """Get current status"""
        self.status.performance_metrics = self.performance_metrics
        return self.status

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check"""
        latest_metrics = self.system_metrics_history[-1] if self.system_metrics_history else None

        return {
            "agent_id": self.agent_id,
            "status": self.status.status,
            "last_activity": self.status.last_activity.isoformat(),
            "performance_metrics": self.performance_metrics,
            "cache_available": self.redis_client is not None,
            "cache_stats": self.cache_stats,
            "latest_system_metrics": asdict(latest_metrics) if latest_metrics else None,
            "healthy": True
        }