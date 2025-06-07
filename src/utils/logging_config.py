"""
Comprehensive logging configuration for the Digital Twin Framework
"""
import logging
import logging.config
import sys
from pathlib import Path
from typing import Dict, Any
import json
from datetime import datetime

def setup_logging(
    log_level: str = "INFO",
    log_file: str = "digital_twin.log",
    log_dir: str = "logs",
    structured_logging: bool = True
) -> None:
    """
    Setup comprehensive logging for the digital twin framework
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Log file name
        log_dir: Directory for log files
        structured_logging: Whether to use structured JSON logging
    """
    
    # Create logs directory if it doesn't exist
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True)
    
    # Define log format
    if structured_logging:
        formatter_class = StructuredFormatter
        log_format = None
    else:
        formatter_class = logging.Formatter
        log_format = (
            "%(asctime)s - %(name)s - %(levelname)s - "
            "[%(filename)s:%(lineno)d] - %(message)s"
        )
    
    # Logging configuration
    logging_config: Dict[str, Any] = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "()": formatter_class,
                "format": log_format,
                "datefmt": "%Y-%m-%d %H:%M:%S"
            },
            "detailed": {
                "()": formatter_class,
                "format": log_format,
                "datefmt": "%Y-%m-%d %H:%M:%S"
            }
        },
        "handlers": {
            "console": {
                "level": log_level,
                "class": "logging.StreamHandler",
                "formatter": "standard",
                "stream": sys.stdout
            },
            "file": {
                "level": "DEBUG",
                "class": "logging.handlers.RotatingFileHandler",
                "formatter": "detailed",
                "filename": str(log_path / log_file),
                "maxBytes": 10485760,  # 10MB
                "backupCount": 5,
                "encoding": "utf8"
            },
            "error_file": {
                "level": "ERROR",
                "class": "logging.handlers.RotatingFileHandler",
                "formatter": "detailed",
                "filename": str(log_path / "errors.log"),
                "maxBytes": 10485760,  # 10MB
                "backupCount": 3,
                "encoding": "utf8"
            }
        },
        "loggers": {
            "": {  # Root logger
                "level": log_level,
                "handlers": ["console", "file", "error_file"],
                "propagate": False
            },
            "uvicorn": {
                "level": log_level,
                "handlers": ["console", "file"],
                "propagate": False
            },
            "fastapi": {
                "level": log_level,
                "handlers": ["console", "file"],
                "propagate": False
            },
            "sqlalchemy": {
                "level": "WARNING",
                "handlers": ["file"],
                "propagate": False
            },
            "digital_twin.macro_twin": {
                "level": "DEBUG",
                "handlers": ["console", "file"],
                "propagate": False
            },
            "digital_twin.forecasting": {
                "level": "DEBUG",
                "handlers": ["console", "file"],
                "propagate": False
            },
            "digital_twin.optimization": {
                "level": "DEBUG",
                "handlers": ["console", "file"],
                "propagate": False
            },
            "digital_twin.simulation": {
                "level": "DEBUG",
                "handlers": ["console", "file"],
                "propagate": False
            }
        }
    }
    
    logging.config.dictConfig(logging_config)


class StructuredFormatter(logging.Formatter):
    """
    Custom formatter for structured JSON logging
    """
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as structured JSON"""
        
        # Base log structure
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "process_id": record.process,
            "thread_id": record.thread
        }
        
        # Add exception information if present
        if record.exc_info:
            log_entry["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": self.formatException(record.exc_info)
            }
        
        # Add extra fields if present
        extra_fields = {}
        for key, value in record.__dict__.items():
            if key not in {
                'name', 'msg', 'args', 'levelname', 'levelno', 'pathname',
                'filename', 'module', 'exc_info', 'exc_text', 'stack_info',
                'lineno', 'funcName', 'created', 'msecs', 'relativeCreated',
                'thread', 'threadName', 'processName', 'process', 'getMessage',
                'message'
            }:
                extra_fields[key] = value
        
        if extra_fields:
            log_entry["extra"] = extra_fields
        
        return json.dumps(log_entry, default=str)


class PerformanceLogger:
    """
    Performance logging utility for tracking execution times
    """
    
    def __init__(self, logger_name: str = "digital_twin.performance"):
        self.logger = logging.getLogger(logger_name)
    
    def log_execution_time(self, operation: str, execution_time_ms: float, **kwargs):
        """Log execution time for operations"""
        self.logger.info(
            f"Operation completed: {operation}",
            extra={
                "operation": operation,
                "execution_time_ms": execution_time_ms,
                "performance_metric": True,
                **kwargs
            }
        )
    
    def log_system_metrics(self, service_name: str, metrics: Dict[str, Any]):
        """Log system performance metrics"""
        self.logger.info(
            f"System metrics for {service_name}",
            extra={
                "service_name": service_name,
                "metrics": metrics,
                "system_metric": True
            }
        )


class AuditLogger:
    """
    Audit logging for tracking user actions and system events
    """
    
    def __init__(self, logger_name: str = "digital_twin.audit"):
        self.logger = logging.getLogger(logger_name)
    
    def log_api_request(self, 
                       endpoint: str, 
                       method: str, 
                       user_id: str = None,
                       request_data: Dict[str, Any] = None,
                       response_status: int = None,
                       execution_time_ms: float = None):
        """Log API requests for audit trail"""
        self.logger.info(
            f"API Request: {method} {endpoint}",
            extra={
                "audit_type": "api_request",
                "endpoint": endpoint,
                "method": method,
                "user_id": user_id,
                "request_data": request_data,
                "response_status": response_status,
                "execution_time_ms": execution_time_ms
            }
        )
    
    def log_optimization_run(self, 
                           community_id: str,
                           objective: str,
                           execution_time_ms: float,
                           optimal_cost: float,
                           user_id: str = None):
        """Log optimization runs for audit trail"""
        self.logger.info(
            f"Optimization completed for community {community_id}",
            extra={
                "audit_type": "optimization_run",
                "community_id": community_id,
                "objective": objective,
                "execution_time_ms": execution_time_ms,
                "optimal_cost": optimal_cost,
                "user_id": user_id
            }
        )
    
    def log_forecast_generation(self,
                              community_id: str,
                              model_type: str,
                              forecast_horizon_hours: int,
                              accuracy_metrics: Dict[str, float] = None):
        """Log forecast generation for audit trail"""
        self.logger.info(
            f"Forecast generated for community {community_id}",
            extra={
                "audit_type": "forecast_generation",
                "community_id": community_id,
                "model_type": model_type,
                "forecast_horizon_hours": forecast_horizon_hours,
                "accuracy_metrics": accuracy_metrics
            }
        )


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with the given name"""
    return logging.getLogger(name)


def log_exception(logger: logging.Logger, message: str = "Exception occurred"):
    """Decorator to log exceptions"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.exception(f"{message} in {func.__name__}: {str(e)}")
                raise
        return wrapper
    return decorator


# Initialize logging when module is imported
if __name__ != "__main__":
    setup_logging()
