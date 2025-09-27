import logging
import sys
from typing import Optional
from datetime import datetime


def setup_logger(
    name: str = "genuity",
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    format_string: Optional[str] = None,
) -> logging.Logger:
    """
    Set up a logger for the Genuity library.

    Args:
        name: Logger name
        level: Logging level
        log_file: Optional file path for logging
        format_string: Custom format string

    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)

    # Avoid duplicate handlers
    if logger.handlers:
        return logger

    logger.setLevel(level)

    # Default format
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    formatter = logging.Formatter(format_string)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str = "genuity") -> logging.Logger:
    """Get the Genuity logger."""
    return logging.getLogger(name)


class GenuityLogger:
    """Enhanced logger with progress tracking and performance monitoring."""

    def __init__(self, name: str = "genuity"):
        self.logger = get_logger(name)
        self.start_time = None
        self.operation_name = None

    def start_operation(self, operation_name: str):
        """Start timing an operation."""
        self.operation_name = operation_name
        self.start_time = datetime.now()
        self.logger.info(f"🚀 Starting: {operation_name}")

    def end_operation(self, success: bool = True):
        """End timing an operation."""
        if self.start_time and self.operation_name:
            duration = datetime.now() - self.start_time
            status = "✅" if success else "❌"
            self.logger.info(
                f"{status} Completed: {self.operation_name} (Duration: {duration})"
            )
            self.start_time = None
            self.operation_name = None

    def log_performance(self, metric_name: str, value: float, unit: str = ""):
        """Log performance metrics."""
        self.logger.info(f"📊 {metric_name}: {value:.4f}{unit}")

    def log_warning(self, message: str):
        """Log a warning."""
        self.logger.warning(f"⚠️ {message}")

    def log_error(self, message: str, exception: Optional[Exception] = None):
        """Log an error."""
        if exception:
            self.logger.error(f"❌ {message}: {str(exception)}")
        else:
            self.logger.error(f"❌ {message}")

    def log_success(self, message: str):
        """Log a success message."""
        self.logger.info(f"✅ {message}")
