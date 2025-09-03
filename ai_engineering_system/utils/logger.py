"""
Logging configuration for the AI engineering system.
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from loguru import logger
import os


def setup_logger(
    name: str = "ai_engineering_system",
    level: str = "INFO",
    log_file: Optional[str] = None,
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    Set up logging for the AI engineering system.
    
    Args:
        name: Logger name
        level: Logging level
        log_file: Optional log file path
        format_string: Optional custom format string
        
    Returns:
        Configured logger
    """
    # Remove default loguru handler
    logger.remove()
    
    # Default format
    if format_string is None:
        format_string = (
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        )
    
    # Add console handler
    logger.add(
        sys.stdout,
        format=format_string,
        level=level,
        colorize=True
    )
    
    # Add file handler if specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.add(
            log_file,
            format=format_string,
            level=level,
            rotation="10 MB",
            retention="30 days",
            compression="zip"
        )
    
    # Configure standard logging to use loguru
    class InterceptHandler(logging.Handler):
        def emit(self, record):
            try:
                level = logger.level(record.levelname).name
            except ValueError:
                level = record.levelno
            
            frame, depth = sys._getframe(6), 6
            while frame and frame.f_code.co_filename == logging.__file__:
                frame = frame.f_back
                depth += 1
            
            logger.opt(depth=depth, exception=record.exc_info).log(
                level, record.getMessage()
            )
    
    # Set up standard library logging
    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)
    
    # Get the specific logger
    std_logger = logging.getLogger(name)
    std_logger.setLevel(getattr(logging, level.upper()))
    
    return std_logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance."""
    return logging.getLogger(name)


class LoggerMixin:
    """Mixin class to add logging capabilities to any class."""
    
    @property
    def logger(self) -> logging.Logger:
        """Get logger for this class."""
        if not hasattr(self, '_logger'):
            self._logger = get_logger(self.__class__.__name__)
        return self._logger


def log_function_call(func):
    """Decorator to log function calls."""
    def wrapper(*args, **kwargs):
        logger_name = func.__module__ + "." + func.__name__
        func_logger = get_logger(logger_name)
        
        func_logger.debug(f"Calling {func.__name__} with args={args}, kwargs={kwargs}")
        
        try:
            result = func(*args, **kwargs)
            func_logger.debug(f"{func.__name__} completed successfully")
            return result
        except Exception as e:
            func_logger.error(f"{func.__name__} failed with error: {e}")
            raise
    
    return wrapper


def log_performance(func):
    """Decorator to log function performance."""
    import time
    
    def wrapper(*args, **kwargs):
        logger_name = func.__module__ + "." + func.__name__
        func_logger = get_logger(logger_name)
        
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        execution_time = end_time - start_time
        func_logger.info(f"{func.__name__} executed in {execution_time:.4f} seconds")
        
        return result
    
    return wrapper


class PerformanceLogger:
    """Context manager for logging performance of code blocks."""
    
    def __init__(self, operation_name: str, logger: Optional[logging.Logger] = None):
        self.operation_name = operation_name
        self.logger = logger or get_logger("performance")
        self.start_time = None
    
    def __enter__(self):
        import time
        self.start_time = time.time()
        self.logger.info(f"Starting {self.operation_name}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        import time
        if self.start_time:
            execution_time = time.time() - self.start_time
            if exc_type is None:
                self.logger.info(f"{self.operation_name} completed in {execution_time:.4f} seconds")
            else:
                self.logger.error(f"{self.operation_name} failed after {execution_time:.4f} seconds: {exc_val}")


def setup_engineering_logger(
    log_dir: str = "./logs",
    level: str = "INFO"
) -> logging.Logger:
    """
    Set up specialized logging for engineering applications.
    
    Args:
        log_dir: Directory for log files
        level: Logging level
        
    Returns:
        Configured logger
    """
    # Create log directory
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    # Set up main logger
    main_logger = setup_logger(
        name="engineering_ai",
        level=level,
        log_file=os.path.join(log_dir, "engineering_ai.log")
    )
    
    # Set up specialized loggers for different modules
    module_loggers = {
        "ml": setup_logger("ml_module", level, os.path.join(log_dir, "ml.log")),
        "nlp": setup_logger("nlp_module", level, os.path.join(log_dir, "nlp.log")),
        "vision": setup_logger("vision_module", level, os.path.join(log_dir, "vision.log")),
        "rl": setup_logger("rl_module", level, os.path.join(log_dir, "rl.log")),
        "neural": setup_logger("neural_module", level, os.path.join(log_dir, "neural.log"))
    }
    
    return main_logger
