import logging
import datetime
from pathlib import Path


class ColorFormatter(logging.Formatter):
    """
    Applies colors to the log level and message based on severity.
    """
    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    
    # We color specific parts of the string using the variables above
    base_format = "%(asctime)s [%(name)s] "
    
    FORMATS = {
        logging.DEBUG: grey + base_format + "[%(levelname)s] %(message)s" + reset,
        logging.INFO: grey + base_format + "[%(levelname)s] %(message)s" + reset,
        logging.WARNING: yellow + base_format + "[%(levelname)s] %(message)s" + reset,
        logging.ERROR: red + base_format + "[%(levelname)s] %(message)s" + reset,
        logging.CRITICAL: bold_red + base_format + "[%(levelname)s] %(message)s" + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


def setup_logging(experiment_name: str, log_dir: Path) -> logging.Logger:
    """
    Sets up a logger that writes to file (clean) and console (colorful).
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_file = log_dir / f"{experiment_name}-{timestamp}.log"

    logger = logging.getLogger("pipeline")
    logger.setLevel(logging.INFO)

    # Safety: Remove existing handlers if this function is called multiple times
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # prevent the logs from propagating to the root logger (which causes double printing)
    logger.propagate = False

    # -- Handler 1: File (Clean Text) --
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(name)s] [%(levelname)s] %(message)s"
    ))
    logger.addHandler(file_handler)

    # -- Handler 2: Console (Colorful) --
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(ColorFormatter())
    logger.addHandler(console_handler)

    logger.info(f"Starting experiment: {experiment_name}")
    return logger

