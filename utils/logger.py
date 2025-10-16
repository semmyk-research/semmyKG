# utils/logger.py

import json
import logging
import sys
from datetime import datetime, timezone, timedelta

class JsonFormatter(logging.Formatter):
    """Minimal JSON formatter for structured logs."""

    def __init__(self, tz_hours=None, date_format:str="%Y-%m-%d : %H:%M:%S"):
        ##SMY: TODO: local time
        self.tz_hours = tz_hours if tz_hours else 0
        self.date_format = date_format
        self.time = datetime.now(timezone.utc) + timedelta(hours=tz_hours if tz_hours else 0) ##SMY: TODO: fix timedelta()

    def format(self, record: logging.LogRecord) -> str:  #
        payload = {
            #"ts": datetime.now(timezone.utc).isoformat(),  ## default to 'YYYY-MM-DD HH:MM:SS.mmmmmm',
            #"ts": datetime.now(timezone.utc).strftime("%Y-%m-%d : %H:%M:%S"),  ## SMY: interested in datefmt="%H:%M:%S",
            "ts": f"{self.time.strftime(self.date_format)} (UTC)",  ## SMY: interested in datefmt="%H:%M:%S",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        # Include extra attributes (fields not in default LogRecord)
        for key, value in record.__dict__.items():
            if key in ("args", "msg", "levelno", "levelname", "name", "pathname", "filename",
                       "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
                       "created", "msecs", "relativeCreated", "thread", "threadName",
                       "processName", "process"):
                continue
            payload[key] = value
        return json.dumps(payload, ensure_ascii=False)

#def setup_logging(level: int = logging.INFO) -> None:  ## Causing non-stop logging on HF spaces
def setup_logging(level: int = None, filename="app_logging.log", tz_hours=0.0, date_format:str="%d%b%Y") -> None:  #'%Y-%m-%d
    """Configure root logger with JSON output to both stdout and file.
    
    Args:
        level: Logging level. If None, uses WARNING for production (HF Spaces) 
               and INFO for local development.
        filename: log file name
        tz_hours: timezone offset in hours
        date_format: date format for log entries
    """
    if level is None:
        # Auto-detect environment: WARNING for production, INFO for local dev
        import os
        is_production = os.getenv("SPACE_ID") or os.getenv("HF_SPACE_ID") or os.getenv("HUGGINGFACE_SPACE_ID")
        level = logging.WARNING if is_production else logging.INFO
    
    # Console handler
    console_handler = logging.StreamHandler(stream=sys.stdout)
    console_handler.setFormatter(JsonFormatter())  #, datefmt="%H:%M:%S",)  ##explicit time format
    
    # File handler
    from utils.file_utils import check_create_logfile
    file_handler = logging.FileHandler(check_create_logfile(filename=filename, tz_hours=tz_hours, date_format=date_format), mode="a", encoding="utf-8")
    
    file_handler.setFormatter(JsonFormatter())
    
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(console_handler)
    root.addHandler(file_handler)
    root.setLevel(level)


def get_logger(name: str) -> logging.Logger:
    """Return a module logger configured with console handler using defined JSON format."""
    return logging.getLogger(name)


