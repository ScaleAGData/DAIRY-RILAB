import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pythonjsonlogger import jsonlogger

POTENTIAL_CALLERS = {
    "orchestration-api": "sdpp.api:app",
    "celery-worker": "worker",
    "celery-beat": "beat",
    "pytest": "pytest",
    "other": "",
}
CALLER, *_ = [
    caller
    for caller, cmd_subpart in POTENTIAL_CALLERS.items()
    if cmd_subpart in " ".join(sys.argv)
]


class ISO8601JsonFormatter(jsonlogger.JsonFormatter):
    def add_fields(
        self,
        log_record: dict[str, Any],
        record: logging.LogRecord,
        message_dict: dict[str, Any],
    ):
        super().add_fields(log_record, record, message_dict)

        # Always set ISO time
        log_record["time"] = datetime.fromtimestamp(
            record.created, tz=timezone.utc
        ).strftime("%Y-%m-%dT%H:%M:%S+00:00")
        if not log_record.get("event_type"):
            log_record["event_type"] = "WIP"
        if not log_record.get("action"):
            log_record["action"] = "WIP"

        # Remove uvicorn color noise
        log_record.pop("color_message", None)


def setup_logger(
    name: str = CALLER,
    add_file_handler: bool = False,
    propagate_to_root: bool = False,
    log_level: int = None,
) -> logging.Logger:
    """
    Insert documentation
    """
    if log_level is None:
        from ecmwf_processor.config import SETTINGS

        log_level = {
            "CRITICAL": logging.CRITICAL,
            "ERROR": logging.ERROR,
            "WARNING": logging.WARNING,
            "INFO": logging.INFO,
            "DEBUG": logging.DEBUG,
        }.get(SETTINGS.LOG_LEVEL, logging.INFO)

    logger = logging.getLogger(name)
    console_formatter = logging.Formatter(
        "%(asctime)s - %(module)s:%(lineno)d - %(levelname)s: %(message)s"
    )
    json_log_fmt: str = (
        "%(time)s %(levelname)s %(name)s %(module)s %(funcName)s %(lineno)d %(event_type)s %(action)s %(message)s"
    )
    if SETTINGS.PRETTY_PRINT_LOGS:
        json_formatter = ISO8601JsonFormatter(
            json_log_fmt,
            json_indent=2,
        )
    else:
        json_formatter = ISO8601JsonFormatter(
            json_log_fmt,
        )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(json_formatter)
    logger.addHandler(console_handler)

    if add_file_handler:
        file_handler_formatter = console_formatter
        log_file_dir = Path(f"logs/{name}")
        log_file_dir.mkdir(exist_ok=True, parents=True)
        # log_file = log_file_dir / f"{datetime.now().strftime('%Y-%m-%d')}.log"
        log_file = log_file_dir / "current.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(file_handler_formatter)
        logger.addHandler(file_handler)

    logger.setLevel(log_level)
    logger.propagate = propagate_to_root

    return logger
