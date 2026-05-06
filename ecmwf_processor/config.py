import logging
from ecmwf_processor import settings, custom_logging

SETTINGS: settings.Settings = settings.Settings()
LOGGER: logging.Logger = custom_logging.setup_logger()


class InternalInvariantError(Exception):
    """Exception raised when an internal invariant is violated."""
