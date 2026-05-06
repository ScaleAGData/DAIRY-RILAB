from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    HERBIE_MAX_RETRIES: int = 1
    DATA_PATH: Path = Path("data")
    DATASET_ID_DATETIME_FORMAT: str = "%Y_%m_%dT%H_%M_%SZ"
    DATETIME_FORMAT_ISO_8601: str = "%Y-%m-%dT%H:%M:%SZ"
    DATETIME_FORMAT_ISO_8601_long: str = "%Y-%m-%dT%H:%M:%S.%fZ"
    DATETIME_FORMAT_UTC: str = "%Y-%m-%dT%H:%M:%S"
    LOG_LEVEL: str = "INFO"
    PRETTY_PRINT_LOGS: bool = False
    SHOULD_PRODUCE_GRAPHS: bool = True
    DISABLE_TQDM: bool = False
    XCUBE_SUPPORTED_ZARR_FORMAT: int = 2
