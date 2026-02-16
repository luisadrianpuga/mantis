import os

MANTIS_VERSION = "0.1.0"


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


MANTIS_DEV_MODE = _env_bool("MANTIS_DEV_MODE", True)
MANTIS_ALLOW_FILE_WRITE = _env_bool("MANTIS_ALLOW_FILE_WRITE", False)
MANTIS_SANDBOX = _env_bool("MANTIS_SANDBOX", True)

MANTIS_RETRY_MAX_RETRIES = int(os.getenv("MANTIS_RETRY_MAX_RETRIES", "2"))
MANTIS_RETRY_BASE_DELAY_SEC = float(os.getenv("MANTIS_RETRY_BASE_DELAY_SEC", "0.25"))
