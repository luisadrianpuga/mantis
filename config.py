import os

MANTIS_VERSION = "0.1.0"


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


MANTIS_DEV_MODE = _env_bool("MANTIS_DEV_MODE", True)
MANTIS_ALLOW_FILE_WRITE = _env_bool("MANTIS_ALLOW_FILE_WRITE", False)
