import logging
import os

from permissions.levels import RiskLevel

logger = logging.getLogger("mantis.permissions")


TOOL_POLICY = {
    "python": RiskLevel.dangerous,
    "filesystem.write": RiskLevel.dangerous,
    "write_file": RiskLevel.dangerous,
    "filesystem.write_file": RiskLevel.dangerous,
    "http": RiskLevel.moderate,
    "search_files": RiskLevel.moderate,
    "read_file": RiskLevel.safe,
    "filesystem.read": RiskLevel.safe,
    "list_files": RiskLevel.safe,
}


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def check_permission(tool_name: str) -> bool:
    risk = TOOL_POLICY.get(tool_name, RiskLevel.safe)
    safe_mode = _env_bool("MANTIS_SAFE_MODE", default=False)

    if safe_mode and risk == RiskLevel.dangerous:
        logger.warning("Blocked dangerous tool in safe mode: %s", tool_name)
        return False

    if risk == RiskLevel.dangerous:
        logger.warning("Dangerous tool requested: %s", tool_name)
        approved = _env_bool("MANTIS_APPROVE_DANGEROUS", default=False)
        if not approved:
            logger.warning("Dangerous tool denied (approval flag disabled): %s", tool_name)
            return False

    return True
