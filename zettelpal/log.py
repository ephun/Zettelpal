# log.py - Shared logging setup for the CLI, GUI, and pipeline modules.
#
# Modules log through get_logger(__name__); the entry point (CLI or GUI)
# decides where output goes by attaching handlers to the "zettelpal" logger.

import logging

LOGGER_NAME = "zettelpal"


def get_logger(name: str | None = None) -> logging.Logger:
    """Return the zettelpal logger, or a child logger for a module name."""
    if not name or name == LOGGER_NAME:
        return logging.getLogger(LOGGER_NAME)
    if name.startswith(LOGGER_NAME + "."):
        return logging.getLogger(name)
    return logging.getLogger(f"{LOGGER_NAME}.{name}")


def setup_console_logging(verbose: bool = False) -> None:
    """Configure plain console output. Safe to call more than once."""
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    if any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        return
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
