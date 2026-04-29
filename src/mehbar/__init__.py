import os

__version__ = 0.1

MEHBAR_LOG_LEVEL = os.getenv("MEHBAR_LOG_LEVEL", "DEBUG").upper()
MEHBAR_EXC_INFO = os.getenv("MEHBAR_LOG_EXCEPTIONS") is not None
