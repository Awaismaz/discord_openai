# services/logger.py
import logging, sys

logger = logging.getLogger("npfbot")
handler = logging.StreamHandler(sys.stdout)
fmt = logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s")
handler.setFormatter(fmt)

if not logger.handlers:
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

def info(msg, *args, **kwargs):
    logger.info(msg, *args, **kwargs)

def error(msg, *args, **kwargs):
    logger.error(msg, *args, **kwargs)

def exception(msg, *args, **kwargs):
    logger.exception(msg, *args, **kwargs)
