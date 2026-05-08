import logging
import os


logger = logging.getLogger(__name__)


def get_langfuse_client():
    try:
        from langfuse import Langfuse
    except ImportError:
        logger.info("langfuse package not installed; tracing disabled")
        return None
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY", "")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY", "")
    if not public_key or not secret_key:
        logger.info("LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY not set; tracing disabled")
        return None
    host = os.getenv("LANGFUSE_HOST", "http://localhost:3000")
    try:
        client = Langfuse(public_key=public_key, secret_key=secret_key, host=host)
        logger.info("Langfuse tracing enabled -> %s", host)
        return client
    except Exception:
        logger.exception("Failed to initialise Langfuse client; tracing disabled")
        return None
