"""Web front end for the checker: a FastAPI app and a single static page."""

from phishguard.web.app import create_app

__all__ = ["create_app"]
