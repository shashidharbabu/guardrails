def create_app():
    from app.backend.main import app
    return app


__all__ = ["create_app"]
