"""Compatibility entry point for the AdaptiveML API.

Preferred command: ``python -m src.api.app``.
"""

import os

from src.api.app import app

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.api.app:app",
        host=os.getenv("ADAPTIVEML_HOST", "127.0.0.1"),
        port=int(os.getenv("ADAPTIVEML_PORT", "8765")),
        reload=False,
    )
