from datetime import datetime
from zoneinfo import ZoneInfo
import uuid

class ResponseBuilder:
    """Builds structured JSON responses for API endpoints."""

    @staticmethod
    def success(data: dict, model_version: str = None, warnings: list = None) -> dict:
        """
        Build a success response with standard fields.
        """
        return {
            "model_version": model_version or "unknown",
            "data": data,
            "metadata": {
                "request_id": str(uuid.uuid4()),
                # Chile timezone (America/Santiago)
                "timestamp": datetime.now(ZoneInfo("America/Santiago")).isoformat()
            }
        }

    @staticmethod
    def error(message: str, code: int = 400) -> dict:
        """
        Build a structured error response.
        """
        return {
            "error": message,
            "code": code,
            "metadata": {
                "request_id": str(uuid.uuid4()),
                # Chile timezone (America/Santiago)
                "timestamp": datetime.now(ZoneInfo("America/Santiago")).isoformat()
            }
        }
