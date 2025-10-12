from datetime import datetime
from zoneinfo import ZoneInfo
import uuid

class ResponseBuilder:
    """Builds structured JSON responses for API endpoints."""

    @staticmethod
    def success(data: dict) -> dict:
        """
        Build a success response with standard fields.
        """
        return {
            "success": True,
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
            "success": False,
            "error": message,
            "code": code,
            "metadata": {
                "request_id": str(uuid.uuid4()),
                # Chile timezone (America/Santiago)
                "timestamp": datetime.now(ZoneInfo("America/Santiago")).isoformat()
            }
        }
