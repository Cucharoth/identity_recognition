import io
import pytest

from app.services.validator_service import ValidatorService
from app.exceptions.validation_exception import ValidationError


class _FakeUpload:
    """Helper to construct an object similar to FastAPI's UploadFile for tests."""
    def __init__(self, content: bytes, filename: str = "test.jpg", content_type: str = "image/jpeg"):
        self.file = io.BytesIO(content)
        self.filename = filename
        self.content_type = content_type


def test_validator_allows_valid_image():
    svc = ValidatorService()
    fake = _FakeUpload(b"\xFF\xD8\xFF\xE0" + b"0" * 1024, filename="ok.jpg", content_type="image/jpeg")
    assert svc.validate(fake) is True


def test_validator_rejects_invalid_content_type():
    svc = ValidatorService()
    fake = _FakeUpload(b"data", filename="bad.txt", content_type="text/plain")
    with pytest.raises(ValidationError) as exc:
        svc.validate(fake)
    assert "Invalid file type" in str(exc.value)


def test_validator_rejects_oversized_file():
    # set max_mb to 0 so any non-empty file is too large
    svc = ValidatorService(max_mb=0)
    fake = _FakeUpload(b"0" * 1024, filename="big.jpg", content_type="image/jpeg")
    with pytest.raises(ValidationError) as exc:
        svc.validate(fake)
    assert "File too large" in str(exc.value)
