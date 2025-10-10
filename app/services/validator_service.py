import os
from app.exceptions.validation_exception import ValidationError
from app.utils.logger import Logger
from fastapi import UploadFile
from typing import Tuple



class ValidatorService:
    """Service to validate uploaded images."""

    def __init__(self, allowed_types=("image/jpeg", "image/png"), max_mb=os.getenv("MAX_MB", 5)):
        self.allowed_types = allowed_types
        self.max_bytes = max_mb * 1024 * 1024
        self.logger = Logger()

    def validate(self, file: UploadFile) -> bool:
        """
        Validate the uploaded file.
        Raises ValidationError if invalid.
        Returns True if valid.
        """
        self.logger.info("[Validator] Starting validation")
        # Check file type
        if file.content_type not in self.allowed_types:
            self.logger.error(f"[Validator] Invalid file type: {file.content_type}. Only JPEG/PNG allowed.")
            raise ValidationError(f"Invalid file type: {file.content_type}. Only JPEG/PNG allowed.")

        # Check file size
        file.file.seek(0, 2)  # Seek to end to get size
        size = file.file.tell()
        file.file.seek(0)     # Reset pointer
        if size > self.max_bytes:
            self.logger.error(f"[Validator] File too large: {size} bytes. Max allowed is {self.max_bytes} bytes.")
            raise ValidationError(f"File too large: {size} bytes. Max allowed is {self.max_bytes} bytes.")

        # Check that file is a valid image
        # from PIL import Image
        # try:
        #     Image.open(file.file).verify()
        # except Exception as e:
        #     raise ValidationError("File is not a valid image.") from e
        # file.file.seek(0)

        return True
