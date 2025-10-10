import os
import time

from app.model.face_model import FaceModel
from app.model.model_stats import ModelStats
from app.utils.logger import Logger

class ClassifierService:
    """
    Service to run face verification.
    For now, it returns dummy results; later it will call the real model.
    """

    def __init__(self, threshold=os.getenv("THRESHOLD", 0.75), model_stats=None):
        self.model = FaceModel.get_instance()
        self.threshold = threshold
        self.model_stats = ModelStats.get_instance()
        self.logger = Logger()

    def verify(self, image) -> dict:
        """
        Accepts an image (PIL or bytes) and returns a prediction dictionary.
        """
        self.logger.info("[Classifier] Starting verification")
        start_time = time.time()

        # Dummy prediction logic
        score = self.model.predict(image)  # random float 0-1
        is_me = score >= self.threshold
        time.sleep(0.5)  # Simulate processing time

        # Calculate latency
        timing_ms = round((time.time() - start_time) * 1000, 2)

        # Optional: log to ModelStats if provided
        if self.model_stats:
            self.model_stats.record_inference({
                "score": score,
                "is_me": is_me,
                "threshold": self.threshold,
                "timing_ms": timing_ms
            })
            
        self.logger.info(f"[Classifier] Verification done in {timing_ms} ms: is_me={is_me}, score={score}")
        # Return formatted response
        return {
            "model_version": self.model.version,
            "is_me": is_me,
            "score": round(score, 2),
            "threshold": self.threshold,
            "timing_ms": timing_ms
        }
