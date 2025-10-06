import csv
import os
import pandas as pd
from datetime import datetime
from threading import Lock

from app.utils.logger import Logger

class ModelStats:
    _instance = None
    _lock = Lock()

    def __init__(self, csv_path="inference_log.csv"):
        if ModelStats._instance is not None:
            raise Exception("ModelStats is a singleton! Use get_instance()")
        self.csv_path = csv_path
        self.logger = Logger()
        # If file doesn't exist, create and write header
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, mode="w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["timestamp", "score", "is_me", "threshold", "timing_ms"])
                writer.writeheader()
        ModelStats._instance = self

    @classmethod
    def get_instance(cls, csv_path="./data/inference_log.csv"):
        with cls._lock:
            if cls._instance is None:
                cls(csv_path)
            return cls._instance

    def record_inference(self, data: dict):
        """
        data: {
            "score": float,
            "is_me": bool,
            "threshold": float,
            "timing_ms": float
        }
        """
        self.logger.info("[ModelStats] Recording inference")
        data_to_write = {
            "timestamp": datetime.utcnow().isoformat(),
            "score": data["score"],
            "is_me": data["is_me"],
            "threshold": data["threshold"],
            "timing_ms": data["timing_ms"]
        }
        # Append to CSV
        with self._lock:
            with open(self.csv_path, mode="a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=data_to_write.keys())
                writer.writerow(data_to_write)

    def get_summary(self):
        """
        Optional: aggregate stats from CSV
        """
        if not os.path.exists(self.csv_path):
            return {}
        df = pd.read_csv(self.csv_path)
        return {
            "total_requests": len(df),
            "mean_score": df["score"].mean(),
            "mean_latency_ms": df["timing_ms"].mean(),
            "percent_positive": (df["is_me"].sum() / len(df)) * 100 if len(df) > 0 else 0
        }
