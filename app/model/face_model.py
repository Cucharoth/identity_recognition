from io import BytesIO
import os
from fastapi import UploadFile
import numpy as np
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image, ImageOps
import joblib

from app.exceptions.no_face_exception import NoFaceDetectedError
from app.utils.logger import Logger

class FaceModel:
    _instance = None

    def __init__(self, classifier_path=None, scaler_path=None, device=None, model_version="me-verifier-v1"):
        # read env when an instance is created (not at import time)
        classifier_path = classifier_path or os.getenv("MODEL_PATH")
        scaler_path = scaler_path or os.getenv("SCALER_PATH")

        if FaceModel._instance is not None:
            raise Exception("FaceModel is a singleton! Use get_instance()")

        self.version = model_version
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Face detection and embeddings
        self.mtcnn = MTCNN(image_size=160, margin=0, device=self.device)
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)

        # Load classifier and scaler
        self.classifier = None
        self.scaler = None
        if classifier_path:
            self.load_classifier(classifier_path)
        if scaler_path:
            self.load_scaler(scaler_path)
        
        FaceModel._instance = self
        self.logger = Logger()

    @classmethod
    def get_instance(cls, classifier_path=None, scaler_path=None, device=None):
        if cls._instance is None:
            cls(classifier_path=classifier_path, scaler_path=scaler_path, device=device)
        return cls._instance

    def load_classifier(self, path):
        self.classifier = joblib.load(path)

    def load_scaler(self, path):
        self.scaler = joblib.load(path)

    def predict(self, imageFile: UploadFile):
        """
        Returns probability score for "me" class.
        """
        self.logger.info("[FaceModel] Starting prediction")
        if self.classifier is None or self.scaler is None:
            raise Exception("Classifier and scaler must be loaded before prediction.")

        self.logger.info("[FaceModel] Image pre-processing")
        # Properly reads the image and fixes orientation
        image = Image.open(BytesIO(imageFile.file.read()))
        imageFile.file.seek(0)
        image = ImageOps.exif_transpose(image)

        # Detect and crop face -> returns tensor (C, H, W) float in [0, 1]
        face = self.mtcnn(image)
        if face is None:
            raise NoFaceDetectedError("No face detected in the image.")

        self.logger.info("[FaceModel] Extracting embedding")
        # Convert image to tensor and scale to [0, 255]
        face_tensor = (face.clamp(0, 1) * 255).byte()

        # Simulates training process
        face_img = Image.fromarray(
            face_tensor.permute(1, 2, 0).cpu().numpy().astype("uint8")
        )
        face_img.save("./data/detected_face.jpg")  # For debugging purposes

        img_tensor = torch.tensor(np.array(face_img)).permute(2, 0, 1).float() / 255.0

        # Get embedding (expects batch tensor of floats)
        face_batch = img_tensor.unsqueeze(0).to(self.device)
        with torch.no_grad():
            embedding = self.resnet(face_batch).cpu().numpy()

        # Scale embedding (sklearn expects 2D arrays)
        embedding_scaled = self.scaler.transform(embedding)

        self.logger.info("[FaceModel] Making prediction")
        # Predict probability for the "me" class
        score = float(self.classifier.predict_proba(embedding_scaled)[0, 1])
        return score
