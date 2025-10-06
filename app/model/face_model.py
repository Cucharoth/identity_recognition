from facenet_pytorch import MTCNN, InceptionResnetV1
import torch

class FaceModel:
    _instance = None

    def __init__(self, classifier_path=None, device=None, model_version="me-verifier-v1"):
        if FaceModel._instance is not None:
            raise Exception("FaceModel is a singleton! Use get_instance()")
        self.version = model_version

        # Device setup
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Load MTCNN for face detection
        self.mtcnn = MTCNN(keep_all=False, device=self.device)

        # Load InceptionResnetV1 (facenet) for embeddings
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)

        self.classifier = None
        self.classifier_path = classifier_path

        FaceModel._instance = self

    @classmethod
    def get_instance(cls, classifier_path=None, device=None):
        if cls._instance is None:
            cls(classifier_path=classifier_path, device=device)
        return cls._instance

    def load_classifier(self, path):
        """
        Later: load your trained sklearn classifier (.joblib)
        """
        self.classifier_path = path
        # Example:
        # import joblib
        # self.classifier = joblib.load(path)

    def predict(self, image):
        # 1. Detect face (stub)
        # face = self.mtcnn(image)
        # if face is None:
        #     raise ValueError("No face detected")

        # 2. Generate embedding (stub)
        # embedding = self.resnet(face.unsqueeze(0).to(self.device))

        # 3. Run classifier (stub)
        # score = self.classifier.predict_proba(embedding)[0][1]

        # For now, return dummy
        import random
        score = random.uniform(0, 1)
        return score
