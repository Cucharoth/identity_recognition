from facenet_pytorch import MTCNN
from PIL import Image
import torch

class FacePreprocessor:
    def __init__(self, image_size=160, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.mtcnn = MTCNN(image_size=image_size, margin=0, device=self.device)

    def crop(self, image: Image.Image):
        """Returns cropped face as PIL.Image or None."""
        try:
            face_tensor = self.mtcnn(image)
            if face_tensor is None:
                return None
            
            # Convert tensor (C,H,W) to PIL image
            face_img = Image.fromarray(
                face_tensor.permute(1, 2, 0).int().cpu().numpy().astype("uint8")
            )
            return face_img
        except Exception as e:
            print(f"[WARN] Could not process image: {e}")
            return None