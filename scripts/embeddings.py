import os
import csv
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from facenet_pytorch import InceptionResnetV1

class FaceModel:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"[INFO] Loading FaceNet model on {cls._instance.device}...")
            cls._instance.model = InceptionResnetV1(pretrained="vggface2").eval().to(cls._instance.device)
        return cls._instance

    def get_embedding(self, image: Image.Image) -> np.ndarray:
        """Return 512-dim embedding for a given cropped face image."""
        try:
            # Convert to tensor (C, H, W) and normalize
            img_tensor = torch.tensor(np.array(image)).permute(2, 0, 1).float() / 255.0
            img_tensor = img_tensor.unsqueeze(0).to(self.device)

            with torch.no_grad():
                embedding = self.model(img_tensor).cpu().numpy().flatten()
            return embedding
        except Exception as e:
            print(f"[WARN] Failed to get embedding: {e}")
            return None


def load_images_from_folder(folder_path: str, label: int):
    """Yields (image, label, filename) for all images in a folder."""
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(folder_path, filename)
            try:
                img = Image.open(img_path).convert("RGB")
                yield img, label, filename
            except Exception as e:
                print(f"[WARN] Could not open {filename}: {e}")
                continue


def generate_embeddings(base_dir="data/cropped", output_dir="data/embeddings"):
    os.makedirs(output_dir, exist_ok=True)

    me_dir = os.path.join(base_dir, "me", "photos")
    not_me_dir = os.path.join(base_dir, "not_me", "photos")

    face_model = FaceModel()

    embeddings = []
    labels = []
    files = []

    print("[INFO] Generating embeddings...")

    # Process both folders
    for folder, label in [(me_dir, 1), (not_me_dir, 0)]:
        if not os.path.exists(folder):
            print(f"[WARN] Folder missing: {folder}")
            continue

        for img, label, filename in tqdm(load_images_from_folder(folder, label), desc=f"Processing {os.path.basename(folder)}"):
            emb = face_model.get_embedding(img)
            if emb is not None:
                embeddings.append(emb)
                labels.append(label)
                files.append(filename)

    embeddings = np.array(embeddings)
    labels = np.array(labels)

    # Save .npy
    np.save(os.path.join(output_dir, "embeddings.npy"), embeddings)
    np.save(os.path.join(output_dir, "labels.npy"), labels)

    # Save CSV with metadata
    csv_path = os.path.join(output_dir, "metadata.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "label"])
        writer.writerows(zip(files, labels))

    print(f"[INFO] Saved {len(embeddings)} embeddings to {output_dir}/")
    print(f"  embeddings.npy shape: {embeddings.shape}")
    print(f"  labels.npy shape: {labels.shape}")


if __name__ == "__main__":
    generate_embeddings()
