import time
from facenet_pytorch import MTCNN
from PIL import Image, ExifTags
import torch
import os
from pathlib import Path

from tqdm import tqdm

class FacePreprocessor:
    def __init__(self, image_size=160, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.mtcnn = MTCNN(image_size=image_size, margin=0, device=self.device)

    def crop(self, image: Image.Image):
        """Returns cropped face as PIL.Image or None."""
        try:
            # Fix image orientation
            image = self.fix_orientation(image)
            
            # Detect face and return as tensor
            face_tensor = self.mtcnn(image)
            if face_tensor is None:
                return None

            # Convert image to tensor and scale to [0, 255]
            face_tensor = (face_tensor.clamp(0, 1) * 255).byte()

            # Convert tensor (C,H,W) to PIL image
            face_img = Image.fromarray(
                face_tensor.permute(1, 2, 0).int().cpu().numpy().astype("uint8")
            )
            return face_img
        except Exception as e:
            tqdm.write(f"[WARN] Could not process image: {e}")
            return None
        
    def fix_orientation(self, image: Image.Image) -> Image.Image:
        """Rotate image according to its EXIF orientation tag."""
        try:
            for orientation in ExifTags.TAGS.keys():
                if ExifTags.TAGS[orientation] == 'Orientation':
                    break
            exif = image._getexif()
            if exif is not None:
                if exif.get(orientation) == 3:
                    image = image.rotate(180, expand=True)
                elif exif.get(orientation) == 6:
                    image = image.rotate(270, expand=True)
                elif exif.get(orientation) == 8:
                    image = image.rotate(90, expand=True)
        except Exception:
            pass
        return image
    
def process_folder(source_dir: str, dest_dir: str, preprocessor: FacePreprocessor):
    tqdm.write(f"Processing images in {source_dir}...")

    max_saved = 300
    saved_count = 0

    # check that source folder exists and is a directory
    if not os.path.exists(source_dir) or not os.path.isdir(source_dir):
        tqdm.write(f"[ERROR] Folder path does not exist or is not a directory: {source_dir}")
    else:
        # build a filtered list of image files to get an accurate progress bar
        all_files = [f for f in os.listdir(source_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if len(all_files) < max_saved:
            max_saved = len(all_files)

        for filename in tqdm(all_files, desc="Images", unit="file", total=max_saved):
            if saved_count >= max_saved:
                tqdm.write(f"Reached cap of {max_saved} saved crops. Stopping.")
                break
            
            img_path = os.path.join(source_dir, filename)

            try:
                img = Image.open(img_path)
                cropped_face = preprocessor.crop(img).convert("RGB")
                if cropped_face:
                    save_path = os.path.join(dest_dir, filename)
                    cropped_face.save(save_path)
                    saved_count += 1
                    tqdm.write(f"Saved cropped face to {save_path} (#{saved_count})")
                else:
                    tqdm.write(f"No face detected in {filename}")
            except Exception as e:
                tqdm.write(f"[WARN] Could not open {img_path}: {e}")
                continue

def main():
    initial_time = time.time()
    face_preprocessor = FacePreprocessor()

    ### Process "me" images
    # source directory with images to process
    base_dir = Path.cwd()
    me_folder_path = base_dir / "data" / "raw" / "me" / "photos"
    # destination directory for cropped faces
    me_dest_dir = base_dir / "data" / "cropped" / "me" / "photos"

    process_folder(me_folder_path, me_dest_dir, face_preprocessor)


    ### Process "not_me" images
    # source directory with images to process
    base_dir = Path.cwd()
    not_me_folder_path = base_dir / "data" / "raw" / "not_me" / "photos"
    # destination directory for cropped faces
    not_me_dest_dir = base_dir / "data" / "cropped" / "not_me" / "photos"

    process_folder(not_me_folder_path, not_me_dest_dir, face_preprocessor)
    total_time = time.time() - initial_time
    tqdm.write(f"Total processing time: {total_time:.2f} seconds")

if __name__ == "__main__":
    main()