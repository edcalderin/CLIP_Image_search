import torch
from torch.utils.data import Dataset, DataLoader
from typing import Any
from PIL import Image
import clip
import logging
from pathlib import Path
from tqdm import tqdm
import pickle

logging.basicConfig(level=0)

class Images(Dataset):
    def __init__(self, image_list, transform) -> None:
        super().__init__()
        self._image_list = image_list
        self._transform = transform

    def __len__(self):
        return len(self._image_list)
    
    def __getitem__(self, index) -> Any:
        image_path = self._image_list[index]
        image = Image.open(image_path)
        image = self._transform(image)
        return {
            "image": image,
            "img_path": image_path
        }

class EmbeddingGenerator:
    def __init__(self) -> None:
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model, self._preprocess = clip.load("ViT-B/32", self._device, jit=False)
        logging.info(f"Device used: {self._device}")

    @staticmethod
    def _filter_valid_images():
        images_path = Path(__file__).parent.parent / "images"
        cleaned_images: list = []
        for image in images_path.iterdir():
            try:
                Image.open(image)
                cleaned_images.append(image.as_posix())
            except:
                logging.error(f"Failed opening for {image}")

        logging.info(f"There are {len(cleaned_images)} images that can be processed")
        return cleaned_images
    
    def _encode_images(self, data)->tuple:
        with torch.no_grad():
            X = data["image"].to(self._device)
            image_embedding = self._model.encode_image(X)
            img_path = data["img_path"]
            return {
                "image_path": img_path, 
                "image_embedding": image_embedding.cpu()
            }

    def _create_data_loader(self):
        dataset = Images(self._filter_valid_images(), self._preprocess)
        return DataLoader(dataset, batch_size=256, shuffle=True)
    
    def _compute_embeddings(self):
        logging.info("Processing images...")
        image_paths, embeddings = [], []
        for data in tqdm(self._create_data_loader()):
            encoded_images = self._encode_images(data)
            image_paths.extend(encoded_images["image_path"])
            embeddings.extend(encoded_images["image_embedding"])
        return image_paths, embeddings
    
    def save_embeddings(self):
        image_paths, embeddings = self._compute_embeddings()

        image_embeddings: dict = dict(zip(image_paths, embeddings))

        logging.info("Saving image embeddings")
        with open("embeddings.pkl", "wb") as f:
            pickle.dump(image_embeddings, f)
    
if __name__ == "__main__":
    embedding_generator = EmbeddingGenerator()
    embedding_generator.save_embeddings()

    

    
        
