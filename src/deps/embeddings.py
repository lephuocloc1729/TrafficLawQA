import torch
import numpy as np
import torch.nn.functional as F
from typing import Callable
from PIL import Image
from transformers import (
    CLIPModel,
    CLIPProcessor,
    AutoModel,
    AutoProcessor,
    AutoImageProcessor,
)
from src.settings import settings


class CLIPEmbedding:
    def __init__(
        self,
        model_name_or_path: str = "zer0int/CLIP-GmP-ViT-L-14",
        device: str = "cpu",
    ):
        self.device = device
        self.processor = CLIPProcessor.from_pretrained(
            model_name_or_path, use_fast=True
        )
        self.model = CLIPModel.from_pretrained(model_name_or_path)
        self.model.to(device).eval()

    def embed_image(self, image: Image.Image) -> np.ndarray:
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            features = self.model.get_image_features(**inputs)
            # L2 Normalizaion
            return F.normalize(features, p=2, dim=1).detach().cpu().numpy().flatten()

    def parameters(self):
        return sum(p.numel() for p in self.model.parameters())


class SigLIPEmbedding:
    def __init__(self, model_name_or_path: str = "google/siglip2-so400m-patch14-384"):
        self.model = AutoModel.from_pretrained(
            model_name_or_path,
            device_map="auto",
        ).eval()
        self.processor = AutoProcessor.from_pretrained(
            model_name_or_path, use_fast=True
        )

    def embed_image(self, image: Image.Image) -> np.ndarray:
        inputs = self.processor(images=[image], return_tensors="pt").to(
            self.model.device
        )
        with torch.no_grad():
            features = self.model.get_image_features(**inputs)
            # L2 Normalizaion
            return F.normalize(features, p=2, dim=1).detach().cpu().numpy().flatten()

    def parameters(self):
        return sum(p.numel() for p in self.model.parameters())


class DINOv2Embedding:
    def __init__(
        self,
        model_name_or_path: str = "facebook/dinov2-with-registers-large",
        device: str = "auto",
    ):
        self.processor = AutoImageProcessor.from_pretrained(
            model_name_or_path, use_fast=True
        )
        self.model = AutoModel.from_pretrained(
            model_name_or_path, device_map=device
        ).eval()

    def embed_image(self, image: Image.Image) -> np.ndarray:
        image = image.convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            features = self.model(**inputs).last_hidden_state[:, 0]
            return F.normalize(features, p=2, dim=1).detach().cpu().numpy().flatten()

    def parameters(self):
        return sum(p.numel() for p in self.model.parameters())


class DINOv3Embedding:
    def __init__(
        self,
        model_name_or_path: str = "facebook/dinov3-vitsplus-pretrain-lvd1689m",
        device: str = "auto",
    ):
        self.processor = AutoImageProcessor.from_pretrained(
            model_name_or_path,
            use_fast=True,
            token=settings.huggingface_token,
        )
        self.model = AutoModel.from_pretrained(
            model_name_or_path,
            device_map=device,
            token=settings.huggingface_token,
        ).eval()

        patch_size = self.model.config.patch_size
        num_register_tokens = self.model.config.num_register_tokens
        print("Patch size:", patch_size)  # 16
        print("Num register tokens:", num_register_tokens)  # 4

    def embed_image(self, image: Image.Image) -> np.ndarray:
        image = image.convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt").to(self.model.device)
        print("Preprocessed image size:", inputs.pixel_values.shape)  # [1, 3, 224, 224]
        with torch.no_grad():
            outputs = self.model(**inputs)  # # [1, 1 + 4 + 256, 384]
            last_hidden_states = outputs.last_hidden_state
            cls_token = last_hidden_states[:, 0, :]

        return F.normalize(cls_token, p=2, dim=1).detach().cpu().numpy().flatten()

    def parameters(self):
        return sum(p.numel() for p in self.model.parameters())


embedding_model_settings = {
    "CLIP-GmP-ViT-L-14": {
        "model_path": "zer0int/CLIP-GmP-ViT-L-14",
        "model_dim": 768,
        "parameters": 427_616_513,
    },
    "siglip2-so400m-patch14-384": {
        "model_path": "google/siglip2-so400m-patch14-384",
        "model_dim": 1152,
        "parameters": 1_136_008_498,
    },
    "siglip2-base-patch16-384": {
        "model_path": "google/siglip2-base-patch16-384",
        "model_dim": 768,
        "parameters": 375_479_810,
    },
    "siglip2-base-patch16-224": {
        "model_path": "google/siglip2-base-patch16-224",
        "model_dim": 768,
        "parameters": 375_187_970,
    },
    "siglip2-large-patch16-384": {
        "model_path": "google/siglip2-large-patch16-384",
        "model_dim": 1024,
        "parameters": 881_854_466,
    },
    "dinov2-with-registers-large": {
        "model_path": "facebook/dinov2-with-registers-large",
        "model_dim": 1024,
    },
    "dinov2-with-registers-base": {
        "model_path": "facebook/dinov2-with-registers-base",
        "model_dim": 768,
    },
    "dinov2-large": {
        "model_path": "facebook/dinov2-large",
        "model_dim": 1024,
    },
    "dinov3-vitsplus-pretrain-lvd1689m": {
        "model_path": "facebook/dinov3-vitsplus-pretrain-lvd1689m",
    },
}


def get_embedding_model(name: str, device: str = "cpu") -> tuple[Callable, int]:
    if name not in embedding_model_settings:
        raise ValueError(
            f"Invalid embedding model: {name}, Supported: {list(embedding_model_settings.keys())}"
        )

    model_name_or_path = embedding_model_settings[name]["model_path"]
    embedding_dim = embedding_model_settings[name]["model_dim"]
    if "clip" in name.lower():
        embedder = CLIPEmbedding(model_name_or_path, device)
    elif "siglip" in name.lower():
        embedder = SigLIPEmbedding(model_name_or_path)
    elif "dinov2" in name.lower():
        embedder = DINOv2Embedding(model_name_or_path, device)
    elif "dinov3" in name.lower():
        embedder = DINOv3Embedding(model_name_or_path, device)
    else:
        raise ValueError(f"Invalid embedding model: {name}")

    return embedder, embedding_dim


if __name__ == "__main__":
    image = Image.open("data/law_db/images.fld/image001.png").convert("RGB")
    embedding_name = "siglip2-large-patch16-384"
    embedder, embedding_dim = get_embedding_model(embedding_name)
    embedded = embedder.embed_image(image)
    print(f"Embedding Name: {embedding_name}")
    print(f"Embedding Dim: {embedding_dim}")
    print(f"Embedding Size: {embedded.shape}")
    print(f"Parameters: {embedder.parameters()}")
