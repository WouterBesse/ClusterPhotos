from pathlib import Path
from util import DimensionReducer, load_resnet_feature_extractor, TripletDataset
import torch
import torch.nn as nn
import os
import random
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader
import torch.optim as optim


class ImageSI:
    """
    Main class to manage the ImageSI interactive projection workflow.
    """

    def __init__(self, image_folder_path, sample_size=20):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"current device: {self.device}")
        self.image_paths = self._load_image_paths(image_folder_path, sample_size)
        self.model = load_resnet_feature_extractor().to(self.device)
        self.reducer = DimensionReducer()

        # Initial feature extraction and projection
        self.features = self.extract_all_features()
        self.projection: np.ndarray = self.reducer.fit_transform(self.features)

    def _load_image_paths(self, folder_path, sample_size):
        """Loads a sample of image paths from the specified folder."""
        paths = []
        for root, _, files in os.walk(folder_path):
            for file in files:
                if file.lower().endswith(("png", "jpg", "jpeg")):
                    paths.append(os.path.join(root, file))
        return random.sample(paths, min(sample_size, len(paths)))

    def extract_all_features(self):
        """Extracts features for all images using the current model state."""
        self.model.eval()
        features_list = []
        transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        with torch.no_grad():
            for path in tqdm(self.image_paths, desc="Extracting Features"):
                image = Image.open(path).convert("RGB")
                image_tensor = transform(image).unsqueeze(0).to(self.device)
                feature = self.model(image_tensor)
                features_list.append(feature.squeeze().cpu().numpy())
        return np.array(features_list)

    def update_model(self, updated_positions, epochs=5, lr=1e-4, margin=1.0):
        """
        Fine-tunes the ResNet model based on new user-defined positions.

        Args:
            updated_positions (dict): A dictionary mapping image paths to new (x, y) coords.
            epochs (int): Number of epochs to fine-tune.
            lr (float): Learning rate for the optimizer.
            margin (float): The margin for the triplet loss.
        """
        print("Starting model fine-tuning...")
        # Get current positions as a standardized list
        current_positions = []
        for path in self.image_paths:
            # If a path was updated, use its new position, otherwise use its old one
            current_positions.append(
                updated_positions.get(
                    path, self.projection[self.image_paths.index(path)]
                )
            )
        print(current_positions)

        # Setup dataset, dataloader, loss, and optimizer
        dataset = TripletDataset(self.image_paths, current_positions)
        dataloader = DataLoader(dataset, batch_size=256, shuffle=True)
        triplet_loss = nn.TripletMarginLoss(margin=margin)
        optimizer = optim.Adam(self.model.parameters(), lr=lr)

        self.model.train()
        for epoch in range(epochs):
            yield (epoch + 1) / epochs * 100
            total_loss = 0
            for anchor, positive, negative in tqdm(
                dataloader, desc=f"Epoch {epoch + 1}/{epochs}"
            ):
                anchor, positive, negative = (
                    anchor.to(self.device),
                    positive.to(self.device),
                    negative.to(self.device),
                )

                optimizer.zero_grad()

                # Get embeddings from the model
                anchor_embedding = self.model(anchor).flatten(start_dim=1)
                positive_embedding = self.model(positive).flatten(start_dim=1)
                negative_embedding = self.model(negative).flatten(start_dim=1)

                # Calculate loss and perform backpropagation
                loss = triplet_loss(
                    anchor_embedding, positive_embedding, negative_embedding
                )
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                yield (epoch + 1) / epochs * 100

            yield (epoch + 1) / epochs * 100

            avg_loss = total_loss / len(dataloader)
            print(f"Epoch {epoch + 1}/{epochs}, Average Loss: {avg_loss:.4f}")

        print("Fine-tuning finished. Re-projecting images...")
        # After training, update features and projection for all images
        self.features = self.extract_all_features()
        self.projection = self.reducer.fit_transform(self.features)
        print("Projection updated.")

    def save_checkpoint(self, path: Path) -> None:
        checkpoint = {
            "state_dict": self.model.state_dict(),
            "projection": self.projection,
            "features": self.features,
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: Path) -> None:
        checkpoint: dict[str, np.ndarray | dict] = torch.load(path, weights_only=False)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.features = checkpoint["features"]
        self.projection = checkpoint["projection"]
