from torchvision import models, transforms
import torch.nn as nn
from PIL import Image
from sklearn.manifold import MDS
from sklearn.metrics.pairwise import euclidean_distances
from torch.utils.data import Dataset
import numpy as np
import random


def load_resnet_feature_extractor(pretrained=True):
    """
    Loads a pre-trained ResNet-18 model and removes its final
    fully connected layer to use it as a feature extractor.

    Args:
        pretrained (bool): Whether to load pretrained weights.

    Returns:
        torch.nn.Module: The ResNet model for feature extraction.
    """
    resnet = models.resnet18(pretrained=pretrained)
    # Remove the final classification layer (the fully connected layer)
    feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
    return feature_extractor


class DimensionReducer:
    """
    A class to perform dimensionality reduction using MDS.
    """

    def __init__(self, n_components=2, random_state=22, **mds_kwargs):
        """
        Initializes the DimensionReducer.

        Args:
            n_components (int): The target number of dimensions (e.g., 2 for a 2D plot).
            random_state (int): The seed for the random number generator.
            **mds_kwargs: Additional arguments for the MDS model.
        """
        self.mds = MDS(
            n_components=n_components,
            random_state=random_state,
            dissimilarity="precomputed",
            **mds_kwargs,
        )

    def fit_transform(self, high_dim_data) -> np.ndarray:
        """
        Computes the pairwise distances and then applies MDS.

        Args:
            high_dim_data (np.array): The high-dimensional feature data.

        Returns:
            np.array: The reduced 2D data.
        """
        # Calculate pairwise Euclidean distance matrix
        distance_matrix = euclidean_distances(high_dim_data)
        # Fit and transform the data using MDS
        low_dim_data = self.mds.fit_transform(distance_matrix)
        return low_dim_data


# --- 3. Custom Dataset for Triplet Mining ---


class TripletDataset(Dataset):
    """
    Dataset for generating triplets (anchor, positive, negative) based on 2D coordinates.
    A positive sample is close to the anchor in the 2D space, and a negative sample is far.
    """

    def __init__(self, image_paths, positions, distance_threshold=0.2):
        self.image_paths = np.array(image_paths)
        self.positions = np.array(positions)
        self.distance_threshold = distance_threshold
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        # Pre-compute pairwise distances for efficiency
        self.pairwise_dist = euclidean_distances(self.positions)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        anchor_path = self.image_paths[index]

        # Find positive samples (closer than threshold)
        positive_indices = np.where(
            self.pairwise_dist[index] < self.distance_threshold
        )[0]
        # Exclude the anchor itself from being a positive sample
        positive_indices = positive_indices[positive_indices != index]

        # Find negative samples (farther than threshold)
        negative_indices = np.where(
            self.pairwise_dist[index] >= self.distance_threshold
        )[0]

        # If no positives or negatives are found, pick randomly
        if len(positive_indices) == 0 or len(negative_indices) == 0:
            # Fallback: pick any other two points
            positive_idx = (index + 1) % len(self.image_paths)
            negative_idx = (index + 2) % len(self.image_paths)
        else:
            positive_idx = random.choice(positive_indices)
            negative_idx = random.choice(negative_indices)

        positive_path = self.image_paths[positive_idx]
        negative_path = self.image_paths[negative_idx]

        # Load and transform images
        anchor_img = self.transform(Image.open(anchor_path).convert("RGB"))
        positive_img = self.transform(Image.open(positive_path).convert("RGB"))
        negative_img = self.transform(Image.open(negative_path).convert("RGB"))

        return anchor_img, positive_img, negative_img
