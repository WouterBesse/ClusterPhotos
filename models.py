import torch.nn as nn
from torchvision import models


class TripletResNetModel(nn.Module):
    def __init__(self):
        super(TripletResNetModel, self).__init__()
        self.Model = models.resnet18(pretrained=True)
        num_filters = (
            self.Model.fc.in_features
        )  # The number of input features for the fully connected layer
        self.Model.fc = nn.Sequential(  # nn.Sequential is a container module that sequentially applies a list of layers
            nn.Linear(
                num_filters, 512
            ),  # Linear transformation layer that maps the input features (the output of the last convolutional layer) to 512 output features.
            nn.LeakyReLU(),  # Activation function to introduce non-linearity to the output of the linear layer; Allows non zero gradients for negative inputs, mitigate vanishing gradient problem
            nn.Linear(512, 10),
        )  # Linear transformation layer as the final classification layer, each output corresponds to a class in the classification task
        self.Triplet_Loss = nn.Sequential(  # Replace the fully connected layer/classification layer with a triplet loss layer
            nn.Linear(10, 2)
        )  # Maps it to a 2D space, represent the distances between the anchor, positive, and negative samples in the triplet loss

    def forward(self, x):
        self.Model.eval()
        features = self.Model(x)
        triplets = self.Triplet_Loss(features)
        return triplets


class Autoencoder(nn.Module):
    def __init__(self, input_size, output_size):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Linear(input_size, output_size)
        self.decoder = nn.Linear(output_size, input_size)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


class InitialTripletResNetModel(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(InitialTripletResNetModel, self).__init__()
        self.Model = models.resnet18(pretrained=True)
        num_filters = self.Model.fc.in_features
        self.Model.fc = nn.Sequential(
            nn.Linear(num_filters, 512), nn.LeakyReLU(), nn.Linear(512, input_size)
        )  # Adjust the output size to match the input size of the autoencoder
        self.Autoencoder = Autoencoder(input_size, hidden_size)

    def forward(self, x):
        features = self.Model(x)
        encoded, _ = self.Autoencoder(features)  # Only need the encoded representation
        return encoded


class CustomResNetModel(nn.Module):
    def __init__(self):
        super(CustomResNetModel, self).__init__()
        # Load the pre-trained ResNet model without the classification layer
        self.pretrained_model = models.resnet18(pretrained=True)
        # remove fully connected layer
        self.model = nn.Sequential(*list(self.pretrained_model.children())[:-1])

    def forward(self, x):
        features = self.model(x)
        return features
