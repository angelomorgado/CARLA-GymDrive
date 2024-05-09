from torchsummary import summary
import torchinfo
import torch
import torch.nn as nn

class DQNNetwork(nn.Module):
    def __init__(self, output_n):
        super(DQNNetwork, self).__init__()

        # Define the neural network architecture
        # Grayscale: (224, 224) -> 512
        self.model1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)  # Global average pooling to get a fixed-size feature vector
        )

        # Rest: 7
        self.model3 = nn.Sequential(
            nn.Linear(7, 256),
        )

        self.final_model = nn.Sequential(
            nn.Linear(512 + 256, 512),  # Combine image and rest features, output 512-dimensional vector
            nn.ReLU(),
            nn.Linear(512, output_n)
        )

        # Initialization using Xavier uniform
        for layer in [self.model1, self.model3, self.final_model]:
            for sub_layer in layer:
                if isinstance(sub_layer, nn.Linear) or isinstance(sub_layer, nn.Conv2d):
                    nn.init.xavier_uniform_(sub_layer.weight)
                    nn.init.constant_(sub_layer.bias, 0.0)

    def forward(self, rgb_input, rest_input):
        # Forward pass through the network
        image_features = self.model1(rgb_input)
        image_features = torch.squeeze(image_features)  # Remove dummy dimensions
        print(image_features.shape)
        rest_output = self.model3(rest_input)
        print(rest_output.shape)
        combined_features = torch.cat((image_features, rest_output), dim=0)
        return self.final_model(combined_features)


model = DQNNetwork(4).to("cuda")

# Print the summary of the model
torchinfo.summary(model, [(1, 224, 224), (7,)])