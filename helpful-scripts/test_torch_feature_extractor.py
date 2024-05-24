import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

resnet50 = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_resnet50', pretrained=True, trust_repo=True)
resnet50.eval().to(device)

# Print the model architecture
print(resnet50)