# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 00:31:10 2024

@author: Akash
"""

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt

# Load the pre-trained ResNet18 model
resnet18 = models.resnet18(pretrained=True)

# Modify the last layer to match the number of your classes if needed
# For example, for 3 classes:
resnet18.fc = nn.Linear(resnet18.fc.in_features, 3)

# Ensure the model is in evaluation mode
resnet18.eval()

# Define a hook function to capture activations
activations = []
def hook_fn(module, input, output):
    activations.append(output)

# Register hooks on all convolutional layers and maxpool
for name, layer in resnet18.named_modules():
    if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.MaxPool2d):
        layer.register_forward_hook(hook_fn)

# Define a transformation to preprocess the input image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load and preprocess an image
image = Image.open(r'E:\MS_AI\Sem-1\3.png').convert('RGB')
input_image = transform(image).unsqueeze(0)  # Add batch dimension

# Forward pass through the model
with torch.no_grad():
    output = resnet18(input_image)

# Visualize the activations
for i, activation in enumerate(activations):
    print(activation.shape)
    # Visualize the first filter's activation map for each layer
    plt.figure(figsize=(25, 25))
    plt.imshow(activation[0][0].cpu(), cmap='viridis')
    plt.title(f'Activation of Layer {i+1}', fontsize=60)
    plt.show()

# Visualize the weights of each convolutional layer
for i, (name, layer) in enumerate(resnet18.named_modules()):
    if isinstance(layer, nn.Conv2d):
        conv_weights = layer.weight.data
        print(f"Layer {i+1} - Weights Shape: {conv_weights.shape}")
        # Visualize the weights of the first filter
        plt.figure(figsize=(25, 25))
        plt.imshow(conv_weights[0][0].cpu(), cmap='viridis')
        plt.title(f'Weights of Layer {i+1}', fontsize=60)
        plt.show()

