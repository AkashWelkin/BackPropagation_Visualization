import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2

def load_image(image_path):
    image = Image.open(image_path).convert('RGB')
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_tensor = preprocess(image).unsqueeze(0)  
    return input_tensor

def visualize(tensor, title):
    tensor = tensor.squeeze().detach().cpu() 
    if tensor.dim() == 3:
        tensor = tensor.permute(1, 2, 0)  
    tensor = tensor.numpy()
    tensor = tensor - tensor.min()  
    tensor = tensor / tensor.max()
    print(tensor.shape)
    tensor = np.clip(tensor, 0, 1)
    np_img = cv2.GaussianBlur(tensor, (3, 3), sigmaX=0.2)
    plt.imshow(np_img,cmap='viridis')
    plt.title(title)
    plt.axis('off')
    plt.show()

class GuidedBackpropReLU(nn.Module):
    def forward(self, input):
        return GuidedBackpropReLUFunction.apply(input)

class GuidedBackpropReLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input < 0] = 0
        grad_input[grad_output < 0] = 0
        return grad_input

class GuidedBackpropReLUModel(nn.Module):
    def __init__(self, model):
        super(GuidedBackpropReLUModel, self).__init__()
        self.model = self._replace_relu_with_guided_backprop(model)

    def _replace_relu_with_guided_backprop(self, module):
        for name, child in module.named_children():
            if isinstance(child, nn.ReLU):
                setattr(module, name, GuidedBackpropReLU())
            elif isinstance(child, nn.Sequential): 
                self._replace_relu_with_guided_backprop(child)
        return module

    def forward(self, x):
        return self.model(x)


image_path = r'file_path'
input_tensor = load_image(image_path)
input_tensor.requires_grad_()

vgg16 = models.vgg16(pretrained=True)
vgg16.classifier[6] = nn.Linear(4096, 5)  
vgg16.eval()

activations = []
def hook_fn(module, input, output):
    activations.append(output)

for layer in vgg16.features:
    if isinstance(layer, nn.Conv2d):
        layer.register_forward_hook(hook_fn)


output = vgg16(input_tensor)
predicted_class = output.argmax().item()
vgg16.zero_grad()
output[0, predicted_class].backward()
saliency, _ = torch.max(input_tensor.grad.data.abs(), dim=1)
saliency = saliency.squeeze().numpy()
saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min())
saliency = cv2.GaussianBlur(saliency, (1, 1), sigmaX=0.8)

plt.figure(figsize=(10, 10))
plt.imshow(saliency, cmap='viridis')
plt.title('Saliency Map',fontsize=30)
plt.axis('off')
plt.show()

guided_model = GuidedBackpropReLUModel(vgg16)
output = guided_model(input_tensor)

target_class = output.argmax().item()
guided_model.zero_grad()
output[0, target_class].backward()

gradients = input_tensor.grad.data
visualize(gradients, "Guided Backpropagation")
