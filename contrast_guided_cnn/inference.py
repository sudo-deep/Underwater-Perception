import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import os
import numpy as np

# Define model architecture
class UAENet(nn.Module):
    def __init__(self):
        super(UAENet, self).__init__()
        # Initial convolution layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=1, stride=1)

        # Depth-wise separable convolutions
        self.depthwise_conv4 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, groups=32)
        self.group_norm4 = nn.GroupNorm(8, 32)

        self.depthwise_conv5 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, groups=32)
        self.group_norm5 = nn.GroupNorm(8, 32)

        self.depthwise_conv6 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, groups=64)
        self.group_norm6 = nn.GroupNorm(8, 64)

        # Final convolutional layer with Tanh activation
        self.final_conv = nn.Conv2d((64 + 3), 3, kernel_size=3, stride=1, padding=1)
        self.group_norm_final = nn.GroupNorm(3, 3)

        self.tanh = nn.Tanh()

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = F.leaky_relu(x1, 0.2)
        x2 = self.conv2(x1)
        x2 = F.leaky_relu(x2, 0.2)
        x3 = self.conv3(x2)
        x3 = F.leaky_relu(x3, 0.2)
        x4 = self.depthwise_conv4(x3)
        x4 = self.group_norm4(x4)
        x4 = F.leaky_relu(x4, 0.2)

        x5 = self.depthwise_conv5(x3)
        x5 = F.leaky_relu(x5, 0.2)
        x5 = self.group_norm5(x5)
        x6 = self.depthwise_conv6(torch.cat((x5, x4), dim=1))
        x6 = F.leaky_relu(x6, 0.2)
        x6 = self.group_norm6(x6)

        # Final output layer
        x = self.tanh(self.final_conv(torch.cat((x6, x), dim=1)))
        # x = self.group_norm_final(x)

        return x

# Define enhancement function
def enhance_image(image, uaenet, iterations=8):
    for _ in range(iterations):
        image = image + uaenet(image) * image * (1 - image)
    return image

# Define model loading function
def load_model(model, checkpoint_path, device):
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    print(f"Model loaded from {checkpoint_path}")
    return model

# Define image processing functions
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)

def postprocess_image(tensor):
    # Returns a numpy array in the range [0, 1]
    image = tensor.squeeze(0).detach().cpu().numpy().transpose(1, 2, 0)
    return (image - image.min()) / (image.max() - image.min())

# Process a folder of images and save the results
def process_folder(model, input_folder, output_folder, device, iterations=2):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Loop over all files in the input folder.
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(input_folder, filename)
            print(f"Processing {image_path} ...")
            
            # Preprocess the image.
            input_image = preprocess_image(image_path).to(device)
            
            # Run enhancement.
            with torch.no_grad():
                enhanced_image = enhance_image(input_image, model, iterations=iterations)
            
            # Postprocess to get a numpy image.
            enhanced_image_np = postprocess_image(enhanced_image)
            enhanced_image_np = (enhanced_image_np * 255).astype(np.uint8)
            
            # Convert the array to an image.
            pil_image = Image.fromarray(enhanced_image_np)
            
            # Save the enhanced image.
            output_path = os.path.join(output_folder, filename)
            pil_image.save(output_path)
            print(f"Saved enhanced image to {output_path}")

# Configuration
SAVE_DIR = "contrast_guided_cnn/saved_models"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize model and load weights
model = UAENet()
model = load_model(model, os.path.join(SAVE_DIR, "uae_net_epoch_10.pth"), device)

# Set input and output folder paths
input_folder = "data/oceandark/samples"      # Change this to your folder path
output_folder = "data/oceandark/output"     # Change this if needed

# Process the folder and save the results
process_folder(model, input_folder, output_folder, device, iterations=2)

torch.cuda.empty_cache()
