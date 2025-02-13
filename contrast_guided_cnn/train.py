
# Import necessary libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import Subset
from torchvision import transforms
from torchvision.utils import save_image
from sklearn.model_selection import train_test_split
import os
from PIL import Image


ROOT_DIR = "data/oceandark/images"
SAVE_DIR = "contrast_guided_cnn/saved_models"


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(len(os.listdir(ROOT_DIR)))

# 1. Define the UAE-Net Network
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

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Create an instance of your model
uae_net = UAENet()

# Print the total number of trainable parameters
print(f"Total trainable parameters: {count_parameters(uae_net):,}")

def enhance_image(image, uaenet, iterations=8):
    """
    Apply the enhancement curve iteratively to the image.
    """
    for _ in range(iterations):
        # print(image.shape)
        image = image + uaenet(image) * image * (1 - image)
    return image

# def underwater_color_adaptive_correction_loss(enhanced, original):
#     """
#     Underwater color adaptive correction loss (Luac).
#     """
#     mean_original = original.mean(dim=[2, 3])  # Average over spatial dimensions
#     mean_enhanced = enhanced.mean(dim=[2, 3])
#     loss = sum((mean_original - mean_enhanced) ** 2).mean()
#     return loss

# def exposure_control_loss(enhanced, E=0.43, M=32):
#     """
#     Exposure control loss (Lexp).
#     """
#     patch_size = M
#     patches = enhanced.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
#     patch_means = patches.mean(dim=[4, 5])  # Mean intensity for each patch
#     loss = ((patch_means - E) ** 2).mean()
#     return loss

def illumination_smoothness_loss(delta):
    """
    Compute the illumination smoothness loss.
    
    Parameters:
      delta (torch.Tensor): Tensor of shape (N, C, H, W)
    
    Returns:
      torch.Tensor: The computed loss.
    """
    # Compute horizontal differences: skip the last column.
    diff_h = torch.abs(delta[:, :, :, 1:] - delta[:, :, :, :-1])
    
    # Compute vertical differences: skip the last row.
    diff_v = torch.abs(delta[:, :, 1:, :] - delta[:, :, :-1, :])
    
    # Crop to a common area.
    # diff_h has shape (N, C, H, W-1) and diff_v has shape (N, C, H-1, W).
    # We take the common region of shape (N, C, H-1, W-1).
    diff_h = diff_h[:, :, :-1, :]
    diff_v = diff_v[:, :, :, :-1]
    
    # Compute loss: (|diff_h| + |diff_v|)^2 averaged over all values.
    loss = torch.mean((diff_h + diff_v) ** 2)
    return loss

# def spatial_consistency_loss(enhanced, original):
#     """
#     Spatial consistency loss (Lspa).
#     """
#     dx_enhanced = enhanced[:, :, 1:, :] - enhanced[:, :, :-1, :]
#     dy_enhanced = enhanced[:, :, :, 1:] - enhanced[:, :, :, :-1]

#     dx_original = original[:, :, 1:, :] - original[:, :, :-1, :]
#     dy_original = original[:, :, :, 1:] - original[:, :, :, :-1]

#     loss = (
#         (dx_enhanced[:, :, :, :-1] - dx_original[:, :, :, :-1]) ** 2 + \
#      (dy_enhanced[:, :, :-1, :] - dy_original[:, :, :-1, :]) ** 2
#         ).mean()
#     return loss

# # Combine the losses
# def total_loss(enhanced, original, curve_params, weights):
#     """
#     Total loss combining all the individual loss functions.
#     """
#     Luac = underwater_color_adaptive_correction_loss(enhanced, original)
#     Lexp = exposure_control_loss(enhanced)
#     Ltvd = illumination_smoothness_loss(curve_params)
#     Lspa = spatial_consistency_loss(enhanced, original)

#     return (
#         weights['Luac'] * Luac +
#         weights['Lexp'] * Lexp +
#         weights['Ltvd'] * Ltvd +
#         weights['Lspa'] * Lspa
#     )

def CCICalculation(image, tolerance=50):
    """
    Compute the Contrast Code Image (CCI) for the given image.
    
    Args:
        image (torch.Tensor): Input image tensor of shape (B, C, H, W).
        tolerance (int): Tolerance parameter (0-100) to prioritize larger patches.
    
    Returns:
        torch.Tensor: CCI tensor of shape (B, H, W) with values in [0, 6].
    """
    B, C, H, W = image.shape
    device = image.device
    
    # Convert to grayscale using BT.601 coefficients
    gray = 0.299 * image[:, 0, ...] + 0.587 * image[:, 1, ...] + 0.114 * image[:, 2, ...]  # (B, H, W)
    gray = gray.unsqueeze(1)  # (B, 1, H, W) for compatibility with avg_pool2d
    
    psize2 = [15, 13, 11, 9, 7, 5, 3]
    num_psize = len(psize2)
    
    std_maps = []
    for p in psize2:
        kernel_size = p
        padding = p // 2
        # Compute mean and mean of squares
        mean = F.avg_pool2d(gray, kernel_size=kernel_size, stride=1, padding=padding, count_include_pad=False)
        mean_sq = F.avg_pool2d(gray.pow(2), kernel_size=kernel_size, stride=1, padding=padding, count_include_pad=False)
        std = (mean_sq - mean.pow(2)).sqrt()
        std_maps.append(std.squeeze(1))  # Remove channel dim: (B, H, W)
    
    # Stack std_maps along dim=1: (B, 7, H, W)
    std_maps = torch.stack(std_maps, dim=1)
    
    # Compute tolerance factors
    tol = 1.0 - (tolerance / 100.0)
    exponents = torch.arange(num_psize-1, -1, -1, device=device).float()  # [6,5,...,0]
    tolerance_factors = tol ** exponents
    tolerance_factors = tolerance_factors.view(1, num_psize, 1, 1)  # Broadcastable shape
    
    # Adjust std maps with tolerance
    adjusted_std = std_maps * tolerance_factors
    
    # Find the index of the minimum adjusted std for each pixel
    CCI = torch.argmin(adjusted_std, dim=1)  # (B, H, W)
    
    return CCI

def exposure_control_loss(enhanced, E=0.43, tolerance=50):
    """
    Exposure control loss (Lexp) using Contrast Code Image (CCI) to determine patch sizes.
    
    Args:
        enhanced (torch.Tensor): Enhanced image tensor of shape (B, C, H, W).
        E (float): Target exposure value.
        tolerance (int): Tolerance parameter for CCI calculation.
    
    Returns:
        torch.Tensor: Computed exposure loss.
    """
    B, C, H, W = enhanced.shape
    device = enhanced.device
    psize2 = [15, 13, 11, 9, 7, 5, 3]
    
    # Compute CCI
    CCI = CCICalculation(enhanced, tolerance)  # (B, H, W)
    
    # Compute mean for each patch size across all channels
    enhanced_means = []
    for p in psize2:
        kernel_size = p
        padding = p // 2
        mean = F.avg_pool2d(enhanced, kernel_size=kernel_size, stride=1, padding=padding, count_include_pad=False)
        enhanced_means.append(mean)
    # Stack along dim=2: (B, C, 7, H, W)
    enhanced_means = torch.stack(enhanced_means, dim=2)
    
    # Gather the selected means based on CCI
    CCI_expanded = CCI.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, H, W)
    CCI_expanded = CCI_expanded.expand(-1, C, -1, -1, -1)  # (B, C, 1, H, W)
    selected_means = torch.gather(enhanced_means, dim=2, index=CCI_expanded).squeeze(2)  # (B, C, H, W)
    
    # Compute exposure loss
    loss = ((selected_means - E) ** 2).mean()
    
    return loss

def total_loss(enhanced, target, curve_params, weights):
    """
    Total loss combining only the exposure control loss.
    
    Args:
        enhanced (torch.Tensor): Enhanced image tensor.
        weights (dict): Dictionary containing the weight for exposure loss.
    
    Returns:
        torch.Tensor: Total loss.
    """
    Lexp = exposure_control_loss(enhanced)
    return weights['Lexp'] * Lexp + weights['Ltvd'] * illumination_smoothness_loss(curve_params)

# 4. Define the Custom Dataset Class
class UnderwaterDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = os.listdir(root_dir)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_name).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, image  # Input and target images are the same for zero-reference tasks

# Define the directory to save model checkpoints
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

# Save the model checkpoint
def save_checkpoint(model, epoch, save_dir):
    save_path = os.path.join(save_dir, f"uae_net_epoch_{epoch + 1}.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Model saved at {save_path}")

# Modified training loop with model saving
def train_uaenet(model, dataloader, optimizer, num_epochs, device, weights, save_dir):
    model.train()  # Set model to training mode
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (input_images, target_images) in enumerate(dataloader):
            # Move the input and target images to the device (GPU or CPU)
            input_images = input_images.to(device)
            target_images = target_images.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass: Generate curve parameter maps using UAE-Net
            curve_params = model(input_images)

            # Enhance the input images iteratively
            enhanced_images = enhance_image(input_images, model)

            # Calculate the loss
            loss = total_loss(enhanced_images, target_images, curve_params, weights)

            # Backward pass and optimization step
            loss.backward()
            optimizer.step()

            # Print statistics
            running_loss += loss.item()
            if i % 5 == 9:  # Print every 10 batches
                print(f"[Epoch {epoch + 1}, Batch {i + 1}] Loss: {running_loss / 10:.4f}")
                running_loss = 0.0

        if epoch % 20 in (0, num_epochs-1):  # Save model checkpoint every 10 epochs
        # Save model at the end of each epoch
            save_checkpoint(model, epoch, save_dir)

        print(f"Epoch [{epoch + 1}/{num_epochs}] completed.")

    print("Training finished.")

# Define hyperparameters
num_epochs = 10
learning_rate = 1e-4
batch_size = 2
weights = {
    'Luac': 1.0,    # Weight for underwater color adaptive correction loss
    'Lexp': 1.0,    # Weight for exposure control loss
    'Ltvd': 0.8,    # Weight for illumination smoothness loss
    'Lspa': 0.3     # Weight for spatial consistency loss
}

# Initialize the model
model = UAENet().to(device)

# Define the optimizer
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Define data transformations (resize and normalize images)
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# Assuming you have a dataset like UnderwaterDataset, we need to split the indices.
dataset = UnderwaterDataset(root_dir=ROOT_DIR, transform=transform)

TEST_SIZE = 0.25

# Get indices for train-test split
train_indices, test_indices = train_test_split(list(range(len(dataset))), test_size=TEST_SIZE, random_state=42)

# Create subsets for training and testing
train_dataset = Subset(dataset, train_indices)
test_dataset = Subset(dataset, test_indices)

# Create DataLoaders for train and test sets
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

save_folder = 'test_set_images'
os.makedirs(save_folder, exist_ok=True)

# Iterate through the test_loader
for batch_idx, (images, labels) in enumerate(test_loader):
    # Save each image in the batch
    for i in range(images.size(0)):
        # Construct the filename
        filename = f'image_{batch_idx * batch_size + i}.png'
        filepath = os.path.join(save_folder, filename)
        
        # Save the image
        save_image(images[i], filepath)

print(f"Test set images saved in {save_folder}")

# Now you can use train_loader for training and test_loader for validation or testing.
train_uaenet(model, train_loader, optimizer, num_epochs, device, weights, SAVE_DIR)

torch.cuda.empty_cache()

# import torch
# from PIL import Image
# from torchvision import transforms
# import matplotlib.pyplot as plt

# # Define the function to load the saved model
# def load_model(model, checkpoint_path, device):
#     model.load_state_dict(torch.load(checkpoint_path, map_location=device))
#     model.to(device)
#     model.eval()  # Set the model to evaluation mode
#     print(f"Model loaded from {checkpoint_path}")
#     return model

# # Function to preprocess input image (resize, normalize, convert to tensor)
# def preprocess_image(image_path):
#     transform = transforms.Compose([
#         transforms.Resize((256, 256)),
#         transforms.ToTensor(),  # Convert to tensor
#     ])
#     image = Image.open(image_path).convert('RGB')
#     image = transform(image).unsqueeze(0)  # Add batch dimension
#     return image

# # Function to post-process the output tensor and convert it back to an image
# def postprocess_image(tensor):
#     image = tensor.squeeze(0)  # Remove batch dimension
#     image = image.detach().cpu().numpy().transpose(1, 2, 0)  # Convert to numpy and reshape
#     image = (image - image.min()) / (image.max() - image.min())  # Normalize to [0, 1]
#     return image

# # Inference function to enhance an underwater image using the trained model
# def run_inference(model, image_path, device):
#     # Preprocess the image
#     input_image = preprocess_image(image_path).to(device)

#     # Perform inference using the trained model
#     with torch.no_grad():
#         enhanced_image = enhance_image(input_image, model, iterations=17)

#     # Post-process the output image and convert it to a format that can be displayed
#     enhanced_image_np = postprocess_image(enhanced_image)

#     # Display the original and enhanced images side by side
#     original_image = Image.open(image_path)
#     original_image = original_image.resize((256, 256))
#     plt.figure(figsize=(10, 5))
#     plt.subplot(1, 2, 1)
#     plt.imshow(original_image)
#     plt.title('Original Image')

#     plt.subplot(1, 2, 2)
#     plt.imshow(enhanced_image_np)
#     plt.title('Enhanced Image')
#     plt.show()

# # Example of how to use the above functions for inference
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = UAENet()

# # Load the trained model from the checkpoint
# checkpoint_path = os.path.join(SAVE_DIR, "uae_net_epoch_10.pth")
# model = load_model(model, checkpoint_path, device)

# # Path to the underwater image you want to enhance
# image_path = "data/oceandark/images/1.jpg"

# # Run inference on the image and display the results
# run_inference(model, image_path, device)


# image_path = "data/oceandark/images/14.jpg"
# run_inference(model, image_path, device)

# image_path = "data/oceandark/images/163.jpg"
# run_inference(model, image_path, device)

