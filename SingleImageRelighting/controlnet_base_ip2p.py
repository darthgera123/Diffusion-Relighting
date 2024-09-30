from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from transformers import CLIPTokenizer, CLIPTextModel
import torch

from transformers import DPTForDepthEstimation, DPTFeatureExtractor
from PIL import Image
import torch
import numpy as np
import matplotlib.pyplot as plt


cache_dir = '/scratch/inf0/user/pgera/FlashingLights/SingleRelight/cache/'
# Load the MiDaS depth estimation model
model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large",cache_dir=cache_dir)
feature_extractor = DPTFeatureExtractor.from_pretrained("Intel/dpt-large",cache_dir=cache_dir)

# Set device (use CUDA if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Load the base image and prepare it for the model
base_image = Image.open("diffuse.png")  # Your input image
inputs = feature_extractor(images=base_image, return_tensors="pt").to(device)

# Forward pass through the model to get depth map
with torch.no_grad():
    outputs = model(**inputs)
    predicted_depth = outputs.predicted_depth

# Resize the depth map back to the original image size
depth_map = predicted_depth.squeeze().cpu().numpy()
depth_map_resized = np.resize(depth_map, base_image.size[::-1])

# Normalize depth map to [0, 255] for saving as an image
depth_map_normalized = (depth_map_resized - depth_map_resized.min()) / (depth_map_resized.max() - depth_map_resized.min())
depth_map_normalized = (depth_map_normalized * 255).astype(np.uint8)

# Convert to PIL image for saving or visualization
depth_image_pil = Image.fromarray(depth_map_normalized)


# Load the ControlNet model (you can choose a specific ControlNet variant, e.g., depth, canny, etc.)
controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/control_v11f1p_sd15_depth",  # Or another ControlNet model variant
    torch_dtype=torch.float16, cache_dir=cache_dir
)

# Load the InstructPix2Pix model as the base model
instruct_pix2pix_model = StableDiffusionControlNetPipeline.from_pretrained(
    "timbrooks/instruct-pix2pix",  # InstructPix2Pix model checkpoint
    controlnet=controlnet,  # Attach ControlNet
    torch_dtype=torch.float16,cache_dir=cache_dir
)

# Optional: Offload the model to CPU when not in use to save memory
instruct_pix2pix_model.enable_model_cpu_offload()

# Optional: Use xFormers memory efficient attention for faster processing (if installed)
# instruct_pix2pix_model.enable_xformers_memory_efficient_attention()

# Set device to CUDA if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
instruct_pix2pix_model.to(device)

# Now you can generate images with the pipeline
# Example usage with a base image and conditioning image

from PIL import Image
base_image = Image.open("diffuse.png")  # Base image (input)
control_image = depth_image_pil  # Conditioning image for ControlNet (e.g., depth map)

# Text prompt for the InstructPix2Pix guidance
text_prompt = "Change the lighting of the scene"

# Run the pipeline to generate the output image
output_images = instruct_pix2pix_model(
    prompt=text_prompt,
    image=base_image,  # Input image for InstructPix2Pix
    control_image=control_image,  # Conditioning image for ControlNet
    num_inference_steps=50,  # Number of diffusion steps
    guidance_scale=7.5,  # Adjust guidance strength for InstructPix2Pix
).images

# Save or display the output
output_images[0].save("output_image.png")