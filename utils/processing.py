import os
import math
import datetime
import torch
import clip
import numpy as np
from PIL import Image
import matplotlib

# Set device and load the CLIP model and its preprocessing function.
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

def predict_heatmap(image, text_prompt):
    """
    Given an input image and a text prompt, generate a heatmap using CLIP.

    The image is processed using a sliding window approach, and the cosine similarity
    is computed for each patch. The resulting low-resolution heatmap is then upsampled 
    to the original image size.

    Parameters:
        image (PIL.Image): Input image.
        text_prompt (str): Text prompt.

    Returns:
        PIL.Image: The generated heatmap image.
    """
    if image is None or text_prompt.strip() == "":
        return None

    # Convert numpy array to PIL Image if necessary.
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image.astype('uint8'), 'RGB')
    width, height = image.size

    # Encode the text prompt.
    text_tokens = clip.tokenize([text_prompt]).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)

    # Define the sliding window patch size.
    patch_size = 32
    n_cols = math.ceil(width / patch_size)
    n_rows = math.ceil(height / patch_size)
    patches = []
    for i in range(n_rows):
        for j in range(n_cols):
            left = j * patch_size
            upper = i * patch_size
            right = min((j + 1) * patch_size, width)
            lower = min((i + 1) * patch_size, height)
            patch = image.crop((left, upper, right, lower))
            patch_preprocessed = preprocess(patch).unsqueeze(0)
            patches.append(patch_preprocessed)

    if len(patches) == 0:
        return None

    patches_tensor = torch.cat(patches, dim=0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(patches_tensor)
        image_features /= image_features.norm(dim=-1, keepdim=True)

    # Compute cosine similarity for each patch with the text prompt.
    similarities = torch.matmul(image_features, text_features.T).squeeze(1)
    sim_grid = similarities.cpu().numpy().reshape(n_rows, n_cols)

    # Normalize the similarity grid to [0, 1].
    sim_min, sim_max = sim_grid.min(), sim_grid.max()
    if sim_max - sim_min > 1e-5:
        sim_norm = (sim_grid - sim_min) / (sim_max - sim_min)
    else:
        sim_norm = np.zeros_like(sim_grid)

    # Convert the low-resolution similarity grid to a grayscale image.
    sim_img = (sim_norm * 255).astype(np.uint8)
    sim_img_pil = Image.fromarray(sim_img, mode='L')
    # Upsample to the original image size.
    sim_img_resized = sim_img_pil.resize((width, height), resample=Image.BILINEAR)
    # (Optional) You can get the underlying normalized similarity as a NumPy array:
    sim_resized_np = np.array(sim_img_resized).astype(np.float32) / 255.0

    # Generate a color heatmap using the 'jet' colormap.
    cmap = matplotlib.colormaps.get_cmap("jet")
    heatmap_rgba = cmap(sim_resized_np)
    heatmap_rgb = np.uint8(heatmap_rgba[:, :, :3] * 255)
    heatmap_pil = Image.fromarray(heatmap_rgb)
    return heatmap_pil

def overlay_images(original, heatmap, alpha):
    """
    Overlay the heatmap on the original image with the given transparency.

    Parameters:
        original (PIL.Image): The original image.
        heatmap (PIL.Image): The generated heatmap.
        alpha (float): The transparency level for the heatmap overlay (0 to 1).

    Returns:
        PIL.Image: The blended image.
    """
    if original is None or heatmap is None:
        return None
    original_rgba = original.convert("RGBA")
    heatmap_rgba = heatmap.convert("RGBA")
    blended = Image.blend(original_rgba, heatmap_rgba, alpha)
    return blended

def predict_and_overlay(image, text_prompt, alpha):
    """
    Generate the heatmap and overlay it on the original image with the given alpha.

    Parameters:
        image (PIL.Image): The original image.
        text_prompt (str): Text prompt.
        alpha (float): Transparency value for overlay.

    Returns:
        tuple: (blended_image, pure_heatmap)
            blended_image (PIL.Image): The image with heatmap overlay.
            pure_heatmap (PIL.Image): The generated heatmap (stored for later slider adjustments).
    """
    if image is None or text_prompt.strip() == "":
        return None, None
    pure_heatmap = predict_heatmap(image, text_prompt)
    overlay = overlay_images(image, pure_heatmap, alpha)
    return overlay, pure_heatmap

def update_overlay_from_slider(image, heatmap, alpha):
    """
    Update the overlay image based on the new alpha value from the slider.

    Parameters:
        image (PIL.Image): The original image.
        heatmap (PIL.Image): The saved pure heatmap.
        alpha (float): New transparency value.

    Returns:
        PIL.Image: The updated blended image.
    """
    if image is None or heatmap is None:
        return None
    return overlay_images(image, heatmap, alpha)

def save_numpy_heatmap(heatmap):
    """
    Save the given heatmap (PIL Image) as a NumPy array (.npy file).

    The file is saved in the "saved_heatmaps" directory with a timestamp in its filename.

    Parameters:
        heatmap (PIL.Image): The heatmap to be saved.

    Returns:
        str: A status message indicating where the file was saved.
    """
    if heatmap is None:
        return "No heatmap available to save."
    # Create the output directory if it doesn't exist.
    output_dir = "saved_heatmaps"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # Convert the heatmap to a NumPy array.
    heatmap_np = np.array(heatmap)
    # Create a filename with a timestamp.
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"heatmap_{timestamp}.npy"
    filepath = os.path.join(output_dir, filename)
    # Save the NumPy array to disk.
    np.save(filepath, heatmap_np)
    return f"Heatmap saved to {filepath}"
