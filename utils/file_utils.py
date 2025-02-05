import os
from PIL import Image
import gradio as gr

def refresh_file_list(directory):
    """
    List all image files in the specified directory.

    Parameters:
        directory (str): Path to the directory.

    Returns:
        list: A list of image file names.
    """
    if not os.path.exists(directory):
        return []
    files = [f for f in os.listdir(directory)
             if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    return files

def load_selected_image(directory, filename):
    """
    Load the selected image from the specified directory.

    Parameters:
        directory (str): Path to the directory.
        filename (str): Image file name.

    Returns:
        PIL.Image: The loaded image in RGB format.
    """
    path = os.path.join(directory, filename)
    image = Image.open(path).convert("RGB")
    return image

def refresh_list(directory):
    """
    Refresh the dropdown list with image files in the specified directory.

    Parameters:
        directory (str): Path to the directory.

    Returns:
        gr.update: Updated dropdown options with the first image selected (if available).
    """
    files = refresh_file_list(directory)
    default_val = files[0] if files else None
    return gr.update(choices=files, value=default_val)

def on_select_image(directory, filename):
    """
    Load the image when a file is selected from the dropdown.

    Parameters:
        directory (str): Path to the directory.
        filename (str): Selected image file name.

    Returns:
        PIL.Image: The loaded image.
    """
    if not filename:
        return None
    return load_selected_image(directory, filename)
