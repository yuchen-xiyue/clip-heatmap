# CLIP Heatmap Visualization Tool
This project is a CLIP-based heatmap overlay application that uses a text prompt to generate a heatmap for an input image and overlays the heatmap on the original image with adjustable transparency. The application is built with [Gradio](https://gradio.app/) for the user interface and leverages OpenAI's [CLIP](https://github.com/openai/CLIP) model to compute image-text similarity.

## Dependencies
```
gradio
torch
clip
Pillow
matplotlib
numpy
```

## Run
```
python app.py
```