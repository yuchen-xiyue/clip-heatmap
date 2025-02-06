import gradio as gr
from utils.file_utils import refresh_list, on_select_image
from utils.processing import (
    predict_and_overlay,
    update_overlay_from_slider,
    save_numpy_heatmap
)

def build_interface():
    """
    Build the Gradio interface for the CLIP heatmap overlay application.

    Returns:
        gr.Blocks: The Gradio Blocks interface.
    """
    with gr.Blocks() as demo:
        gr.Markdown("## CLIP-Based Heatmap Overlay Application")
        with gr.Row():
            
            with gr.Column(scale=0.5):
                dir_text = gr.Textbox(label="Input Image Directory", value="input")
                refresh_btn = gr.Button("Refresh Image List")
                image_list = gr.Dropdown(label="Select Image", choices=[], interactive=True)
                text_prompt = gr.Textbox(label="Text Prompt", placeholder="Enter text prompt here")
                patch_size_slider = gr.Slider(label="Patch Size", minimum=4, maximum=256, step=1, value=32)
                stride_slider = gr.Slider(label="Sliding Window Stride", minimum=1, maximum=256, step=1, value=16)
                predict_btn = gr.Button("Predict")
                status_text = gr.Textbox(label="Status", interactive=False)
            
            with gr.Column():
                with gr.Row():
                    input_image = gr.Image(label="Input Image", type="pil")
                    overlay_image = gr.Image(label="Overlay Image", type="pil")
                
                alpha_slider = gr.Slider(label="Heatmap Transparency", minimum=0, maximum=1, value=1, step=0.01)
                save_btn = gr.Button("Save Numpy Heatmap")

        
        heatmap_state = gr.State(None)

        refresh_btn.click(fn=refresh_list, inputs=dir_text, outputs=image_list)
        image_list.change(fn=on_select_image, inputs=[dir_text, image_list], outputs=input_image)
        
        predict_btn.click(
            fn=predict_and_overlay,
            inputs=[input_image, text_prompt, alpha_slider, patch_size_slider, stride_slider],
            outputs=[overlay_image, heatmap_state]
        )
        
        # The slider updates the overlay image in real time based on the new alpha value.
        alpha_slider.change(
            fn=update_overlay_from_slider,
            inputs=[input_image, heatmap_state, alpha_slider],
            outputs=overlay_image
        )
        
        # The "Save Numpy Heatmap" button saves the pure heatmap as a NumPy array.
        save_btn.click(
            fn=save_numpy_heatmap,
            inputs=heatmap_state,
            outputs=status_text
        )
    return demo

if __name__ == "__main__":
    interface = build_interface()
    interface.launch()
