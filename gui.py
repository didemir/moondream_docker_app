import os
import sys
from typing import Optional
import gradio as gr
import torch
import transformers
from transformers import AutoModelForCausalLM, PreTrainedModel
from PIL import Image

# Suppress transformer library warnings for cleaner logs.
transformers.utils.logging.set_verbosity_error()
model = None

def load_model() -> PreTrainedModel:
  """
  Loads vikhyatk/moondream2 model into global scope.

  Returns:
  --------
    PreTrainedModel: The loaded Moondream2 Model

  """
  global model
  if model is None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model to {device}...")
    try:
      model = AutoModelForCausalLM.from_pretrained(
        "vikhyatk/moondream2",
        local_files_only=True,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        device_map=device,
      )
      print("Model is ready!")
      return model
    except Exception as e:
      sys.exit(f"Error loading model: {e}")


def handle_upload(file_path: Optional[str]) -> tuple(Optional[str], str, Optional[str]):
  """
    Processes uploaded image file and prepares it.
    
    Args:
        file_path: Path to the uploaded image file
        
    Returns:
        Tuple containing:
        - State value (file path)
        - Status message (markdown formatted)
        - Preview image path

  """
  if file_path:
    return file_path, f"## Loaded {os.path.basename(file_path)}", file_path
  return None, "## No image selected.\nClick 'Browse Image' to load a file.", None


def query_moondream(file_obj: Optional[str], prompt: str, temp: float, maxtok: int, topp: float) -> str:
  """
    Queries the Moondream2 model with selected image, entered text prompt, and settings.
    
    Args:
        file_obj: Path to the image file
        prompt: Prompt about the image
        temp: Temperature parameter
        max_tokens: Maximum number of tokens to generate
        top_p: Top-p sampling parameter
        
    Returns:
        Model's answer, or error message
  """

  if file_obj is None:
    return "Error: No image loaded."
  
  if not prompt or not prompt.strip():
    return "Error: Please enter a text prompt."

  try:
    global model
    image = Image.open(file_obj).convert("RGB")
    settings = {"temperature": float(temp), "max_tokens": int(maxtok), "top_p": float(topp)}
    print(f"Query settigns: {settings}")
    result = model.query(image, prompt, settings)
    return result["answer"]
  except Exception as e:
    return f"Inference Error: {str(e)}"


load_model()
with gr.Blocks(title="Moondream2 Inference", theme=gr.themes.Soft()) as demo:
  curr_state = gr.State(value=None)
  with gr.Row():
    with gr.Column(scale=1):
      # Image upload button that only accepts image files.
      upload_btn = gr.UploadButton(label="Browse Image", file_types=["image"], type="filepath")
      # An information text prompts user to select an image. If already selected, prints the file name.
      status_text = gr.Markdown("## No image selected.\nClick 'Browse Image' to load a file.")
      # Model settings, comes in closed form.
      with gr.Accordion("Advanced Generation Settings", open=False):
        temp_slider = gr.Slider(minimum=0.1, maximum=1.0, value=0.5, step=0.1, label="Temperature")
        max_tok_slider = gr.Slider(minimum=10, maximum=512, value=126, step=1, label="Max Tokens")
        top_p_slider = gr.Slider(minimum=0.1, maximum=1.0, value=0.3, step=0.1, label="Top P")
    # Preview of selected image
    with gr.Column(scale=1):
      image_prev = gr.Image(label="Preview", interactive=False, height=300)
  # User input area
  user_input = gr.Textbox(label="Your Question", placeholder="Describe this image or ask a question...", lines=2)
  # A button to send request to model
  with gr.Row():
    submit_btn = gr.Button("Send to Model", variant="primary")
    clear_btn = gr.Button("Clear", variant="secondary")
  # An output area showing model's answer. Reasoning is not shown.
  output_text = gr.TextArea(label="Model Response", interactive=False, lines=5)


  gr.Examples(
    examples=[
      ["Describe the image."],
      ["What is the main subject?"]
    ],
    inputs=user_input,
    label="Example prompts"
  )

  # The upload button triggers handle_upload function
  upload_btn.upload(
    fn=handle_upload,
    inputs=upload_btn, 
    outputs=[curr_state, status_text, image_prev]
  )

  # The submit button triggers query funtion
  submit_btn.click(
    fn=query_model,
    inputs=[curr_state, user_input, temp_slider, max_tok_slider, top_p_slider],
    outputs=output_text
  )

  clear_btn.click(
    fn=lambda: (None, None, "## No image selected\nClick 'Browse Image' to load a file.", "", ""),
    outputs=[curr_state, image_prev, status_text, user_input, output_text]
  )

if __name__ == "__main__":
  demo.launch(server_name="0.0.0.0", server_port=7860)



