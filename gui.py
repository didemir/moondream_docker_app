import os
import sys
import gradio as gr
import torch
import transformers
transformers.utils.logging.set_verbosity_error()
from transformers import AutoModelForCausalLM, PreTrainedModel
from PIL import Image

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


def handle_upload(file_path):
  if file_path:
    return file_path, f"## Loaded image: {os.path.basename(file_path)}", file_path
  return None, "## No image selected", None


def process_image(file_obj, prompt, temp, maxtok, topp):
  """
  Callback function for the Gradio interface.
  """
  if file_obj is None:
    return "Error: No image loaded."
  
  if not prompt:
    return "Error: Please enter a text prompt."

  try:
    global model
    image = Image.open(file_obj).convert("RGB")
    settings = {"temperature": float(temp), "max_tokens": int(maxtok), "top_p": float(topp)}
    result = model.query(image, prompt, settings)
    return result["answer"]
  except Exception as e:
    return f"Inference Error: {str(e)}"


load_model()
with gr.Blocks(title="Moondream2 Interface") as demo:
  curr_state = gr.State(value=None)
  with gr.Row():
    with gr.Column(scale=1):
      upload_btn = gr.UploadButton(label="Browse Image", file_types=["image"], type="filepath")
      status_text = gr.Markdown("## No image selected.\nClick 'Browse Image' to load a file.")
      with gr.Accordion("Advanced Generation Settings", open=False):
        temp_slider = gr.Slider(minimum=0.1, maximum=1.0, value=0.5, step=0.1, label="Temperature")
        max_tok_slider = gr.Slider(minimum=10, maximum=512, value=126, step=1, label="Max Tokens")
        top_p_slider = gr.Slider(minimum=0.1, maximum=1.0, value=0.3, step=0.1, label="Top P")
    with gr.Column(scale=1):
      image_prev = gr.Image(label="Preview", interactive=False, height=300)
  user_input = gr.Textbox(label="Your Question", placeholder="Describe this image or ask a question...")
  submit_btn = gr.Button("Send to Model")
  output_text = gr.TextArea(label="Model Response", interactive=False)

  upload_btn.upload(
    fn=handle_upload,
    inputs=upload_btn, 
    outputs=[curr_state, status_text, image_prev]
  )

  submit_btn.click(
    fn=process_image,
    inputs=[curr_state, user_input, temp_slider, max_tok_slider, top_p_slider],
    outputs=output_text
  )

if __name__ == "__main__":
  demo.launch(server_name="0.0.0.0", server_port=7860)



