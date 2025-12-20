#!/usr/bin/env python3

"""
This Python script is a command-line interface (CLI) application for performing vision–language tasks using the Moondream2 model.
It enables users to supply an image along with a natural language question,
and then generates an AI-driven description or answer based on the visual content of the image.
"""

import sys
import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
import argparse
from pathlib import Path

from transformers import AutoModelForCausalLM, PreTrainedModel
from PIL import Image
import torch


def load_moondream() -> PreTrainedModel:
  """
  Loads vikhyatk/moondream2 model to detected device.

  Returns:
  --------
    PreTrainedModel: The loaded Moondream2 Model

  """
  try:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"loading model to {device} ...")

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


def start_chat(image_pth: Path, settings: dict) -> None:
  """
  Main function to process image with Moondream model.
  Allows continuous conversation about the same image until user exits.

  Args:
  -----
    image_pth: Path to the input image
    settings: Model generation settings
  """
  model = load_moondream()

  try:
    image = Image.open(image_pth)
  except Exception as e:
    sys.exit(f"Error opening image: {e}")

  print("="*60)
  print("\tYou can ask questions about the image.\n\tType 'exit' to stop conversation.")
  print("="*60)
  # Requirement 2: Prompt the user to input a question for the model to answer
  while True:
    prompt = input("User: ").strip()
    if prompt.lower() == "exit":
      print("Exiting ...")
      break
    # Requirement 3: Print answer to the user-provided question about the input image
    answer = model.query(image, prompt, settings) 
    print(f"\nModel: {answer['answer']}\n")

def main():
  # Requirement 1: Take input image path (png) as a CLI argument
  parser = argparse.ArgumentParser()
  parser.add_argument("--image", "-i", type=Path, help="Image path.")
  parser.add_argument("--temp", "-t", type=float, default=0.5, help="Model temperature. Must be float. Default value is 0.5.")
  parser.add_argument("--max_token", "-m", type=int, default=126, help="Maximum tokens.Must be integer. Default is 126.")
  parser.add_argument("--topp", "-p", type=float, default=0.3, help="Top P value. Must e float. Default is 0.3.")
  args = parser.parse_args()

  # if the cli tag has not been used to take image, prompts user to enter path
  image_path = Path(args.image or input("Enter a valid input image path (you can test with husky.png path): "))
  if not image_path.exists():
    sys.exit(f"Error: The entered image path does not exist: {image_path}")
  # Because of the specific addressing to take input image path (png), image format restriction included
  if image_path.suffix != ".png":
    sys.exit(f"Error: The entered image path is not valid. {image_path}")
  settings = {"temperature": args.temp, "max_tokens": args.max_token, "top_p": args.topp}
  start_chat(image_path, settings) 


if __name__ == "__main__":
  main()


