"""
This code downloads vikhyatk/moondream2 model and it's moondream/starmie-v1 tokenizer from the hub.
"""
from huggingface_hub import snapshot_download

snapshot_download(repo_id="vikhyatk/moondream2", repo_type="model")

# tokenizer
snapshot_download(repo_id="moondream/starmie-v1")

print("Model is ready to use!")
