FROM python:3.11-slim

# update the apt packages
RUN apt-get update -y
RUN apt-get install -y python3-pip python3-dev gcc g++ make

# install moondream dependencies
RUN pip install "transformers>=4.51.1" "torch>=2.7.0" "accelerate>=1.10.0" "Pillow>=11.0.0"

WORKDIR /moondream
COPY . .

# install moondream models as cache to be used when needed
RUN python install_model.py && rm install_model.py

# disable model installation from internet (just to make sure)
ENV HF_HUB_OFFLINE=1

RUN chmod +x main.py

