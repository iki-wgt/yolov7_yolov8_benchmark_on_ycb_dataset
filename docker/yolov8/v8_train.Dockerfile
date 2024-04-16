FROM nvcr.io/nvidia/pytorch:22.05-py3

ENV DEBIAN_FRONTEND noninteractive

# Install usefull tools
RUN apt-get update && \
    apt-get install -y \
    screen \
    git \
    vim \
    python3-pip \
    htop \
    ffmpeg \
    libsm6 \
    libxext6

WORKDIR /root
COPY src/yolov8_training /root/yolov8_training

WORKDIR /root/yolov8_training
RUN pip3 install -r requirements.txt

# Error when installing in requirements.txt
RUN pip3 install opencv-python==4.5.5.64