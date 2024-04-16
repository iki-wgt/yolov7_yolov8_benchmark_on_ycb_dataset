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
COPY src/yolov7_training yolov7_training

WORKDIR /root/yolov7_training 
RUN pip3 install -r requirements.txt
RUN pip3 install -r requirements_gpu.txt
RUN wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7x.pt