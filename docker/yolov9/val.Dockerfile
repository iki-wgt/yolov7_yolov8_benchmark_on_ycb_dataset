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
    libxext6 \
    feh \
    numpyencoder

WORKDIR /root/yolov9
RUN git clone https://github.com/WongKinYiu/yolov9.git /root/yolov9
RUN wget https://github.com/WongKinYiu/yolov9/releases/download/v0.1/yolov9-c-converted.pt

RUN pip install -r requirements.txt
RUN pip3 install opencv-python==4.5.5.64