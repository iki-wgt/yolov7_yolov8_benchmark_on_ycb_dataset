FROM nvcr.io/nvidia/pytorch:21.08-py3

SHELL ["/bin/bash", "-c"]

ENV DEBIAN_FRONTEND noninteractive

RUN apt update \
&& apt install -y \
ffmpeg \ 
libsm6 \
libxext6 \
&& rm -rf /var/lib/apt/lists/*

RUN pip3 install opencv-python 

WORKDIR /root
COPY src/validation/requirements.txt /root/
RUN pip3 install -r requirements.txt 