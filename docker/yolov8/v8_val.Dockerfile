FROM ultralytics/ultralytics:latest

SHELL ["/bin/bash", "-c"]

ENV DEBIAN_FRONTEND noninteractive

RUN apt update \
&& apt install -y \
ffmpeg \ 
libsm6 \
libxext6 \
&& rm -rf /var/lib/apt/lists/*

RUN pip3 install numpyencoder>=0.3.0
