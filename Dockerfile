# This dockerfile is will build an image that is set to table, figure, caption detection from given images and content recognition

# To build a docker image from dockerfile:
# docker build -f <dockerfile filename> . --rm -t <repo_name:tag>

FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y python3-pip \
    apt-utils \
    git \
    vim \
    wget \
    poppler-utils \
    uvicorn \
    tesseract-ocr

# for CV opencv package
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 -y

RUN pip install --no-cache-dir torch==2.3.0 torchvision torchtext

RUN pip install --no-cache-dir opencv-python\
    easyocr \
    fastapi \
    pdf2image \
    pillow \
    pytesseract \
    python-multipart \
    transformers==4.36.1 \
    jsonlines \
    tokenizers \
    einops \
    pandas \
    bs4

# detectron2
RUN python3 -m pip install 'git+https://github.com/facebookresearch/detectron2.git'

RUN apt-get clean && apt-get autoremove && rm -rf /var/lib/apt/lists/* /tmp/* ~/*

RUN apt-get update && apt-get install -y uvicorn
RUN mkdir tablefigextraction

COPY tablefigextraction ./tablefigextraction

#ENTRYPOINT ["/bin/bash", "-l", "-c"](bash)a
COPY ./start.sh ./tablefigextraction
WORKDIR /tablefigextraction
