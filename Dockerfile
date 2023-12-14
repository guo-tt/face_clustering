from python:3.8

USER root

RUN apt-get update && apt-get install -y \
    librdkafka1 librdkafka-dev \
    libpq-dev \
    gcc \
    vim \
    libzbar0 \
    ffmpeg libsm6 libxext6 \
    libgl1

RUN pip install fastapi[all]==0.92.0

RUN pip install \
    pandas==1.3.4 \
    pyarrow==7.0.0 \
    numpy --upgrade \
    httpx==0.24.0 \
    typing==3.7.4.3 \
    python-jose==3.0.1 \
    Pillow==9.5.0 \
    matplotlib==3.7.1 \
    torch==2.0.0 \
    torchvision==0.15.0 \
    ultralytics==8.0.114 \
    scipy==1.10.1 \
    opencv-contrib-python==4.6.0.66 \
    opencv-python==4.6.0.66 \
    opencv-python-headless==4.8.0.74 \
    numpy==1.23.5 \
    face_recognition \
    jupyterlab \
    transformers \
    mmcv-full==1.5.0 \
    mmsegmentation==0.27.0 \
    mmcv==0.6.2 \
    mmdet==2.3.0 \
    scikit-learn \
    infomap \
    faiss-cpu

RUN python -m pip install git+https://github.com/mike-holcomb/torch-spatial-kmeans.git

RUN python -m pip install ipykernel -U --user --force-reinstall

WORKDIR /srv    