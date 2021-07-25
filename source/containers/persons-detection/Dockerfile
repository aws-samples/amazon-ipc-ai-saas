# Reference: https://hub.docker.com/layers/nvidia/cuda/11.1.1-cudnn8-runtime-ubuntu18.04/images
FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu18.04

ENV DEBIAN_FRONTEND=noninteractive

# INSTALL DEPENDENCIES
RUN apt-get -y update && \
    apt-get install -y nginx \
    wget \
    git \
    cmake \
    libopencv-dev \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip
RUN pip3 install --no-cache opencv-python==4.5.2.54 \
    pycuda==2021.1 \
    Cython==0.29.24 \
    onnx==1.4.1 \
    numpy==1.19.5 \
    flask==1.1.2 \
    gevent==20.12.1 \
    gunicorn==20.0.4

# INSTALL TENSORRT AND ITS ALL DEPENDENCIES
RUN mkdir -p /opt/software
RUN wget -c https://ip-camera-ai-saas.s3.amazonaws.com/software/nv-tensorrt-repo-ubuntu1804-cuda11.1-trt7.2.3.4-ga-20210226_1-1_amd64.deb -O /opt/software/nv-tensorrt-repo-ubuntu1804-cuda11.1-trt7.2.3.4-ga-20210226_1-1_amd64.deb
RUN dpkg -i /opt/software/nv-tensorrt-repo-ubuntu1804-cuda11.1-trt7.2.3.4-ga-20210226_1-1_amd64.deb
RUN apt-key add /var/nv-tensorrt-repo-ubuntu1804-cuda11.1-trt7.2.3.4-ga-20210226/7fa2af80.pub
RUN apt-get update
RUN apt-get -y install libnvinfer-dev=7.2.3-1+cuda11.1
RUN apt-get -y install libnvinfer-plugin-dev=7.2.3-1+cuda11.1
RUN apt-get -y install libnvparsers-dev=7.2.3-1+cuda11.1
RUN apt-get -y install libnvonnxparsers-dev=7.2.3-1+cuda11.1
RUN apt-get -y install libnvinfer-samples=7.2.3-1+cuda11.1
RUN apt-get install -y tensorrt

# DOWNLOAD INFERENCE MODEL WEIGHTS (TENSORRT)
RUN mkdir -p /opt/ml/model
# RUN wget -c https://ipc-models-zoo.s3.amazonaws.com/body-detector/body_detector_yolo3_mobilenet1.0_coco-symbol.json -O /opt/ml/model/body_detector_yolo3_mobilenet1.0_coco-symbol.json

# Set some environment variables. PYTHONUNBUFFERED keeps Python from buffering our standard
# output stream, which means that logs can be delivered to the user quickly. PYTHONDONTWRITEBYTECODE
# keeps Python from writing the .pyc files which are unnecessary in this case. We also update
# PATH so that the train and serve programs are found when the container is invoked.
ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/program:${PATH}"

# SETUP WORKDIR
COPY detector /opt/program
RUN chmod +x /opt/program/serve
WORKDIR /opt/program

# BUILD libyolo_layer.so
RUN cd /opt/program/plugins && make
ENV LD_LIBRARY_PATH="/opt/program/plugins:${LD_LIBRARY_PATH}"

ENTRYPOINT ["python", "serve"]
