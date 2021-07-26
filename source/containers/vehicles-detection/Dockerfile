FROM ubuntu:18.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/usr/local/cuda/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"


# Install dependencies
RUN apt-get -y update && \
    apt-get install -y nginx \
    wget \
    git \
    cmake \
    python3 \
    python3-opencv \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip==21.2.1
RUN pip3 install --no-cache Cython==0.29.24 \
    onnx==1.4.1 \
    numpy==1.19.5 \
    flask==1.1.2 \
    gevent==20.12.1 \
    gunicorn==20.0.4


# Install CUDA
RUN mkdir -p /opt/software
RUN wget -c https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-ubuntu1804.pin -O /opt/software/cuda-ubuntu1804.pin
RUN mv /opt/software/cuda-ubuntu1804.pin /etc/apt/preferences.d/cuda-repository-pin-600
RUN wget -c https://developer.download.nvidia.com/compute/cuda/11.1.0/local_installers/cuda-repo-ubuntu1804-11-1-local_11.1.0-455.23.05-1_amd64.deb -O /opt/software/cuda-repo-ubuntu1804-11-1-local_11.1.0-455.23.05-1_amd64.deb
RUN dpkg -i /opt/software/cuda-repo-ubuntu1804-11-1-local_11.1.0-455.23.05-1_amd64.deb
RUN apt-key add /var/cuda-repo-ubuntu1804-11-1-local/7fa2af80.pub
RUN apt-get update && \
    apt-get -y install cuda

# Install pycuda for python
RUN pip3 install --no-cache  'pycuda<2021.1'

# Install TensorRT and its dependencies
RUN wget -c https://ip-camera-ai-saas.s3.amazonaws.com/software/nv-tensorrt-repo-ubuntu1804-cuda11.1-trt7.2.3.4-ga-20210226_1-1_amd64.deb -O /opt/software/nv-tensorrt-repo-ubuntu1804-cuda11.1-trt7.2.3.4-ga-20210226_1-1_amd64.deb
RUN dpkg -i /opt/software/nv-tensorrt-repo-ubuntu1804-cuda11.1-trt7.2.3.4-ga-20210226_1-1_amd64.deb
RUN apt-key add /var/nv-tensorrt-repo-ubuntu1804-cuda11.1-trt7.2.3.4-ga-20210226/7fa2af80.pub
RUN apt-get -y update && \
    apt-get -y install tensorrt
RUN apt-get -y install python3-libnvinfer-dev

# Download TensorRT model
RUN mkdir -p /opt/ml/model
RUN wget -c https://ip-camera-ai-saas.s3.amazonaws.com/models/vehicles_detection/yolov4-vehicles.trt -O /opt/ml/model/yolov4-vehicles.trt

# Copy source code to container
COPY detector /opt/program
RUN chmod +x /opt/program/serve
WORKDIR /opt/program

# Set some environment variables. PYTHONUNBUFFERED keeps Python from buffering our standard
# output stream, which means that logs can be delivered to the user quickly. PYTHONDONTWRITEBYTECODE
# keeps Python from writing the .pyc files which are unnecessary in this case. We also update
# PATH so that the train and serve programs are found when the container is invoked.
ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/program:${PATH}"

ENTRYPOINT ["python3", "serve"]

