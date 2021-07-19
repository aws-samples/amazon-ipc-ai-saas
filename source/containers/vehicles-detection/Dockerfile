# Reference: https://github.com/aws/deep-learning-containers/blob/master/available_images.md
ARG REGISTRY_URI
FROM ${REGISTRY_URI}

RUN apt-get -y update && \
    apt-get install -y nginx unzip \
    && rm -rf /var/lib/apt/lists/*

RUN /usr/local/bin/python3.7 -m pip install --upgrade pip
RUN pip3 install --no-cache tensorflow-gpu==2.4.0 \
    Pillow==8.3.1 \
    numpy==1.19.5 \
    flask==1.1.2 \
    gevent==20.12.1 \
    gunicorn==20.0.4

# download model parameters
RUN mkdir -p /opt/ml/model
RUN wget -c https://ip-camera-ai-saas.s3.amazonaws.com/model-zoo/vehicles_detection/yolov4-512x512-vehicles-detection-tensorflow-tensorrt.zip -O /opt/ml/model/yolov4-512x512-vehicles-detection-tensorflow-tensorrt.zip
RUN unzip /opt/ml/model/yolov4-512x512-vehicles-detection-tensorflow-tensorrt.zip -d /opt/ml/model/

# Set some environment variables. PYTHONUNBUFFERED keeps Python from buffering our standard
# output stream, which means that logs can be delivered to the user quickly. PYTHONDONTWRITEBYTECODE
# keeps Python from writing the .pyc files which are unnecessary in this case. We also update
# PATH so that the train and serve programs are found when the container is invoked.
ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/program:${PATH}"

# Set up the program in the image
COPY detector /opt/program
RUN chmod +x /opt/program/serve
WORKDIR /opt/program

ENTRYPOINT ["python", "serve"]
