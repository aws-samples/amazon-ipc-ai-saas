# Reference: https://github.com/aws/deep-learning-containers/blob/master/available_images.md
ARG REGISTRY_URI
FROM ${REGISTRY_URI}

RUN apt-get -y update && \
    apt-get install -y libopencv-dev \
    nginx \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache gluoncv==0.8.0 \
    insightface==0.1.5 \
    opencv-python==4.5.1.48 \
    numpy==1.19.5 \
    flask==1.1.2 \
    gevent==20.12.1 \
    gunicorn==20.0.4

# download face comparison model parameters and place them in /opt/ml/model
RUN mkdir -p /opt/ml/model/model-r100-ii
RUN wget -c https://ipc-models-zoo.s3.amazonaws.com/face-recognizer/model-r100-ii/model-0000.params -O /opt/ml/model/model-r100-ii/model-0000.params
RUN wget -c https://ipc-models-zoo.s3.amazonaws.com/face-recognizer/model-r100-ii/model-symbol.json -O /opt/ml/model/model-r100-ii/model-symbol.json

RUN mkdir -p /opt/ml/model/model-y1-test2
RUN wget -c https://ipc-models-zoo.s3.amazonaws.com/face-recognizer/model-y1-test2/model-0000.params -O /opt/ml/model/model-y1-test2/model-0000.params
RUN wget -c https://ipc-models-zoo.s3.amazonaws.com/face-recognizer/model-y1-test2/model-symbol.json -O /opt/ml/model/model-y1-test2/model-symbol.json

RUN mkdir -p /opt/ml/model/retinaface_mnet025_v2
RUN wget -c https://ipc-models-zoo.s3.amazonaws.com/face-detector/retinaface_mnet025_v2/mnet12-0000.params -O /opt/ml/model/retinaface_mnet025_v2/mnet12-0000.params
RUN wget -c https://ipc-models-zoo.s3.amazonaws.com/face-detector/retinaface_mnet025_v2/mnet12-symbol.json -O /opt/ml/model/retinaface_mnet025_v2/mnet12-symbol.json

RUN mkdir -p /opt/ml/model/retinaface_r50_v1
RUN wget -c https://ipc-models-zoo.s3.amazonaws.com/face-detector/retinaface_r50_v1/R50-0000.params -O /opt/ml/model/retinaface_r50_v1/R50-0000.params
RUN wget -c https://ipc-models-zoo.s3.amazonaws.com/face-detector/retinaface_r50_v1/R50-symbol.json -O /opt/ml/model/retinaface_r50_v1/R50-symbol.json

# Set some environment variables. PYTHONUNBUFFERED keeps Python from buffering our standard
# output stream, which means that logs can be delivered to the user quickly. PYTHONDONTWRITEBYTECODE
# keeps Python from writing the .pyc files which are unnecessary in this case. We also update
# PATH so that the train and serve programs are found when the container is invoked.
ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/program:${PATH}"

# Set up the program in the image
COPY recognizer /opt/program
RUN chmod +x /opt/program/serve
WORKDIR /opt/program

ENTRYPOINT ["python", "serve"]
