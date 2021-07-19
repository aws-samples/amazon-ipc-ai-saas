# Reference: https://github.com/aws/deep-learning-containers/blob/master/available_images.md
ARG REGISTRY_URI
FROM ${REGISTRY_URI}

RUN apt-get -y update && \
    apt-get install -y nginx \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache gluoncv==0.8.0 \
    numpy==1.19.5 \
    flask==1.1.2 \
    gevent==20.12.1 \
    gunicorn==20.0.4

# download model parameters
RUN mkdir -p /opt/ml/model
RUN wget -c https://ipc-models-zoo.s3.amazonaws.com/body-detector/body_detector_yolo3_mobilenet1.0_coco-symbol.json -O /opt/ml/model/body_detector_yolo3_mobilenet1.0_coco-symbol.json
RUN wget -c https://ipc-models-zoo.s3.amazonaws.com/body-detector/body_detector_yolo3_mobilenet1.0_coco-0000.params -O /opt/ml/model/body_detector_yolo3_mobilenet1.0_coco-0000.params

RUN wget -c https://ipc-models-zoo.s3.amazonaws.com/body-detector/body_detector_yolo3_darknet53_coco-symbol.json -O /opt/ml/model/body_detector_yolo3_darknet53_coco-symbol.json
RUN wget -c https://ipc-models-zoo.s3.amazonaws.com/body-detector/body_detector_yolo3_darknet53_coco-0000.params -O /opt/ml/model/body_detector_yolo3_darknet53_coco-0000.params

RUN wget -c https://ipc-models-zoo.s3.amazonaws.com/body-detector/body_detector_ssd_512_resnet50_v1_coco-symbol.json -O /opt/ml/model/body_detector_ssd_512_resnet50_v1_coco-symbol.json
RUN wget -c https://ipc-models-zoo.s3.amazonaws.com/body-detector/body_detector_ssd_512_resnet50_v1_coco-0000.params -O /opt/ml/model/body_detector_ssd_512_resnet50_v1_coco-0000.params

RUN wget -c https://ipc-models-zoo.s3.amazonaws.com/body-detector/body_detector_faster_rcnn_fpn_resnet101_v1d_coco-symbol.json -O /opt/ml/model/body_detector_faster_rcnn_fpn_resnet101_v1d_coco-symbol.json
RUN wget -c https://ipc-models-zoo.s3.amazonaws.com/body-detector/body_detector_faster_rcnn_fpn_resnet101_v1d_coco-0000.params -O /opt/ml/model/body_detector_faster_rcnn_fpn_resnet101_v1d_coco-0000.params

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
