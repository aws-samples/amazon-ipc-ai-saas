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

# download face representation mdoels
RUN mkdir -p /opt/ml
RUN wget -c https://ip-camera-ai-saas.s3.amazonaws.com/models/faces/models.zip -O /opt/ml/models.zip
RUN cd /opt/ml/ && unzip models.zip && rm -rf models.zip

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
