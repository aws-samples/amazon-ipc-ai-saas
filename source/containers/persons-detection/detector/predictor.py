from __future__ import print_function
import os
import base64
import flask
import json
import cv2
import numpy as np
import time
import pycuda.autoinit  # This is needed for initializing CUDA driver
from yolo_with_plugins import TrtYOLO


MODEL_FULL_PATH = '/opt/ml/model/yolov4-persons.trt'
CATEGORY_NUM = 5
CLASS_ID_NAME_LUT = {
    0: 'pedestrian',
    1: 'rider',
    2: 'partially-visible person',
    3: 'ignore region',
    4: 'crowd'
}


# A singleton for holding the model. This simply loads the model and holds it.
# It has a predict function that does a prediction based on the model and the input data.
class ObjectDetectionService(object):
    detector = None

    @classmethod
    def load_model(cls):
        """
        get object detector for this instance, loading it if it is not already loaded
        :return:
        """
        if cls.detector is None:
            print('Initialize TensorRT Model...')
            cls.detector = TrtYOLO(
                model_full_path=MODEL_FULL_PATH,
                category_num=CATEGORY_NUM,
                letter_box=False)

        return cls.detector

    @classmethod
    def predict(cls, image_data):
        detector = cls.load_model()
        boxes, scores, classes = detector.detect(img=image_data, conf_th=0.05)
        return boxes, scores, classes


# The flask app for serving predictions
app = flask.Flask(__name__)


@app.route('/ping', methods=['GET'])
def ping():
    """
    Determine if the container is working and healthy. In this sample container, we declare
    it healthy if we can load the model successfully.

    :return:
    """
    health = ObjectDetectionService.load_model() is not None     # You can insert a health check here
    status = 200 if health else 404
    return flask.Response(response='\n', status=status, mimetype='application/json')


@app.route('/invocations', methods=['POST'])
def transformation():
    """
    Do an inference on a single batch of data. In this sample server, we take image data as base64 formation,
    decode it for internal use and then convert the predictions to json format

    :return:
    """
    t1 = time.time()
    if flask.request.content_type == 'application/json':
        request_body = flask.request.data.decode('utf-8')
        request_body = json.loads(request_body)
        image_bytes = request_body['image_bytes']
    else:
        return flask.Response(
            response='Object detector only supports application/json data',
            status=415,
            mimetype='text/plain')

    # decode image bytes
    base64_decoded = base64.b64decode(image_bytes)
    img_array = np.frombuffer(base64_decoded, np.uint8)
    image_data = cv2.imdecode(img_array, cv2.IMREAD_COLOR)   # BGR format
    height, width, channels = image_data.shape
    t2 = time.time()

    # Inference
    boxes, scores, class_ids = ObjectDetectionService.predict(image_data=image_data)
    t3 = time.time()

    ret_boxes = boxes.tolist()
    ret_scores = list()
    ret_class_names = list()

    for score in scores:
        ret_scores.append([score])

    for class_id in class_ids:
        ret_class_names.append([CLASS_ID_NAME_LUT[int(class_id)]])

    body = {
        'width': width,
        'height': height,
        'channels': channels,
        'bbox_coords': ret_boxes,           # shape = (N, 4)
        'bbox_scores': ret_scores,          # shape = (N, 1)
        'class_names': ret_class_names      # shape = (N, 1)
    }

    t4 = time.time()
    print('Total time cost = {} ms'.format(1000.0 * (t4 - t1)))
    print('\tTime cost of image decoding = {} ms'.format(1000.0 * (t2 - t1)))
    print('\tTime cost of inference (including image resize) = {} ms'.format(1000.0 * (t3 - t2)))
    print('\tTime cost of response post-processing = {} ms'.format(1000.0 * (t4 - t3)))
    print('Response = {}'.format(body))

    return flask.Response(response=json.dumps(body), status=200, mimetype='application/json')
