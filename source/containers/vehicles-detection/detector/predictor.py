from __future__ import print_function
import base64
import tensorflow as tf
from tensorflow.python.saved_model import tag_constants, signature_constants
from tensorflow.python.framework import convert_to_constants
import time
import flask
import numpy as np
import json
import io
from PIL import Image


gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


MODEL_ROOT_PATH = '/opt/ml/model/yolov4-512x512-vehicles-detection-tensorflow-tensorrt/'
PROCESS_START_TIMESTAMP = time.time()


class ObjectDetectionService(object):
    """
    A singleton for holding the model. This simply loads the model and holds it. It has a predict function that
    does a prediction based on the model and the input data.
    """
    detector = None

    @classmethod
    def load_model(cls):
        """
        get object detector for this instance, loading it if it is not already loaded
        :return:
        """
        if cls.detector is None:
            saved_model = tf.saved_model.load(MODEL_ROOT_PATH, tags=[tag_constants.SERVING])
            graph_func = saved_model.signatures[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
            cls.detector = convert_to_constants.convert_variables_to_constants_v2(graph_func)

        return cls.detector

    # @classmethod
    # def warmup(cls, dummy_data):
    #     detector = cls.load_model()
    #     for i in range(5):
    #         detector(dummy_data)

    @classmethod
    def predict(cls, image_batch_data):
        detector = cls.load_model()
        pred_bbox = detector(image_batch_data)
        return pred_bbox


# The flask app for serving predictions
app = flask.Flask(__name__)
ObjectDetectionService.load_model()


@app.route('/ping', methods=['GET'])
def ping():
    """
    Determine if the container is working and healthy. In this sample container, we declare
    it healthy if we can load the model successfully.

    :return:
    """
    health = ObjectDetectionService.load_model() is not None
    status = 200 if health else 404
    return flask.Response(response='\n', status=status, mimetype='application/json')


@app.route('/invocations', methods=['POST'])
def transformation():
    """
    Do an inference on a single batch of data. In this sample server, we take image data as base64 formation,
    decode it for internal use and then convert the predictions to json format

    :return:
    """
    t_start = time.time()

    if flask.request.content_type == 'application/json':
        request_body = flask.request.data.decode('utf-8')
        request_body = json.loads(request_body)
        image_bytes = request_body['image_bytes']
    else:
        return flask.Response(
            response='Object detector only supports application/json data',
            status=415,
            mimetype='text/plain')

    cls_id_cls_name_mapping = {
        0: 'pedestrian',
        1: 'riders',
        2: 'pv person',
        3: 'ignore',
        4: 'crowd'
    }

    # decode image
    base64_decoded = base64.b64decode(image_bytes)

    # get Image object and obtain its original height, width, channels
    image = Image.open(io.BytesIO(base64_decoded))
    height, width, channels = image.shape

    # resize the image and store the scaling ratio in both width and height dimension
    resized_image = image.resize((512, 512))
    scale_height_ratio = 512.0 / height
    scale_width_ratio = 512.0 / width

    # convert image data into tensor
    resized_image_np = np.array(resized_image)
    resized_normalized_image = resized_image_np / 255.0
    images_data = np.asarray([resized_normalized_image]).astype(np.float32)
    batch_data = tf.constant(images_data)

    # inference
    res = ObjectDetectionService.predict(image_batch_data=batch_data)
    matrix = res[0]
    boxes = matrix[:, :, 0:4]
    pred_conf = matrix[:, :, 4:]

    bbox_coords, bbox_scores, class_ids, valid_detections = tf.image.combined_non_max_suppression(
        boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
        scores=tf.reshape(pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
        max_output_size_per_class=50,
        max_total_size=50,
        iou_threshold=0.45,
        score_threshold=0.0
    )

    ret_bbox_coords = list()    # shape = (N, 4)
    ret_bbox_scores = list()    # shape = (N, 1)
    ret_class_ids = list()      # shape = (N, 1)

    for index in range(valid_detections[0]):
        [scale_y_min, scale_x_min, scale_y_max, scale_x_max] = bbox_coords[0][index]
        confidence = bbox_scores[0][index]
        cls_id = class_ids[0][index]

        ret_bbox_coords.append([
            int(scale_x_min * 512 / scale_width_ratio),
            int(scale_y_min * 512 / scale_height_ratio),
            int(scale_x_max * 512 / scale_width_ratio),
            int(scale_y_max * 512 / scale_height_ratio),
        ])
        ret_bbox_scores.append([confidence])
        ret_class_ids.append([cls_id])

    ret_bbox_coords = np.array(ret_bbox_coords)
    ret_bbox_scores = np.array(ret_bbox_scores)
    ret_class_ids = np.array(ret_class_ids)

    print('ret_bbox_coords.shape = {}'.format(ret_bbox_coords.shape))
    print('ret_bbox_scores.shape = {}'.format(ret_bbox_scores.shape))
    print('ret_class_ids.shape = {}'.format(ret_class_ids.shape))

    body = {
        'width': width,
        'height': height,
        'channels': channels,
        'bbox_scores': ret_bbox_scores,     # shape = (N, 1)
        'bbox_coords': ret_bbox_coords,     # shape = (N, 4)
        'class_ids': ret_class_ids,         # shape = (N, 1)
    }
    t_end = time.time()
    print('Time consumption = {} second'.format(t_end - t_start))
    print('Response = {}'.format(body))

    return flask.Response(response=json.dumps(body), status=200, mimetype='application/json')
