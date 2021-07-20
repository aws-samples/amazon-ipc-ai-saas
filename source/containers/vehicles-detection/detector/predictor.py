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
import os

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpus, True)
        except RuntimeError as e:
            print(e)

MODEL_ROOT_PATH = '/opt/ml/model/yolov4-512x512-vehicles-detection-tensorflow-tensorrt/'


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
            t1 = time.time()
            saved_model = tf.saved_model.load(MODEL_ROOT_PATH, tags=[tag_constants.SERVING])
            t2 = time.time()
            print('[Tensorflow TensorRT] Time cost of loading saved model = {}'.format(t2 - t1))
            graph_func = saved_model.signatures[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
            t3 = time.time()
            print('[Tensorflow TensorRT] Time cost of Signature = {}'.format(t3 - t2))
            cls.detector = convert_to_constants.convert_variables_to_constants_v2(graph_func)
            t4 = time.time()
            print('[Tensorflow TensorRT] Time cost of Converting to Constants = {}'.format(t4 - t3))

        return cls.detector

    @classmethod
    def warmup(cls):
        if cls.detector is not None:
            for _ in range(3):
                print('Warm up the model...')
                images_data = np.zeros(shape=(1, 512, 512, 3), dtype=np.float32)
                batch_data = tf.constant(images_data)
                _ = ObjectDetectionService.predict(image_batch_data=batch_data)

    @classmethod
    def predict(cls, image_batch_data):
        detector = cls.load_model()
        pred_bbox = detector(image_batch_data)
        return pred_bbox


# initialize the detection model
_ = ObjectDetectionService.load_model()

# warm up the inference
ObjectDetectionService.warmup()

command = "nvidia-smi"
os.system(command)

# the flask app for serving predictions
app = flask.Flask(__name__)


@app.route('/ping', methods=['GET'])
def ping():
    """
    Determine if the container is working and healthy. In this sample container, we declare
    it healthy if we can load the model successfully.

    :return:
    """
    health = ObjectDetectionService.detector is not None
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

    cls_id_cls_name_mapping = {
        0: 'bicycle',
        1: 'car',
        2: 'motorcycle',
        3: 'bus',
        4: 'train',
        5: 'truck'
    }
    t2 = time.time()

    # decode image
    base64_decoded = base64.b64decode(image_bytes)

    # get Image object and obtain its original height, width, channels
    image = Image.open(io.BytesIO(base64_decoded))
    width, height = image.size
    channels = len(image.getbands())
    t3 = time.time()

    # resize the image and store the scaling ratio in both width and height dimension
    resized_image = image.resize((512, 512), Image.NEAREST)
    scale_height_ratio = 512.0 / height
    scale_width_ratio = 512.0 / width
    t4 = time.time()

    # convert image data into tensor
    resized_image_np = np.array(resized_image)
    resized_normalized_image = resized_image_np / 255.0
    images_data = np.asarray([resized_normalized_image]).astype(np.float32)
    batch_data = tf.constant(images_data)
    t5 = time.time()

    # inference
    res = ObjectDetectionService.predict(image_batch_data=batch_data)
    matrix = res[0]
    boxes = matrix[:, :, 0:4]
    pred_conf = matrix[:, :, 4:]
    t6 = time.time()

    bbox_coords, bbox_scores, class_ids, valid_detections = tf.image.combined_non_max_suppression(
        boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
        scores=tf.reshape(pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
        max_output_size_per_class=50,
        max_total_size=50,
        iou_threshold=0.45,
        score_threshold=0.0
    )
    t7 = time.time()

    # convert EagerTensor to Numpy format
    bbox_coords = bbox_coords.numpy()
    bbox_scores = bbox_scores.numpy()
    class_ids = class_ids.numpy()

    ret_bbox_coords = list()    # shape = (N, 4)
    ret_bbox_scores = list()    # shape = (N, 1)
    ret_class_names = list()    # shape = (N, 1)

    for index in range(valid_detections[0]):
        [scale_y_min, scale_x_min, scale_y_max, scale_x_max] = bbox_coords[0][index]
        confidence = float(bbox_scores[0][index])
        cls_id = int(class_ids[0][index])

        ret_bbox_coords.append([
            int(scale_x_min * 512 / scale_width_ratio),
            int(scale_y_min * 512 / scale_height_ratio),
            int(scale_x_max * 512 / scale_width_ratio),
            int(scale_y_max * 512 / scale_height_ratio),
        ])
        ret_bbox_scores.append([confidence])
        cls_name = cls_id_cls_name_mapping[cls_id]
        ret_class_names.append([cls_name])

    t8 = time.time()

    body = {
        'width': width,
        'height': height,
        'channels': channels,
        'bbox_scores': ret_bbox_scores,     # shape = (N, 1)
        'bbox_coords': ret_bbox_coords,     # shape = (N, 4)
        'class_names': ret_class_names,     # shape = (N, 1)
    }
    print('Time Cost of Image input = {} second'.format(t2 - t1))
    print('Time Cost of Decode Image & IO Image Open = {} second'.format(t3 - t2))
    print('Time Cost of Image Resize = {} second'.format(t4 - t3))
    print('Time Cost of Tensor Conversion = {} second'.format(t5 - t4))
    print('Time Cost of Inference = {} second'.format(t6 - t5))
    print('Time Cost of NMS = {} second'.format(t7 - t6))
    print('Time Cost of Output JSON Preparation = {} second'.format(t8 - t7))

    print('Total Time Cost = {}'.format(t8 - t1))
    print('Response = {}'.format(body))

    return flask.Response(response=json.dumps(body), status=200, mimetype='application/json')
