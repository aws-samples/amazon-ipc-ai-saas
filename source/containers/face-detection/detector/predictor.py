from __future__ import print_function
import os
import base64
from mxnet import gluon
import mxnet as mx
import flask
import json
import time

model_root_dir = '/opt/ml/model'
object_detection_model_name = os.environ.get('OBJECT_DETECTION_MODEL_NAME', 'yolo3_darknet53_coco')


# A singleton for holding the model. This simply loads the model and holds it.
# It has a predict function that does a prediction based on the model and the input data.
class ObjectDetectionService(object):
    # class attributes
    detector = None
    ctx = mx.cpu() if mx.context.num_gpus() == 0 else mx.gpu()

    @classmethod
    def get_model(cls):
        """
        Get the model object for this instance, loading it if it's not already loaded.

        :return:
        """
        if cls.detector is None:
            if object_detection_model_name == 'ssd_512_resnet50_v1_coco':
                cls.detector = gluon.nn.SymbolBlock.imports(
                    symbol_file=os.path.join(model_root_dir, 'face_detector_ssd_512_resnet50_v1_coco-symbol.json'),
                    input_names=['data'],
                    param_file=os.path.join(model_root_dir, 'face_detector_ssd_512_resnet50_v1_coco-0000.params'),
                    ctx=cls.ctx
                )
            elif object_detection_model_name == 'yolo3_darknet53_coco':
                cls.detector = gluon.nn.SymbolBlock.imports(
                    symbol_file=os.path.join(model_root_dir, 'face_detector_yolo3_darknet53_coco-symbol.json'),
                    input_names=['data'],
                    param_file=os.path.join(model_root_dir, 'face_detector_yolo3_darknet53_coco-0000.params'),
                    ctx=cls.ctx
                )
            elif object_detection_model_name == 'yolo3_mobilenet1.0_coco':
                cls.detector = gluon.nn.SymbolBlock.imports(
                    symbol_file=os.path.join(model_root_dir, 'face_detector_yolo3_mobilenet1.0_coco-symbol.json'),
                    input_names=['data'],
                    param_file=os.path.join(model_root_dir, 'face_detector_yolo3_mobilenet1.0_coco-0000.params'),
                    ctx=cls.ctx
                )
            elif object_detection_model_name == 'faster_rcnn_fpn_resnet101_v1d_coco':
                cls.detector = gluon.nn.SymbolBlock.imports(
                    symbol_file=os.path.join(model_root_dir, 'face_detector_faster_rcnn_fpn_resnet101_v1d_coco-symbol.json'),
                    input_names=['data'],
                    param_file=os.path.join(model_root_dir, 'face_detector_faster_rcnn_fpn_resnet101_v1d_coco-0000.params'),
                    ctx=cls.ctx
                )
            else:
                return None
        return cls.detector

    @classmethod
    def predict(cls, resized_rescaled_normalized_img):
        handler = cls.get_model()
        class_ids, mx_scores, mx_bounding_boxes = handler(resized_rescaled_normalized_img)
        return class_ids, mx_scores, mx_bounding_boxes


# The flask app for serving predictions
app = flask.Flask(__name__)


@app.route('/ping', methods=['GET'])
def ping():
    """
    Determine if the container is working and healthy. In this sample container, we declare
    it healthy if we can load the model successfully.

    :return:
    """
    health = ObjectDetectionService.get_model() is not None     # You can insert a health check here
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
        short_size = request_body['short_size']
    else:
        return flask.Response(
            response='Object detector only supports application/json data',
            status=415,
            mimetype='text/plain')

    # pre-process
    img = mx.img.imdecode(base64.b64decode(image_bytes))
    height, width, channels = img.shape[0], img.shape[1], img.shape[2]

    mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    resized_img = mx.image.resize_short(img, size=short_size)
    resized_rescaled_img = mx.nd.image.to_tensor(resized_img)
    resized_rescaled_normalized_img = mx.nd.image.normalize(resized_rescaled_img, mean=mean, std=std)
    resized_rescaled_normalized_img = resized_rescaled_normalized_img.expand_dims(0)
    if mx.context.num_gpus() != 0:
        resized_rescaled_normalized_img = resized_rescaled_normalized_img.copyto(mx.gpu())

    # inference
    class_ids, mx_scores, mx_bounding_boxes = ObjectDetectionService.predict(resized_rescaled_normalized_img)

    # post-process
    class_ids = class_ids.asnumpy()
    mx_scores = mx_scores.asnumpy()
    mx_bounding_boxes = mx_bounding_boxes.asnumpy()

    # resize detection results back to original image size
    scale_ratio = short_size / height if height < width else short_size / width
    bbox_coords, bbox_scores = list(), list()
    for index, bbox in enumerate(mx_bounding_boxes[0]):
        prob = float(mx_scores[0][index][0])
        if prob < 0.0:
            continue

        [x_min, y_min, x_max, y_max] = bbox
        x_min = int(x_min / scale_ratio)
        y_min = int(y_min / scale_ratio)
        x_max = int(x_max / scale_ratio)
        y_max = int(y_max / scale_ratio)
        bbox_coords.append([x_min, y_min, x_max, y_max])
        bbox_scores.append([prob])

    body = {
        'width': width,
        'height': height,
        'channels': channels,
        'bbox_scores': bbox_scores,  # shape = (N, 1)
        'bbox_coords': bbox_coords,  # shape = (N, 4)
    }
    t_end = time.time()
    print('Time consumption = {} second'.format(t_end - t_start))
    print('Response = {}'.format(body))

    return flask.Response(response=json.dumps(body), status=200, mimetype='application/json')
