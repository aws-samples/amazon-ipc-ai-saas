import cv2
import numpy as np
import time
import base64
import io
from PIL import Image
import pycuda.autoinit  # This is needed for initializing CUDA driver
from yolo_with_plugins import TrtYOLO
from gluoncv import utils
from matplotlib import pyplot as plt

MODEL_FULL_PATH = './yolov4-persons.trt'
CATEGORY_NUM = 5


class ObjectDetector(object):
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


def visualize(full_path, bbox_coords, bbox_scores, class_ids, class_names):
    image = cv2.imread(full_path, cv2.IMREAD_COLOR)
    image = image[:, :, ::-1]
    ax = utils.viz.plot_bbox(
        img=image,
        bboxes=bbox_coords,
        scores=bbox_scores,
        labels=class_ids,
        thresh=0.25,
        class_names=class_names)
    plt.axis('off')
    plt.savefig('person-detect-yolo-v4-result.png', dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()


CLASS_ID_NAME_LUT = {
    0: 'pedestrian',
    1: 'rider',
    2: 'partially-visible person',
    3: 'ignore region',
    4: 'crowd'
}


def get_base64_encoding(full_path):
    with open(full_path, "rb") as f:
        data = f.read()
        image_base64_enc = base64.b64encode(data)
        image_base64_enc = str(image_base64_enc, 'utf-8')

    return image_base64_enc


def evaluate_on_inference_speed(image_full_path, iter_times=1000):
    """
    evaluate the inference speed

    :param image_full_path: full path of test image
    :param iter_times: total iterative times
    :return:
    """
    image_bytes = get_base64_encoding(image_full_path)

    print('Start to inference...')
    # inference speed evaluation
    t_start = time.time()
    for i in range(iter_times):
        base64_decoded = base64.b64decode(image_bytes)
        img_array = np.frombuffer(base64_decoded, np.uint8)
        image_data = cv2.imdecode(img_array, cv2.IMREAD_COLOR)  # BGR format
        height, width, channels = image_data.shape
        boxes, scores, class_ids = ObjectDetector.predict(image_data=image_data)

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
        'class_names': ret_class_names      # shape = (N, 4)
    }

    t_end = time.time()
    print('Inference Time Cost Each Frame (512x512x3) With TensorRT = {} ms'.format(1000.0 * (t_end - t_start) / iter_times))

    vis_boxes = np.array(boxes)
    vis_scores = np.array([[s] for s in scores])
    vis_class_ids = np.array([[cls_id] for cls_id in class_ids])
    visualize(
        full_path=image_full_path,
        bbox_coords=vis_boxes,
        bbox_scores=vis_scores,
        class_ids=vis_class_ids,
        class_names=['pedestrian', 'rider', 'partially-visible person', 'ignore region', 'crowd'])


if __name__ == '__main__':
    evaluate_on_inference_speed(image_full_path='./persons_detect_test.jpg')

