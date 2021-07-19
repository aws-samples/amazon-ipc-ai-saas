import base64
import os
import json
import time
import cv2
import numpy as np
from gluoncv import utils
from matplotlib import pyplot as plt
import requests


class DetectorSimulator(object):
    def __init__(self, endpoint_url, test_images_dir):
        self._endpoint_url = endpoint_url + 'inference'
        self._test_images_dir = test_images_dir

    @staticmethod
    def get_base64_encoding(full_path):
        with open(full_path, "rb") as f:
            data = f.read()
            image_base64_enc = base64.b64encode(data)
            image_base64_enc = str(image_base64_enc, 'utf-8')

        return image_base64_enc

    @staticmethod
    def visualize(full_path, bbox_coords, bbox_scores, class_ids, class_names):
        image = cv2.imread(full_path, cv2.IMREAD_COLOR)
        image = image[:, :, ::-1]
        ax = utils.viz.plot_bbox(image, bbox_coords, bbox_scores, class_ids, class_names=class_names, thresh=0.25)
        plt.axis('off')
        plt.show()

    def run(self):
        image_names = [f for f in os.listdir(self._test_images_dir) if f.startswith('test_')]
        image_names = sorted(image_names)

        for name in image_names:
            full_path = os.path.join(self._test_images_dir, name)
            print('Test image {}:'.format(full_path))

            # Step 1: read image and execute base64 encoding
            image_base64_enc = self.get_base64_encoding(full_path)

            # Step 2: send request to backend
            request_body = {
                "timestamp": str(time.time()),
                "request_id": 1242322,
                "image_base64_enc": image_base64_enc
            }

            t1 = time.time()
            response = requests.post(self._endpoint_url, data=json.dumps(request_body))
            t2 = time.time()
            print('Time cost = {}'.format(t2 - t1))

            # Step 3: visualization
            response = json.loads(response.text)
            print('Response = {}'.format(response))

            bbox_coords = np.array(response['bbox_coords'])
            bbox_scores = np.array(response['bbox_scores'])
            class_ids = np.zeros(shape=(bbox_scores.shape[0], ))
            print('bbox_coords.shape = {}'.format(bbox_coords.shape))
            print('bbox_scores.shape = {}'.format(bbox_scores.shape))

            self.visualize(full_path, bbox_coords, bbox_scores, class_ids,
                           class_names=['pedestrian', 'riders', 'pv person', 'ignore', 'crowd'])


if __name__ == '__main__':
    simulator = DetectorSimulator(
        endpoint_url="https://zsyf7d49k3.execute-api.us-east-1.amazonaws.com/prod/",
        test_images_dir='./vehicles/'
    )
    simulator.run()
