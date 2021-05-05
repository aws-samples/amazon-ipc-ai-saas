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
    def __init__(self, faces_detect_root_dir, body_detect_root_dir):
        self._face_detect_url = "https://5neg1jax88.execute-api.us-east-1.amazonaws.com/prod/inference"
        self._body_detect_url = "https://5neg1jax88.execute-api.us-east-1.amazonaws.com/prod/inference"

        self._faces_detect_root_dir = faces_detect_root_dir
        self._body_detect_root_dir = body_detect_root_dir

    @staticmethod
    def get_base64_encoding(full_path):
        with open(full_path, "rb") as f:
            data = f.read()
            image_base64_enc = base64.b64encode(data)
            image_base64_enc = str(image_base64_enc, 'utf-8')

        return image_base64_enc

    @staticmethod
    def visualize(full_path, bbox_coords, bbox_scores, class_ids, label_name='face'):
        image = cv2.imread(full_path, cv2.IMREAD_COLOR)
        image = image[:, :, ::-1]
        ax = utils.viz.plot_bbox(image, bbox_coords, bbox_scores, class_ids, class_names=[label_name], thresh=0.25)
        plt.axis('off')
        plt.show()

    def face_detect_simulate(self):
        image_names = [f for f in os.listdir(self._faces_detect_root_dir) if f.startswith('test_')]
        image_names = sorted(image_names)

        for name in image_names:
            full_path = os.path.join(self._faces_detect_root_dir, name)
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
            response = requests.post(self._face_detect_url, data=json.dumps(request_body))
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

            self.visualize(full_path, bbox_coords, bbox_scores, class_ids, label_name='face')

    def body_detect_simulate(self):
        image_names = [f for f in os.listdir(self._body_detect_root_dir) if f.startswith('test_')]
        image_names = sorted(image_names)

        for name in image_names:
            full_path = os.path.join(self._body_detect_root_dir, name)
            print('Test image {}:'.format(full_path))

            # Step 1: read image and execute base64 encoding
            image_base64_enc = self.get_base64_encoding(full_path)

            # Step 2: send request to backend
            request_body = {
                "timestamp": str(time.time()),
                "request_id": 1242322,
                "image_base64_enc": image_base64_enc
            }

            response = requests.post(self._body_detect_url, data=json.dumps(request_body))

            # Step 3: visualization
            response = json.loads(response.text)
            print('Response = {}'.format(response))

            bbox_coords = np.array(response['bbox_coords'])
            bbox_scores = np.array(response['bbox_scores'])
            class_ids = np.zeros(shape=(bbox_scores.shape[0], ))
            print('bbox_coords.shape = {}'.format(bbox_coords.shape))
            print('bbox_scores.shape = {}'.format(bbox_scores.shape))

            self.visualize(full_path, bbox_coords, bbox_scores, class_ids, label_name='body')


if __name__ == '__main__':
    simulator = DetectorSimulator(
        faces_detect_root_dir='./faces',
        body_detect_root_dir='./persons'
    )

    # simulator.face_detect_simulate()
    simulator.body_detect_simulate()
