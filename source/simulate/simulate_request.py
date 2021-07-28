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
    def __init__(self, endpoint_url, application_type='PetsDetection'):
        self._endpoint_url = endpoint_url + 'inference'
        self._application_type = application_type

        if self._application_type == 'PetsDetection':
            self._test_images_dir = './test_imgs/pets'
            self._cls_name_to_id_mapping = {
                'cat': 0,
                'dog': 1
            }
            self._class_names_lut = ['cat', 'dog']
        elif self._application_type == 'VehiclesDetection':
            self._test_images_dir = './test_imgs/vehicles'
            self._cls_name_to_id_mapping = {
                'bicycle': 0,
                'car': 1,
                'motorcycle': 2,
                'bus': 3,
                'train': 4,
                'truck': 5
            }
            self._class_names_lut = ['bicycle', 'car', 'motorcycle', 'bus', 'train', 'truck']
        elif self._application_type == 'PersonsDetection':
            self._test_images_dir = './test_imgs/persons'
            self._cls_name_to_id_mapping = {
                'pedestrian': 0,
                'rider': 1,
                'partially-visible person': 2,
                'ignore region': 3,
                'crowd': 4
            }
            self._class_names_lut = ['pedestrian', 'rider', 'partially-visible person', 'ignore region', 'crowd']
        else:
            print('Not supported application type.')

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
        image_names = [f for f in os.listdir(self._test_images_dir) if f.endswith('jpg')]
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

            # json.dump(request_body, open('post_data.txt', 'w'))

            t1 = time.time()
            response = requests.post(self._endpoint_url, data=json.dumps(request_body))
            t2 = time.time()
            print('Time cost = {}'.format(t2 - t1))

            # Step 3: visualization
            response = json.loads(response.text)
            print('Response = {}'.format(response))

            bbox_coords = np.array(response['bbox_coords'])
            bbox_scores = np.array(response['bbox_scores'])
            class_names = np.array(response['class_names'])
            print('bbox_coords.shape = {}'.format(bbox_coords.shape))
            print('bbox_scores.shape = {}'.format(bbox_scores.shape))
            print('class_names.shape = {}'.format(class_names.shape))

            cls_ids = list()
            for cls_name in class_names:
                cls_id = self._cls_name_to_id_mapping[cls_name[0]]
                cls_ids.append([cls_id])

            self.visualize(
                full_path,
                np.array(bbox_coords),
                np.array(bbox_scores),
                np.array(cls_ids),
                class_names=self._class_names_lut)


if __name__ == '__main__':
    simulator = DetectorSimulator(
        endpoint_url="https://jwfaindhqi.execute-api.cn-northwest-1.amazonaws.com.cn/prod/",
        application_type='PersonsDetection'
    )
    simulator.run()
