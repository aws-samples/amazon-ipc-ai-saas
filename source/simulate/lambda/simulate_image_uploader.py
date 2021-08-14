import base64
import os
import json
import time
import cv2
import numpy as np
import requests


class ImageUploaderDemo(object):
    def __init__(self, uploader_url, test_image_full_path):
        self._uploader_url = uploader_url
        self._test_image_full_path = test_image_full_path

    @staticmethod
    def get_base64_encoding(test_image_full_path):
        with open(test_image_full_path, "rb") as f:
            data = f.read()
            image_base64_enc = base64.b64encode(data)
            image_base64_enc = str(image_base64_enc, 'utf-8')

        return image_base64_enc

    def run(self):
        image_base64_enc = self.get_base64_encoding(test_image_full_path=self._test_image_full_path)

        request_body = {
            "activity_id": "Activity_2021_08_29_BJ001",
            "image_id": "VIZ10029",
            "image_base64_enc": image_base64_enc
        }

        response = requests.post(self._uploader_url, data=json.dumps(request_body))
        print('Response = {}'.format(response.text))


if __name__ == '__main__':
    demo = ImageUploaderDemo(
        uploader_url="https://your_api_gateway_endpoint/prod/upload",
        test_image_full_path="./test_imgs/face.jpeg"
    )
    demo.run()
