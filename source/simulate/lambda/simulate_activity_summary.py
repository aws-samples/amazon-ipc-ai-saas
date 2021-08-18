import base64
import os
import json
import time
import cv2
import numpy as np
import requests


class ActivitySummaryDemo(object):
    def __init__(self, request_url, activity_id):
        self._request_url = request_url
        self._activity_id = activity_id

    def run(self):
        request_body = {
            "activity_id": self._activity_id,
            "distance_threshold": 0.52
        }

        response = requests.post(self._request_url, data=json.dumps(request_body))
        print('Response = {}'.format(response.text))


if __name__ == '__main__':
    demo = ActivitySummaryDemo(
        request_url="https://your_api_gateway_endpoint_url/activity",
        activity_id="Activity_2021_08_29_BJ001"
    )
    demo.run()
