import base64
import json
import requests


class FaceQueryDemo(object):
    def __init__(self, url, query_image_full_path):
        self._url = url
        self._query_image_full_path = query_image_full_path

    @staticmethod
    def get_base64_encoding(test_image_full_path):
        with open(test_image_full_path, "rb") as f:
            data = f.read()
            image_base64_enc = base64.b64encode(data)
            image_base64_enc = str(image_base64_enc, 'utf-8')

        return image_base64_enc

    def run(self):
        # Case 1: From local image path
        image_base64_enc = self.get_base64_encoding(test_image_full_path=self._query_image_full_path)

        # Case 2: From memory, should be RGB base64 encoding
        # pass

        request_body = {
            "activity_id": "Activity_2021_08_29_BJ001",
            "image_base64_enc": image_base64_enc,
            "distance_threshold": 0.52
        }

        response = requests.post(self._url, data=json.dumps(request_body))
        print('Response = {}'.format(response.text))


if __name__ == '__main__':
    demo = FaceQueryDemo(
        url="https://your_api_gateway_endpoint_url/query",
        query_image_full_path="./test_imgs/query.jpeg"
    )
    demo.run()
