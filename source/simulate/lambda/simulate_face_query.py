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
        image_base64_enc = self.get_base64_encoding(test_image_full_path=self._query_image_full_path)

        request_body = {
            "activity_id": "Activity_2021_08_29_BJ001",
            "image_base64_enc": image_base64_enc
        }

        response = requests.post(self._url, data=json.dumps(request_body))
        print('Response = {}'.format(response.text))


if __name__ == '__main__':
    demo = FaceQueryDemo(
        url="https://8bfha86lq8.execute-api.cn-northwest-1.amazonaws.com.cn/prod/query",
        query_image_full_path="./test_imgs/query.jpeg"
    )
    demo.run()
