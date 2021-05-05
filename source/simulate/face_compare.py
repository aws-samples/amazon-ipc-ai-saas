import base64
import os
import json
import time
import cv2
import matplotlib.patches as patches
from matplotlib import pyplot as plt
import requests


class FaceCompareSimulator(object):
    def __init__(self, face_compare_pairs_root_dir):
        self._face_compare_url = "https://facecomp.gaowexu.solutions.aws.a2z.org.cn/inference"

        self._face_compare_pairs_root_dir = face_compare_pairs_root_dir

    @staticmethod
    def get_base64_encoding(full_path):
        with open(full_path, "rb") as f:
            data = f.read()
            image_base64_enc = base64.b64encode(data)
            image_base64_enc = str(image_base64_enc, 'utf-8')

        return image_base64_enc

    @staticmethod
    def visualize(image_1, image_2, response):
        fig, (ax1, ax2) = plt.subplots(2, 1)
        source_face_bbox = response['SourceImageFace']['BoundingBox']
        source_face_confidence = response['SourceImageFace']['Confidence']
        source_face_keypoints = response['SourceImageFace']['KeyPoints']

        source_eye_left = source_face_keypoints['eyeLeft']
        source_eye_right = source_face_keypoints['eyeRight']
        source_mouth_left = source_face_keypoints['mouthLeft']
        source_mouth_right = source_face_keypoints['mouthRight']
        source_nose = source_face_keypoints['nose']

        ax1.plot(source_eye_left[0], source_eye_left[1], 'r.')
        ax1.plot(source_eye_right[0], source_eye_right[1], 'b.')
        ax1.plot(source_nose[0], source_nose[1], 'g.')
        ax1.plot(source_mouth_left[0], source_mouth_left[1], 'k.')
        ax1.plot(source_mouth_right[0], source_mouth_right[1], 'y.')

        ax1.imshow(image_1[:, :, ::-1])
        rect = patches.Rectangle((source_face_bbox[0], source_face_bbox[1]),
                                 source_face_bbox[2] - source_face_bbox[0],
                                 source_face_bbox[3] - source_face_bbox[1], linewidth=1, edgecolor='r',
                                 facecolor='none')
        ax1.add_patch(rect)
        ax1.text(source_face_bbox[0], source_face_bbox[1],
                 'confidence = {}'.format(round(source_face_confidence, 2)), color='red', fontsize=8)

        ax2.imshow(image_2[:, :, ::-1])

        face_matches = response['FaceMatches']
        for match_face in face_matches:
            similarity_score = match_face['Similarity']
            face_bbox = match_face['Face']['BoundingBox']
            face_confidence = match_face['Face']['Confidence']
            face_keypoints = match_face['Face']['KeyPoints']

            eye_left = face_keypoints['eyeLeft']
            eye_right = face_keypoints['eyeRight']
            mouth_left = face_keypoints['mouthLeft']
            mouth_right = face_keypoints['mouthRight']
            nose = face_keypoints['nose']

            ax2.plot(eye_left[0], eye_left[1], 'r.')
            ax2.plot(eye_right[0], eye_right[1], 'b.')
            ax2.plot(nose[0], nose[1], 'g.')
            ax2.plot(mouth_left[0], mouth_left[1], 'k.')
            ax2.plot(mouth_right[0], mouth_right[1], 'y.')

            ax2.imshow(image_2[:, :, ::-1])
            rect = patches.Rectangle((face_bbox[0], face_bbox[1]), face_bbox[2] - face_bbox[0],
                                     face_bbox[3] - face_bbox[1], linewidth=1, edgecolor='r',
                                     facecolor='none')
            ax2.add_patch(rect)
            ax2.text(face_bbox[0], face_bbox[1], 'confidence = {}'.format(round(face_confidence, 2)),
                     color='red', fontsize=8)
            ax2.text(face_bbox[2], face_bbox[3], 'similarity = {}'.format(round(similarity_score, 2)),
                     color='blue', fontsize=8)

        plt.show()

    def compare(self):
        image_pair_names = [
            ['test_1_source.jpg', 'test_1_target.jpg'],
        ]

        for [source_image_name, target_image_name] in image_pair_names:
            source_full_path = os.path.join(self._face_compare_pairs_root_dir, source_image_name)
            target_full_path = os.path.join(self._face_compare_pairs_root_dir, target_image_name)
            print('Test image {} (source), {} (target):'.format(source_full_path, target_full_path))

            # Step 1: read image and execute base64 encoding
            source_image_base64_enc = self.get_base64_encoding(source_full_path)
            target_image_base64_enc = self.get_base64_encoding(target_full_path)

            # Step 2: send request to backend
            request_body = {
                "timestamp": str(time.time()),
                "request_id": 1242322,
                "source_image": source_image_base64_enc,
                "target_image": target_image_base64_enc
            }

            t1 = time.time()
            response = requests.post(self._face_compare_url, data=json.dumps(request_body))
            t2 = time.time()
            print('Time cost = {}'.format(t2 - t1))

            # Step 3: visualization
            response = json.loads(response.text)
            print('Response = {}'.format(response))

            image_1 = cv2.imread(source_full_path, cv2.IMREAD_COLOR)
            image_2 = cv2.imread(target_full_path, cv2.IMREAD_COLOR)
            self.visualize(image_1, image_2, response)


if __name__ == '__main__':
    simulator = FaceCompareSimulator(
        face_compare_pairs_root_dir='./facepairs',
    )

    simulator.compare()
