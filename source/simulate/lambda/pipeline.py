import cv2
import os
import base64
import requests
import json
import numpy as np
from gluoncv import utils
from matplotlib import pyplot as plt


class PipelineSimulator(object):
    """
    This simulator can do four steps:
    1. Parse a video and dump each frame (with interval configuration) into local disk
    2. Upload each frame to S3 (Backend will trigger lambda / SageMaker endpoint to detect and represent faces)
    3. Query all face clusters in all frames uploaded in Step 2.
    4. Query all faces based a given image.
    """
    def __init__(self, test_video_path, output_dir, base_url, distance_threshold):
        """
        Constructor

        :param test_video_path: we use a video as an activity scenario, all faces in this video will be regarded in the same activity
        :param output_dir: visualization root director
        :param base_url: API Gateway URL
        :param distance_threshold: distance threshold between faces pair, two faces will distance smaller than this threshold will be regarded as a same person
        """
        self._test_video_path = test_video_path
        self._output_dir = output_dir
        self._base_url = base_url
        self._distance_threshold = distance_threshold

        self._frames_root_dir = os.path.join(self._output_dir, 'frames')
        if not os.path.exists(self._frames_root_dir):
            os.makedirs(self._frames_root_dir)

    @staticmethod
    def get_base64_encoding(test_image_full_path):
        with open(test_image_full_path, "rb") as f:
            data = f.read()
            image_base64_enc = base64.b64encode(data)
            image_base64_enc = str(image_base64_enc, 'utf-8')

        return image_base64_enc

    def dump_frames(self, interval=800):
        """
        dump frame from the video with given interval

        :return:
        """
        cap = cv2.VideoCapture(self._test_video_path)

        frame_index = 1
        while cap.isOpened():
            ret, frame = cap.read()

            if ret is False:
                break

            if frame_index % interval == 0:
                save_full_path = os.path.join(self._frames_root_dir, 'frame_index_{}.jpg'.format(frame_index))
                cv2.imwrite(save_full_path, frame)
                print('Saving frame {} into {}...'.format(frame_index, save_full_path))
            frame_index += 1

        cap.release()
        cv2.destroyAllWindows()

    def send_frame_to_s3(self):
        """
        Upload images into S3 bucket

        :return:
        """
        images = [image_name for image_name in os.listdir(self._frames_root_dir) if image_name.endswith('.jpg')]
        images = sorted(images, key=lambda x: int(x.split('frame_index_')[1].split('.jpg')[0]))

        for image_name in images:
            image_id = image_name.split('.jpg')[0]

            test_image_full_path = os.path.join(self._frames_root_dir, image_name)
            image_base64_enc = self.get_base64_encoding(test_image_full_path=test_image_full_path)

            request_body = {
                "activity_id": "Activity_Test_001",
                "image_id": image_id,
                "image_base64_enc": image_base64_enc
            }

            response = requests.post(self._base_url + 'upload', data=json.dumps(request_body))
            print('Image Id = {}: response = {}'.format(image_id, response.text))

    def get_activity_summary(self):
        """
        Get all faces cluster and visualization

        :return:
        """
        request_body = {
            "activity_id": "Activity_Test_001",
            "distance_threshold": self._distance_threshold
        }

        response = requests.post(self._base_url + 'activity', data=json.dumps(request_body))
        summary = json.loads(response.text)

        for cluster_id, faces in summary.items():
            cluster_root_dir = os.path.join(self._output_dir, "clusters", "person_{}".format(cluster_id))
            if not os.path.exists(cluster_root_dir):
                os.makedirs(cluster_root_dir)

            for face in faces:
                image_id = face['image_id']
                bbox = face['bbox']
                confidence = face['confidence']
                face_id = face['face_id']
                landmarks = face['landmarks']

                # visualization
                self.plot_face(
                    image_id,
                    bbox,
                    confidence,
                    landmarks,
                    face_id,
                    dump_image_path=os.path.join(
                        cluster_root_dir,
                        "image_id_{}_face_id_{}.jpg".format(image_id, face_id)))
                print("Plotting face {} in cluster {}...".format(face_id, cluster_id))

    def plot_face(self, image_id, bbox, confidence, landmarks, face_id, dump_image_path):
        """
        Plot face in same clusters

        :param image_id:
        :param bbox:
        :param confidence:
        :param landmarks:
        :param face_id:
        :param dump_image_path:
        :return:
        """
        image_full_path = os.path.join(self._frames_root_dir, image_id + '.jpg')
        image = cv2.imread(image_full_path, cv2.IMREAD_COLOR)
        image = image[:, :, ::-1]
        bboxes = np.array([bbox])
        scores = np.array([confidence])

        ax = utils.viz.plot_bbox(image, bboxes, scores, thresh=0.25)
        plt.axis('off')
        plt.savefig(dump_image_path, dpi=200, bbox_inches="tight", pad_inches=0)
        plt.close()

    def query_face(self, query_face_image):
        """
        Query face based on a given query_face_image

        :param query_face_image: full path of query image
        :return:
        """
        image_base64_enc = self.get_base64_encoding(test_image_full_path=query_face_image)

        request_body = {
            "activity_id": "Activity_Test_001",
            "image_base64_enc": image_base64_enc,
            "distance_threshold": self._distance_threshold
        }

        response = requests.post(self._base_url + 'query', data=json.dumps(request_body))
        faces = json.loads(response.text)

        for face in faces:
            image_id = face['image_id']
            bbox = face['bbox']
            confidence = face['confidence']
            face_id = face['face_id']
            landmarks = face['landmarks']

            image_full_path = os.path.join(self._frames_root_dir, image_id + '.jpg')
            image = cv2.imread(image_full_path, cv2.IMREAD_COLOR)
            image = image[:, :, ::-1]
            bboxes = np.array([bbox])
            scores = np.array([confidence])

            ax = utils.viz.plot_bbox(image, bboxes, scores, thresh=0.25)
            plt.axis('off')
            plt.show()


if __name__ == '__main__':
    simulator = PipelineSimulator(
        test_video_path="your_test_video.mp4",
        output_dir="./activity/",
        base_url="https://your_api_gateway_url/prod/",
        distance_threshold=0.52
    )

    simulator.dump_frames(interval=800)
    simulator.send_frame_to_s3()
    simulator.get_activity_summary()
    simulator.query_face(query_face_image='your_query_image_full_path.jpg')

