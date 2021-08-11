import cv2
import numpy as np
import math
from mtcnn import MTCNN
from PIL import Image


class MTCNNDetector(object):
    def __init__(self):
        self._mtcnn_detector = MTCNN(weights_file="/opt/models/mtcnn_weights.npy")

    @staticmethod
    def find_euclidean_distance(source_representation, test_representation):
        if type(source_representation) is list:
            source_representation = np.array(source_representation)

        if type(test_representation) is list:
            test_representation = np.array(test_representation)

        euclidean_distance = source_representation - test_representation
        euclidean_distance = np.sum(np.multiply(euclidean_distance, euclidean_distance))
        euclidean_distance = np.sqrt(euclidean_distance)
        return euclidean_distance

    @staticmethod
    def alignment_procedure(img, left_eye, right_eye):
        """
        align given face in img based on left and right eye coordinates

        :param img: image data
        :param left_eye: coordinate of left eye
        :param right_eye: coordinate of right eye
        :return: aligned image
        """
        left_eye_x, left_eye_y = left_eye
        right_eye_x, right_eye_y = right_eye

        if left_eye_y > right_eye_y:
            point_3rd = (right_eye_x, left_eye_y)
            direction = -1          # rotate same direction to clock
        else:
            point_3rd = (left_eye_x, right_eye_y)
            direction = 1           # rotate inverse direction of clock

        a = MTCNNDetector.find_euclidean_distance(np.array(left_eye), np.array(point_3rd))
        b = MTCNNDetector.find_euclidean_distance(np.array(right_eye), np.array(point_3rd))
        c = MTCNNDetector.find_euclidean_distance(np.array(right_eye), np.array(left_eye))

        # apply cosine rule
        # this multiplication causes division by zero in cos_a calculation
        if (b - 0.0) < 1e-7 or (c - 0.0) < 1e-7:
            return img

        cos_a = (b * b + c * c - a * a) / (2 * b * c)
        angle = np.arccos(cos_a)            # angle in radian
        angle = (angle * 180) / math.pi     # radian to degree

        # rotate base image
        if direction == -1:
            angle = 90 - angle

        img = Image.fromarray(img)
        img = np.array(img.rotate(direction * angle))

        return img

    def detect_face(self, img, align=True):
        # MTCNN expects RGB but OpenCV read BGR
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        detections = self._mtcnn_detector.detect_faces(img_rgb)

        resp = list()
        for index, detection in enumerate(detections):
            x, y, w, h = detection["box"]
            confidence = detection["confidence"]
            landmarks = detection["keypoints"]
            detected_face = img[int(y):int(y + h), int(x):int(x + w)]
            img_region = [x, y, w, h]

            if align:
                left_eye = landmarks["left_eye"]
                right_eye = landmarks["right_eye"]
                nose = landmarks["nose"]
                mouth_left = landmarks["mouth_left"]
                mouth_right = landmarks["mouth_right"]
                detected_face = self.alignment_procedure(detected_face, left_eye, right_eye)

            resp.append({
                "detected_face": detected_face,
                "confidence": confidence,
                "bounding_box": img_region,
                "landmarks": landmarks
            })

        return resp
