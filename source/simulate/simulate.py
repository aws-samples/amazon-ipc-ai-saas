import cv2
from detectors.retinaface_detector import RetinaFaceDetector
from detectors.mtcnn_detector import MTCNNDetector

from basemodels.vgg_face import VGGFaceFeatureExtractor
from basemodels.facenet import FacenetFeatureExtractor
from basemodels.facenet512 import Facenet512FeatureExtractor
from basemodels.fb_deepface import FbDeepFaceFeatureExtractor
from basemodels.deep_id import DeepIDFeatureExtractor
from basemodels.arc_face import ArcFaceFeatureExtractor

import time
import matplotlib.pyplot as plt


# DO NOT MODIFY
DETECTOR_BACKENDS_MAPPING = {
    'mtcnn': MTCNNDetector,
    'retinaface': RetinaFaceDetector
}

# DO NOT MODIFY
REPRESENT_MODELS_MAPPING = {
    'VGG-Face': VGGFaceFeatureExtractor,
    'Facenet': FacenetFeatureExtractor,
    'Facenet512': Facenet512FeatureExtractor,
    'DeepFace': FbDeepFaceFeatureExtractor,
    'DeepID': DeepIDFeatureExtractor,
    'ArcFace': ArcFaceFeatureExtractor
}


DETECTOR_BACKEND = "retinaface"
REPRESENT_MODEL = "Facenet"
FACE_RECOGNITION_THRESHOLD = 0.90


class FaceDetectAndRepresentProcessor(object):
    face_detector = None
    face_features_extractor = None
    face_properties_analyzer = None

    @classmethod
    def load_model(cls):
        # load face detector
        face_detector = DETECTOR_BACKENDS_MAPPING.get(DETECTOR_BACKEND)
        cls.face_detector = face_detector()

        # load face embedding vector extractor
        face_features_extractor = REPRESENT_MODELS_MAPPING.get(REPRESENT_MODEL)
        cls.face_features_extractor = face_features_extractor()

        return cls.face_detector, cls.face_features_extractor

    @classmethod
    def predict(cls, image_data):
        face_detector, face_features_extractor = cls.load_model()

        iter_times = 100

        t1 = time.time()
        for i in range(iter_times):
            # Step 1: detect and align faces in input image
            detected_and_aligned_faces = face_detector.detect_face(image_data, align=True)

            for face_obj in detected_and_aligned_faces:
                aligned_face_image = face_obj['detected_face']      # BGR
                confidence = face_obj['confidence']
                landmarks = face_obj['landmarks']

                if confidence < FACE_RECOGNITION_THRESHOLD:
                    continue

                # Step 2: face embedding feature vector extraction
                face_embedding_vectors = face_features_extractor.represent(aligned_face_image)
                print('len(face_embedding_vectors) = {}'.format(len(face_embedding_vectors)))
                print('face_embedding_vectors = {}'.format(face_embedding_vectors))

        t2 = time.time()
        time_cost = t2 - t1
        print('Time cost = {} seconds'.format(time_cost / iter_times))

        # return detected_and_aligned_faces, face_embedding_vectors


if __name__ == '__main__':
    processor = FaceDetectAndRepresentProcessor()
    image = cv2.imread('./test_images/test_1_source.jpg', cv2.IMREAD_COLOR)
    processor.predict(image_data=image)










