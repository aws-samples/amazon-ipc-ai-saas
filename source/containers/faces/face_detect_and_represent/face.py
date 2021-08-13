#!/usr/bin/python3


class Face(object):
    def __init__(self, bbox, aligned_face_data, confidence, landmarks, representation=None):
        self._bbox = bbox
        self._aligned_face_data = aligned_face_data
        self._confidence = confidence
        self._landmarks = landmarks
        self._representation = representation

    @property
    def bbox(self):
        return self._bbox

    @property
    def confidence(self):
        return self._confidence

    @property
    def landmarks(self):
        return self._landmarks

    @property
    def aligned_face_data(self):
        return self._aligned_face_data

    @property
    def representation(self):
        return self._representation.tolist()

    @representation.setter
    def representation(self, embedding_vector):
        self._representation = embedding_vector
