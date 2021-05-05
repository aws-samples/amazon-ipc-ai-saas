#!/usr/bin/python3

class Face(object):
    def __init__(self, bbox, aligned_face_img, confidence, key_points):
        self._bbox = bbox           # [x_min, y_min, x_max, y_max]
        self._aligned_face_img = aligned_face_img
        self._confidence = confidence
        self._key_points = key_points

    @property
    def bbox(self):
        return self._bbox

    @property
    def confidence(self):
        return self._confidence

    @property
    def key_points(self):
        return self._key_points

    @property
    def aligned_face_img(self):
        return self._aligned_face_img
