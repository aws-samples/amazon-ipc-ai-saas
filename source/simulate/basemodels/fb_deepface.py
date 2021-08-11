import cv2
import numpy as np
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Convolution2D, LocallyConnected2D, MaxPooling2D, Flatten, Dense, Dropout


class FbDeepFaceFeatureExtractor(object):
    def __init__(self, model_path='/opt/models/VGGFace2_DeepFace_weights_val-0.9034.h5'):
        self._model_path = model_path
        self._deep_face = self.load_model()

    def load_model(self):
        base_model = Sequential()
        base_model.add(Convolution2D(32, (11, 11), activation='relu', name='C1', input_shape=(152, 152, 3)))
        base_model.add(MaxPooling2D(pool_size=3, strides=2, padding='same', name='M2'))
        base_model.add(Convolution2D(16, (9, 9), activation='relu', name='C3'))
        base_model.add(LocallyConnected2D(16, (9, 9), activation='relu', name='L4'))
        base_model.add(LocallyConnected2D(16, (7, 7), strides=2, activation='relu', name='L5'))
        base_model.add(LocallyConnected2D(16, (5, 5), activation='relu', name='L6'))
        base_model.add(Flatten(name='F0'))
        base_model.add(Dense(4096, activation='relu', name='F7'))
        base_model.add(Dropout(rate=0.5, name='D0'))
        base_model.add(Dense(8631, activation='softmax', name='F8'))

        base_model.load_weights(self._model_path)
        deepface_model = Model(inputs=base_model.layers[0].input, outputs=base_model.layers[-3].output)

        return deepface_model

    def represent(self, detected_face):
        """
        represent the detected and aligned face

        :param detected_face: face (numpy array format) after alignment
        :return: face embedding features
        """
        # resize to network's input size, and perform normalization
        detected_face = self.preprocess(face_img=detected_face)

        # represent
        embedding = self._deep_face.predict(detected_face)[0].tolist()
        return embedding

    @staticmethod
    def preprocess(face_img):
        """
        preprocess face image including image resize and padding, normalization

        :param face_img: a face area image, shape is (height, width, 3)
        :return:
        """
        target_size = (152, 152)        # (height, width)

        # resize image to expected shape
        factor_0 = float(target_size[0]) / face_img.shape[0]
        factor_1 = float(target_size[1]) / face_img.shape[1]
        factor = min(factor_0, factor_1)

        dsize = (int(face_img.shape[1] * factor), int(face_img.shape[0] * factor))
        img = cv2.resize(face_img, dsize)

        # pad the other side to the target size by adding black pixels
        diff_height = target_size[0] - img.shape[0]
        diff_width = target_size[1] - img.shape[1]
        img = np.pad(
            array=img,
            pad_width=(
                (diff_height // 2, diff_height - diff_height // 2),     # height
                (diff_width // 2, diff_width - diff_width // 2),        # width
                (0, 0)                                                  # channels
            )
        )

        # normalizing the image pixels
        face_batch_data = np.expand_dims(img, axis=0)
        face_batch_data = face_batch_data.astype(np.float32)

        # normalization
        face_batch_data[..., 0] -= 91.4953          # B
        face_batch_data[..., 1] -= 103.8827         # G
        face_batch_data[..., 2] -= 131.0912         # R

        # change channels order from BGR to RGB
        face_batch_data = face_batch_data[..., ::-1]

        return face_batch_data
