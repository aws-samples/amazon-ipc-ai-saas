import cv2
import numpy as np
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Convolution2D, ZeroPadding2D, MaxPooling2D, Flatten, Dropout, Activation


class VGGFaceFeatureExtractor(object):
    def __init__(self, model_path='/opt/models/vgg_face_weights.h5'):
        self._model = None
        self._model_path = model_path
        self._vgg_face = self.load_model(model_path=self._model_path)

    def load_model(self, model_path):
        self._model = VGGFaceFeatureExtractor.base_model()
        self._model.load_weights(model_path)
        vgg_face_descriptor = Model(inputs=self._model.layers[0].input, outputs=self._model.layers[-2].output)
        return vgg_face_descriptor

    @staticmethod
    def base_model():
        model = Sequential()
        model.add(ZeroPadding2D((1, 1), input_shape=(224, 224, 3)))
        model.add(Convolution2D(64, (3, 3), activation='relu'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(128, (3, 3), activation='relu'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(128, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(256, (3, 3), activation='relu'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(256, (3, 3), activation='relu'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(256, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, (3, 3), activation='relu'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, (3, 3), activation='relu'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, (3, 3), activation='relu'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, (3, 3), activation='relu'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        model.add(Convolution2D(4096, (7, 7), activation='relu'))
        model.add(Dropout(0.5))
        model.add(Convolution2D(4096, (1, 1), activation='relu'))
        model.add(Dropout(0.5))
        model.add(Convolution2D(2622, (1, 1)))
        model.add(Flatten())
        model.add(Activation('softmax'))

        return model

    def represent(self, detected_face):
        """
        represent the detected and aligned face

        :param detected_face: face (numpy array format) after alignment
        :return: face embedding features
        """
        # resize to network's input size, and perform normalization
        detected_face = self.preprocess(face_img=detected_face)

        # represent
        embedding = self._vgg_face.predict(detected_face)[0].tolist()
        return embedding

    @staticmethod
    def preprocess(face_img):
        """
        preprocess face image including image resize and padding, normalization

        :param face_img: a face area image, shape is (height, width, 3)
        :return:
        """
        target_size = (224, 224)        # (height, width)

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
        face_batch_data[..., 0] -= 93.5940          # B
        face_batch_data[..., 1] -= 104.7624         # G
        face_batch_data[..., 2] -= 129.1863         # R

        # change channels order from BGR to RGB
        face_batch_data = face_batch_data[..., ::-1]

        return face_batch_data


if __name__ == '__main__':
    face_feat_extractor = VGGFaceFeatureExtractor()
    print(face_feat_extractor.find_input_shape())
