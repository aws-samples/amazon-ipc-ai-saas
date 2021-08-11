import cv2
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Activation, Input, Add, MaxPooling2D, Flatten, Dense, Dropout


class DeepIDFeatureExtractor(object):
    def __init__(self, model_path='/opt/models/deepid_keras_weights.h5'):
        self._model_path = model_path
        self._deep_id = self.load_model()

    def load_model(self):
        image_input = Input(shape=(55, 47, 3))

        x = Conv2D(20, (4, 4), name='Conv1', activation='relu', input_shape=(55, 47, 3))(image_input)
        x = MaxPooling2D(pool_size=2, strides=2, name='Pool1')(x)
        x = Dropout(rate=0.99, name='D1')(x)

        x = Conv2D(40, (3, 3), name='Conv2', activation='relu')(x)
        x = MaxPooling2D(pool_size=2, strides=2, name='Pool2')(x)
        x = Dropout(rate=0.99, name='D2')(x)

        x = Conv2D(60, (3, 3), name='Conv3', activation='relu')(x)
        x = MaxPooling2D(pool_size=2, strides=2, name='Pool3')(x)
        x = Dropout(rate=0.99, name='D3')(x)

        x1 = Flatten()(x)
        fc11 = Dense(160, name='fc11')(x1)

        x2 = Conv2D(80, (2, 2), name='Conv4', activation='relu')(x)
        x2 = Flatten()(x2)
        fc12 = Dense(160, name='fc12')(x2)

        y = Add()([fc11, fc12])
        y = Activation('relu', name='deepid')(y)

        model = Model(inputs=[image_input], outputs=y)

        model.load_weights(self._model_path)
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
        embedding = self._deep_id.predict(detected_face)[0].tolist()
        return embedding

    @staticmethod
    def preprocess(face_img):
        """
        preprocess face image including image resize and padding, normalization

        :param face_img: a face area image, shape is (height, width, 3)
        :return:
        """
        target_size = (55, 47)        # (height, width)

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
        face_batch_data /= 255.0

        # change channels order from BGR to RGB
        face_batch_data = face_batch_data[..., ::-1]

        return face_batch_data
