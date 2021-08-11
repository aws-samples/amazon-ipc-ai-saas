import numpy as np
import cv2
from tensorflow.python.keras.engine import training
import tensorflow
from tensorflow import keras


class ArcFaceFeatureExtractor(object):
    def __init__(self, model_path='/opt/models/arcface_weights.h5'):
        self._model_path = model_path
        self._arc_face = self.load_model()

    def load_model(self):
        base_model = ArcFaceFeatureExtractor.ResNet34()
        inputs = base_model.inputs[0]
        arcface_model = base_model.outputs[0]
        arcface_model = keras.layers.BatchNormalization(momentum=0.9, epsilon=2e-5)(arcface_model)
        arcface_model = keras.layers.Dropout(0.4)(arcface_model)
        arcface_model = keras.layers.Flatten()(arcface_model)
        arcface_model = keras.layers.Dense(512, activation=None, use_bias=True, kernel_initializer="glorot_normal")(
            arcface_model)
        embedding = keras.layers.BatchNormalization(momentum=0.9, epsilon=2e-5, name="embedding", scale=True)(arcface_model)
        model = keras.models.Model(inputs, embedding, name=base_model.name)

        model.load_weights(self._model_path)
        return model

    @staticmethod
    def ResNet34():
        img_input = tensorflow.keras.layers.Input(shape=(112, 112, 3))

        x = tensorflow.keras.layers.ZeroPadding2D(padding=1, name='conv1_pad')(img_input)
        x = tensorflow.keras.layers.Conv2D(64, 3, strides=1, use_bias=False, kernel_initializer='glorot_normal',
                                           name='conv1_conv')(x)
        x = tensorflow.keras.layers.BatchNormalization(axis=3, epsilon=2e-5, momentum=0.9, name='conv1_bn')(x)
        x = tensorflow.keras.layers.PReLU(shared_axes=[1, 2], name='conv1_prelu')(x)
        x = ArcFaceFeatureExtractor.stack_fn(x)

        model = training.Model(img_input, x, name='ResNet34')

        return model

    @staticmethod
    def block1(x, filters, kernel_size=3, stride=1, conv_shortcut=True, name=None):
        bn_axis = 3

        if conv_shortcut:
            shortcut = tensorflow.keras.layers.Conv2D(filters, 1, strides=stride, use_bias=False,
                                                      kernel_initializer='glorot_normal', name=name + '_0_conv')(x)
            shortcut = tensorflow.keras.layers.BatchNormalization(axis=bn_axis, epsilon=2e-5, momentum=0.9,
                                                                  name=name + '_0_bn')(shortcut)
        else:
            shortcut = x

        x = tensorflow.keras.layers.BatchNormalization(axis=bn_axis, epsilon=2e-5, momentum=0.9, name=name + '_1_bn')(x)
        x = tensorflow.keras.layers.ZeroPadding2D(padding=1, name=name + '_1_pad')(x)
        x = tensorflow.keras.layers.Conv2D(filters, 3, strides=1, kernel_initializer='glorot_normal', use_bias=False,
                                           name=name + '_1_conv')(x)
        x = tensorflow.keras.layers.BatchNormalization(axis=bn_axis, epsilon=2e-5, momentum=0.9, name=name + '_2_bn')(x)
        x = tensorflow.keras.layers.PReLU(shared_axes=[1, 2], name=name + '_1_prelu')(x)

        x = tensorflow.keras.layers.ZeroPadding2D(padding=1, name=name + '_2_pad')(x)
        x = tensorflow.keras.layers.Conv2D(filters, kernel_size, strides=stride, kernel_initializer='glorot_normal',
                                           use_bias=False, name=name + '_2_conv')(x)
        x = tensorflow.keras.layers.BatchNormalization(axis=bn_axis, epsilon=2e-5, momentum=0.9, name=name + '_3_bn')(x)

        x = tensorflow.keras.layers.Add(name=name + '_add')([shortcut, x])
        return x

    @staticmethod
    def stack1(x, filters, blocks, stride1=2, name=None):
        x = ArcFaceFeatureExtractor.block1(x, filters, stride=stride1, name=name + '_block1')
        for i in range(2, blocks + 1):
            x = ArcFaceFeatureExtractor.block1(x, filters, conv_shortcut=False, name=name + '_block' + str(i))
        return x

    @staticmethod
    def stack_fn(x):
        x = ArcFaceFeatureExtractor.stack1(x, 64, 3, name='conv2')
        x = ArcFaceFeatureExtractor.stack1(x, 128, 4, name='conv3')
        x = ArcFaceFeatureExtractor.stack1(x, 256, 6, name='conv4')
        return ArcFaceFeatureExtractor.stack1(x, 512, 3, name='conv5')

    def represent(self, detected_face):
        """
        represent the detected and aligned face

        :param detected_face: face (numpy array format) after alignment
        :return: face embedding features
        """
        # resize to network's input size, and perform normalization
        detected_face = self.preprocess(face_img=detected_face)

        # represent
        embedding = self._arc_face.predict(detected_face)[0].tolist()
        return embedding

    @staticmethod
    def preprocess(face_img):
        """
        preprocess face image including image resize and padding, normalization

        :param face_img: a face area image, shape is (height, width, 3)
        :return:
        """
        target_size = (112, 112)        # (height, width)

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

        face_batch_data = np.expand_dims(img, axis=0)
        face_batch_data = face_batch_data.astype(np.float32)

        # normalization
        face_batch_data -= 127.5
        face_batch_data /= 128.0

        # change channels order from BGR to RGB
        face_batch_data = face_batch_data[..., ::-1]

        return face_batch_data
