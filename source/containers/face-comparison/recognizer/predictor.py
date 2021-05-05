from __future__ import print_function
import os
import flask
import json
import time
import mxnet as mx
import cv2
import base64
from face import Face
import numpy as np
import insightface
from insightface.utils import face_align


model_root_dir = '/opt/ml/model'

# 'retinaface_mnet025_v2+LResNet100E-IR'
# 'retinaface_mnet025_v2+LResNet50E-IR'
# 'retinaface_mnet025_v2+LResNet34E-IR'
# 'retinaface_mnet025_v2+MobileFaceNet'
# 'retinaface_r50_v1+LResNet100E-IR'
# 'retinaface_r50_v1+LResNet50E-IR'
# 'retinaface_r50_v1+LResNet34E-IR'
# 'retinaface_r50_v1+MobileFaceNet'
face_detection_and_comparison_model_name = os.environ.get('FACE_DETECTION_AND_COMPARISON_MODEL_NAME', 'retinaface_mnet025_v2+MobileFaceNet')

face_detection_model_name = face_detection_and_comparison_model_name.split('+')[0]
face_representation_model_name = face_detection_and_comparison_model_name.split('+')[1]


# A singleton for holding the model. This simply loads the model and holds it.
# It has a predict function that does a prediction based on the model and the input data.
class FaceRecognizerService(object):
    # class attributes
    face_detector = None
    face_embedding_model = None

    ctx = mx.cpu() if mx.context.num_gpus() == 0 else mx.gpu()

    # face representation configuration
    face_size = (112, 112)
    if face_representation_model_name == 'LResNet100E-IR':
        face_representation_model_prefix = os.path.join(model_root_dir, 'model-r100-ii/model')
    elif face_representation_model_name == 'LResNet50E-IR':
        face_representation_model_prefix = os.path.join(model_root_dir, 'model-r50-am-lfw/model')
    elif face_representation_model_name == 'LResNet34E-IR':
        face_representation_model_prefix = os.path.join(model_root_dir, 'model-r34-amf/model')
    elif face_representation_model_name == 'MobileFaceNet':
        face_representation_model_prefix = os.path.join(model_root_dir, 'model-y1-test2/model')
    else:
        face_representation_model_prefix = 'None'

    @classmethod
    def get_model(cls):
        """
        Get the face detector and face representation model for this instance, loading it if it's not already loaded.

        :return:
        """
        # face detector model
        if cls.face_detector is None:
            cls.face_detector = insightface.model_zoo.get_model(
                name=face_detection_model_name,
                root=model_root_dir
            )
            ctx_id = -1 if mx.context.num_gpus() == 0 else 0
            cls.face_detector.prepare(ctx_id=ctx_id)

        # face representation (embedding vector representation) model
        if cls.face_embedding_model is None:
            sym, arg_params, aux_params = mx.model.load_checkpoint(
                prefix=cls.face_representation_model_prefix, epoch=0)
            all_layers = sym.get_internals()
            sym = all_layers['fc1_output']
            cls.face_embedding_model = mx.mod.Module(symbol=sym, context=cls.ctx, label_names=None)
            cls.face_embedding_model.bind(data_shapes=[('data', (1, 3, cls.face_size[0], cls.face_size[1]))])
            cls.face_embedding_model.set_params(arg_params, aux_params)

        return cls.face_detector, cls.face_embedding_model

    @classmethod
    def get_largest_face(cls, bbox_list):
        largest_area_index = 0
        largest_area = -1.0
        for i in range(bbox_list.shape[0]):
            bbox = bbox_list[i]
            area = (bbox[0] - bbox[2]) * (bbox[1] - bbox[3])
            if largest_area < area:
                largest_area = area
                largest_area_index = i

        return largest_area_index

    @classmethod
    def detect_and_align(cls, raw_input_image, is_source_image=False, threshold=0.70):
        face_detector, _ = cls.get_model()

        height, width, _ = raw_input_image.shape
        short_size = height if height < width else width
        scale = 1.0 if short_size < 480.0 else 480.0 / short_size

        bbox_list, pts5_list = face_detector.detect(raw_input_image, threshold=threshold, scale=scale)

        if bbox_list.shape[0] == 0:
            return None

        if is_source_image:
            max_face_index = cls.get_largest_face(bbox_list)
            bbox = bbox_list[max_face_index, :]
            pts5 = pts5_list[max_face_index, :]
            aligned_source_face = face_align.norm_crop(raw_input_image, pts5)

            face = Face(
                bbox=[float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])],
                aligned_face_img=aligned_source_face,
                confidence=float(bbox[-1]),
                key_points={
                    'eyeLeft': [float(pts5[0][0]), float(pts5[0][1])],
                    'eyeRight': [float(pts5[1][0]), float(pts5[1][1])],
                    'nose': [float(pts5[2][0]), float(pts5[2][1])],
                    'mouthLeft': [float(pts5[3][0]), float(pts5[3][1])],
                    'mouthRight': [float(pts5[4][0]), float(pts5[4][1])],
                }
            )

            return face
        else:
            face_list = list()
            for index in range(len(bbox_list)):
                bbox = bbox_list[index]
                pts5 = pts5_list[index]

                aligned_target_face = face_align.norm_crop(raw_input_image, pts5)

                face = Face(
                    bbox=[float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])],
                    aligned_face_img=aligned_target_face,
                    confidence=float(bbox[-1]),
                    key_points={
                        'eyeLeft': [float(pts5[0][0]), float(pts5[0][1])],
                        'eyeRight': [float(pts5[1][0]), float(pts5[1][1])],
                        'nose': [float(pts5[2][0]), float(pts5[2][1])],
                        'mouthLeft': [float(pts5[3][0]), float(pts5[3][1])],
                        'mouthRight': [float(pts5[4][0]), float(pts5[4][1])],
                    }
                )

                face_list.append(face)
            return face_list

    @classmethod
    def get_feature(cls, aligned):
        _, face_embedding_model = cls.get_model()

        a = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)
        a = np.transpose(a, (2, 0, 1))
        input_blob = np.expand_dims(a, axis=0)
        data = mx.nd.array(input_blob)
        db = mx.io.DataBatch(data=(data, ))
        face_embedding_model.forward(db, is_train=False)
        emb = face_embedding_model.get_outputs()[0].asnumpy()[0]
        norm = np.sqrt(np.sum(emb * emb) + 0.00001)
        emb /= norm
        return emb

    @classmethod
    def predict(cls, source_image_base64, target_image_base64, min_confidence_thresh=0.40):
        source_image = cv2.imdecode(np.frombuffer(base64.b64decode(source_image_base64), np.uint8), cv2.IMREAD_COLOR)
        target_image = cv2.imdecode(np.frombuffer(base64.b64decode(target_image_base64), np.uint8), cv2.IMREAD_COLOR)

        t1 = time.time()
        source_detected_face = cls.detect_and_align(source_image, is_source_image=True, threshold=min_confidence_thresh)
        target_detected_faces = cls.detect_and_align(target_image, is_source_image=False, threshold=min_confidence_thresh)
        t2 = time.time()
        print('Time Cost of Face Detecting & Aligning for 2 Images = {} seconds'.format(t2 - t1))

        response = {
            'SourceImageFace': None,
            'FaceMatches': []
        }

        if source_detected_face is not None:
            [x_min, y_min, x_max, y_max] = source_detected_face.bbox
            response['SourceImageFace'] = {
                'BoundingBox': [x_min, y_min, x_max, y_max],
                'Confidence': source_detected_face.confidence,
                'KeyPoints': source_detected_face.key_points
            }
        else:
            return response

        for target_comp_face in target_detected_faces:
            base_feat_representation = cls.get_feature(source_detected_face.aligned_face_img)
            target_feat_representation = cls.get_feature(target_comp_face.aligned_face_img)
            similarity_score = np.dot(base_feat_representation, target_feat_representation)

            # add comparison to response body
            [x_min, y_min, x_max, y_max] = target_comp_face.bbox
            response['FaceMatches'].append({
                'Similarity': float(similarity_score),
                'Face': {
                    'BoundingBox': [x_min, y_min, x_max, y_max],
                    'Confidence': target_comp_face.confidence,
                    'KeyPoints': target_comp_face.key_points
                }
            })

        return response


# The flask app for serving predictions
app = flask.Flask(__name__)


@app.route('/ping', methods=['GET'])
def ping():
    """
    Determine if the container is working and healthy. In this sample container, we declare
    it healthy if we can load the model successfully.

    :return:
    """
    detector_mode, embedding_model = FaceRecognizerService.get_model()
    health = (detector_mode is not None) and (embedding_model is not None)
    status = 200 if health else 404
    return flask.Response(response='\n', status=status, mimetype='application/json')


@app.route('/invocations', methods=['POST'])
def transformation():
    """
    Do an inference on a single batch of data. In this sample server, we take image data as base64 formation,
    decode it for internal use and then convert the predictions to json format

    :return:
    """
    t_start = time.time()

    if flask.request.content_type == 'application/json':
        request_body = flask.request.data.decode('utf-8')
        request_body = json.loads(request_body)
        source_image_base64 = request_body['source_image_bytes']
        target_image_base64 = request_body['target_image_bytes']
    else:
        return flask.Response(
            response='Face comparison only supports application/json data',
            status=415,
            mimetype='text/plain')

    # inference
    body = FaceRecognizerService.predict(
        source_image_base64,
        target_image_base64,
        min_confidence_thresh=0.65
    )

    t_end = time.time()
    print('Time consumption = {} second'.format(t_end - t_start))
    print('Response = {}'.format(body))

    return flask.Response(response=json.dumps(body), status=200, mimetype='application/json')
