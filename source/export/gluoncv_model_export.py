from gluoncv import model_zoo, data, utils
from matplotlib import pyplot as plt
from mxnet import gluon
import mxnet as mx
import numpy as np
from gluoncv.utils import export_block
import time
import os
import cv2


class ModelExportBlock(object):
    """
    Person (Body) Detection Export Class

    Reference:
    SSD: https://cv.gluon.ai/build/examples_detection/demo_ssd.html#sphx-glr-build-examples-detection-demo-ssd-py
    YOLOv3: https://cv.gluon.ai/build/examples_detection/demo_yolo.html
    Faster RCNN: https://cv.gluon.ai/build/examples_detection/demo_faster_rcnn.html
    """
    def __init__(self, model_names_list):
        """
        constructor

        :param model_names_list: list of model names
        """
        self._model_names_list = model_names_list
        self._ctx = mx.cpu() if mx.context.num_gpus() == 0 else mx.gpu()

        self._short_size_mapping = {
            'yolo3_darknet53_coco': 416,
            'yolo3_mobilenet1.0_coco': 416,
            'ssd_512_resnet50_v1_coco': 512,
            'faster_rcnn_fpn_resnet101_v1d_coco': 600,
        }

    def export(self):
        for model_name in self._model_names_list:
            net = model_zoo.get_model(model_name, pretrained=True)
            classes = ['person']        # only one foreground class here
            net.reset_class(
                classes,
                reuse_weights=['person']
            )

            net.hybridize()
            net.collect_params().reset_ctx(mx.cpu())
            net.forward(x=mx.nd.zeros((1, 3, 600, 600)))
            net.export(model_name)

            print('Export params and symbol of {}...'.format(model_name))

    @staticmethod
    def resize(image, short_size):
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        resized_img = mx.image.resize_short(image, size=short_size)
        resized_rescaled_img = mx.nd.image.to_tensor(resized_img)
        resized_rescaled_normalized_img = mx.nd.image.normalize(resized_rescaled_img, mean=mean, std=std)
        resized_rescaled_normalized_img = resized_rescaled_normalized_img.expand_dims(0)
        return resized_rescaled_normalized_img

    def forward_test(self, model_name, image_full_path, iter_times):
        """
        inference test for given model

        :param model_name: model name which is expected to be test
        :param image_full_path: path of test image
        :param iter_times: iteration times for evaluating inference time cost
        :return:
        """
        detector = gluon.nn.SymbolBlock.imports(
            symbol_file=os.path.join('{}-symbol.json'.format(model_name)),
            input_names=['data'],
            param_file=os.path.join('{}-0000.params'.format(model_name)),
            ctx=self._ctx
        )

        image = mx.image.imread(image_full_path)
        height, width, channels = image.shape

        t_start = time.time()
        for _ in range(iter_times):
            short_size = self._short_size_mapping[model_name]
            normalized_image = self.resize(image=image, short_size=short_size)
            if mx.context.num_gpus() != 0:
                normalized_image = normalized_image.copyto(mx.gpu())

            # inference
            mx_ids, mx_scores, mx_bounding_boxes = detector(normalized_image)

            # post-process
            mx_ids = mx_ids.asnumpy()
            mx_scores = mx_scores.asnumpy()
            mx_bounding_boxes = mx_bounding_boxes.asnumpy()

            # resize detection results back to original image size
            scale_ratio = short_size / height if height < width else short_size / width
            bbox_coords, bbox_scores, class_ids = list(), list(), list()
            for index, bbox in enumerate(mx_bounding_boxes[0]):
                prob = float(mx_scores[0][index][0])
                if prob < 0.0:
                    continue

                [x_min, y_min, x_max, y_max] = bbox
                x_min = int(x_min / scale_ratio)
                y_min = int(y_min / scale_ratio)
                x_max = int(x_max / scale_ratio)
                y_max = int(y_max / scale_ratio)
                bbox_coords.append([x_min, y_min, x_max, y_max])
                bbox_scores.append([prob])
                class_ids.append([mx_ids[0][index][0]])

        t_end = time.time()

        bbox_coords = np.array(bbox_coords)
        bbox_scores = np.array(bbox_scores)
        class_ids = np.array(class_ids)

        print('Mode = {}: Inference average time cost = {} seconds'.format(model_name, (t_end - t_start)/iter_times))

        image = cv2.imread(image_full_path, cv2.IMREAD_COLOR)
        image = image[:, :, ::-1]
        ax = utils.viz.plot_bbox(image, bbox_coords, bbox_scores, class_ids, class_names=['Person'], thresh=0.5)
        plt.axis('off')
        plt.show()


if __name__ == '__main__':
    export_handler = ModelExportBlock(
        model_names_list=[
            'ssd_512_resnet50_v1_coco',
            'yolo3_darknet53_coco',
            'yolo3_mobilenet1.0_coco',
            'faster_rcnn_fpn_resnet101_v1d_coco'
        ])

    # export_handler.export()

    export_handler.forward_test(
        model_name='ssd_512_resnet50_v1_coco',
        image_full_path='./street_small.jpg',
        iter_times=10)

    export_handler.forward_test(
        model_name='yolo3_darknet53_coco',
        image_full_path='./street_small.jpg',
        iter_times=10)

    export_handler.forward_test(
        model_name='yolo3_mobilenet1.0_coco',
        image_full_path='./street_small.jpg',
        iter_times=10)

    export_handler.forward_test(
        model_name='faster_rcnn_fpn_resnet101_v1d_coco',
        image_full_path='./street_small.jpg',
        iter_times=10)
