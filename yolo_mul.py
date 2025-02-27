import colorsys
import os
import time

import numpy as np
import torch
import torch.nn as nn
from PIL import ImageDraw, ImageFont

from nets.yolo_mul import YoloBody
from utils.utils import (cvtColor, get_classes, preprocess_input,
                         resize_image, show_config)
from utils.utils_bbox import DecodeBox


class YOLO(object):
    _defaults = {
        "model_path": r'D:\A_Python\yolov8_dd\logs\ours_sppf_cbam1_n_2024_12_19_23_01_48\best_epoch_weights.pth',  # Path to the model weights
        "classes_path": 'model_data/voc_classes.txt',  # Path to the classes file
        "input_shape": [640, 640],  # Input shape for the model
        "phi": 'n',  # Model variant
        "confidence": 0.5,  # Confidence threshold for detection
        "nms_iou": 0.3,  # IoU threshold for non-maximum suppression
        "letterbox_image": True,  # Whether to use letterbox image resizing
        "cuda": True,  # Whether to use CUDA for GPU acceleration
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)  # Set parameters from kwargs
            self._defaults[name] = value  # Update defaults with provided values

        self.class_names, self.num_classes = get_classes(self.classes_path)
        self.bbox_util = DecodeBox(self.num_classes, (self.input_shape[0], self.input_shape[1]))

        hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))
        self.generate()

        show_config(**self._defaults)

    def generate(self, onnx=False):
        self.net = YoloBody(self.input_shape, self.num_classes, self.phi)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net.load_state_dict(torch.load(self.model_path, map_location=device))
        self.net = self.net.eval()
        print('{} model, and classes loaded.'.format(self.model_path))

        if not onnx:
            if self.cuda:
                self.net = nn.DataParallel(self.net)
                self.net = self.net.cuda()

    def detect_image(self, image_rgb, image_nir):
        image_shape = np.array(np.shape(image_rgb)[0:2])

        image_rgb = cvtColor(image_rgb)
        image_nir = cvtColor(image_nir)

        image_data_rgb = resize_image(image_rgb, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        image_data_nir = resize_image(image_nir, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)

        image_data_rgb = np.expand_dims(
            np.transpose(preprocess_input(np.array(image_data_rgb, dtype='float32')), (2, 0, 1)), 0)
        image_data_nir = np.expand_dims(
            np.transpose(preprocess_input(np.array(image_data_nir, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images_rgb = torch.from_numpy(image_data_rgb)
            images_nir = torch.from_numpy(image_data_nir)
            if self.cuda:
                images_rgb = images_rgb.cuda()
                images_nir = images_nir.cuda()
            outputs = self.net(images_rgb, images_nir)
            outputs = self.bbox_util.decode_box(outputs)
            results = self.bbox_util.non_max_suppression(outputs, self.num_classes, self.input_shape,
                                                         image_shape, self.letterbox_image, conf_thres=self.confidence,
                                                         nms_thres=self.nms_iou)

            if results[0] is None:
                return image_rgb

            top_label = np.array(results[0][:, 5], dtype='int32')
            top_conf = results[0][:, 4]
            top_boxes = results[0][:, :4]

        font = ImageFont.truetype(font='model_data/simhei.ttf',
                                  size=np.floor(3e-2 * image_rgb.size[1] + 0.5).astype('int32'))
        thickness = int(max((image_rgb.size[0] + image_rgb.size[1]) // np.mean(self.input_shape), 1))

        for i, c in list(enumerate(top_label)):
            predicted_class = self.class_names[int(c)]
            box = top_boxes[i]
            score = top_conf[i]

            top, left, bottom, right = box

            top = max(0, np.floor(top).astype('int32'))
            left = max(0, np.floor(left).astype('int32'))
            bottom = min(image_rgb.size[1], np.floor(bottom).astype('int32'))
            right = min(image_rgb.size[0], np.floor(right).astype('int32'))

            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image_rgb)
            label_size = draw.textsize(label, font)
            label = label.encode('utf-8')
            print(label, top, left, bottom, right)

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            for i in range(thickness):
                draw.rectangle([left + i, top + i, right - i, bottom - i], outline=self.colors[c])
            draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=self.colors[c])
            draw.text(text_origin, str(label, 'UTF-8'), fill=(0, 0, 0), font=font)
            del draw

        return image_rgb

    def get_FPS(self, image_rgb, image_nir, test_interval):
        image_shape = np.array(np.shape(image_rgb)[0:2])
        image_rgb = cvtColor(image_rgb)
        image_nir = cvtColor(image_nir)
        image_data_rgb = resize_image(image_rgb, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        image_data_nir = resize_image(image_nir, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        image_data_rgb = np.expand_dims(
            np.transpose(preprocess_input(np.array(image_data_rgb, dtype='float32')), (2, 0, 1)), 0)
        image_data_nir = np.expand_dims(
            np.transpose(preprocess_input(np.array(image_data_nir, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images_rgb = torch.from_numpy(image_data_rgb)
            images_nir = torch.from_numpy(image_data_nir)
            if self.cuda:
                images_rgb = images_rgb.cuda()
                images_nir = images_nir.cuda()
            outputs = self.net(images_rgb, images_nir)
            outputs = self.bbox_util.decode_box(outputs)
            results = self.bbox_util.non_max_suppression(outputs, self.num_classes, self.input_shape,
                                                         image_shape, self.letterbox_image, conf_thres=self.confidence,
                                                         nms_thres=self.nms_iou)

        t1 = time.time()
        for _ in range(test_interval):
            with torch.no_grad():
                outputs = self.net(images_rgb, images_nir)
                outputs = self.bbox_util.decode_box(outputs)
                results = self.bbox_util.non_max_suppression(outputs, self.num_classes, self.input_shape,
                                                             image_shape, self.letterbox_image,
                                                             conf_thres=self.confidence, nms_thres=self.nms_iou)

        t2 = time.time()
        tact_time = (t2 - t1) / test_interval
        return tact_time

    def detect_heatmap(self, image_rgb, image_nir, heatmap_save_path):
        import cv2
        import matplotlib.pyplot as plt
        def sigmoid(x):
            y = 1.0 / (1.0 + np.exp(-x))
            return y

        image_rgb = cvtColor(image_rgb)
        image_nir = cvtColor(image_nir)
        image_data_rgb = resize_image(image_rgb, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        image_data_nir = resize_image(image_nir, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        image_data_rgb = np.expand_dims(
            np.transpose(preprocess_input(np.array(image_data_rgb, dtype='float32')), (2, 0, 1)), 0)
        image_data_nir = np.expand_dims(
            np.transpose(preprocess_input(np.array(image_data_nir, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images_rgb = torch.from_numpy(image_data_rgb)
            images_nir = torch.from_numpy(image_data_nir)
            if self.cuda:
                images_rgb = images_rgb.cuda()
                images_nir = images_nir.cuda()
            dbox, cls, x, anchors, strides = self.net(images_rgb, images_nir)
            outputs = [xi.split((xi.size()[1] - self.num_classes, self.num_classes), 1)[1] for xi in x]

        plt.imshow(image_rgb, alpha=1)
        plt.axis('off')
        mask = np.zeros((image_rgb.size[1], image_rgb.size[0]))
        for sub_output in outputs:
            sub_output = sub_output.cpu().numpy()
            b, c, h, w = np.shape(sub_output)
            sub_output = np.transpose(np.reshape(sub_output, [b, -1, h, w]), [0, 2, 3, 1])[0]
            score = np.max(sigmoid(sub_output[..., :]), -1)
            score = cv2.resize(score, (image_rgb.size[0], image_rgb.size[1]))
            normed_score = (score * 255).astype('uint8')
            mask = np.maximum(mask, normed_score)

        plt.imshow(mask, alpha=0.5, interpolation='nearest', cmap="jet")
        plt.axis('off')
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.savefig(heatmap_save_path, dpi=200, bbox_inches='tight', pad_inches=-0.1)
        print("Save to the " + heatmap_save_path)
        plt.show()

    def get_map_txt(self, image_id, image_rgb, image_nir, class_names, map_out_path):
        f = open(os.path.join(map_out_path, "detection-results/" + image_id + ".txt"), "w", encoding='utf-8')
        image_shape = np.array(np.shape(image_rgb)[0:2])
        image_rgb = cvtColor(image_rgb)
        image_nir = cvtColor(image_nir)
        image_data_rgb = resize_image(image_rgb, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        image_data_nir = resize_image(image_nir, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        image_data_rgb = np.expand_dims(
            np.transpose(preprocess_input(np.array(image_data_rgb, dtype='float32')), (2, 0, 1)), 0)
        image_data_nir = np.expand_dims(
            np.transpose(preprocess_input(np.array(image_data_nir, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images_rgb = torch.from_numpy(image_data_rgb)
            images_nir = torch.from_numpy(image_data_nir)
            if self.cuda:
                images_rgb = images_rgb.cuda()
                images_nir = images_nir.cuda()
            outputs = self.net(images_rgb, images_nir)
            outputs = self.bbox_util.decode_box(outputs)
            results = self.bbox_util.non_max_suppression(outputs, self.num_classes, self.input_shape,
                                                         image_shape, self.letterbox_image, conf_thres=self.confidence,
                                                         nms_thres=self.nms_iou)

            if results[0] is None:
                return

            top_label = np.array(results[0][:, 5], dtype='int32')
            top_conf = results[0][:, 4]
            top_boxes = results[0][:, :4]

        for i, c in list(enumerate(top_label)):
            predicted_class = self.class_names[int(c)]
            box = top_boxes[i]
            score = str(top_conf[i])

            top, left, bottom, right = box
            if predicted_class not in class_names:
                continue

            f.write("%s %s %s %s %s %s\n" % (
                predicted_class, score[:6], str(int(left)), str(int(top)), str(int(right)), str(int(bottom))))

        f.close()
        return