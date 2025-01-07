from random import sample, shuffle
import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset
from utils.utils import cvtColor, preprocess_input


class YoloDataset(Dataset):
    def __init__(self, annotation_lines, input_shape, num_classes, epoch_length, mosaic, mixup, mosaic_prob, mixup_prob,
                 train, special_aug_ratio=0.7):
        super(YoloDataset, self).__init__()
        self.annotation_lines = annotation_lines
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.epoch_length = epoch_length
        self.mosaic = mosaic
        self.mosaic_prob = mosaic_prob
        self.mixup = mixup
        self.mixup_prob = mixup_prob
        self.train = train
        self.special_aug_ratio = special_aug_ratio

        self.epoch_now = -1
        self.length = len(self.annotation_lines)
        self.bbox_attrs = 5 + num_classes

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        index = index % self.length

        # ---------------------------------------------------#
        #   Random data augmentation during training
        #   No random data augmentation during validation
        # ---------------------------------------------------#
        if self.mosaic and self.rand() < self.mosaic_prob and self.epoch_now < self.epoch_length * self.special_aug_ratio:
            lines = sample(self.annotation_lines, 3)
            lines.append(self.annotation_lines[index])
            shuffle(lines)
            image_rgb, image_nir, box = self.get_random_data_with_Mosaic(lines, self.input_shape)

            if self.mixup and self.rand() < self.mixup_prob:
                lines = sample(self.annotation_lines, 1)
                image_rgb_2, image_nir_2, box_2 = self.get_random_data(lines[0], self.input_shape, random=self.train)
                image_rgb, image_nir, box = self.get_random_data_with_MixUp(image_rgb, image_nir, box, image_rgb_2,
                                                                            image_nir_2, box_2)
        else:
            image_rgb, image_nir, box = self.get_random_data(self.annotation_lines[index], self.input_shape,
                                                             random=self.train)

        # Convert RGB and NIR images to numpy arrays and preprocess
        image_rgb = np.transpose(preprocess_input(np.array(image_rgb, dtype=np.float32)), (2, 0, 1))
        image_nir = np.transpose(preprocess_input(np.array(image_nir, dtype=np.float32)), (2, 0, 1))
        box = np.array(box, dtype=np.float32)

        # ---------------------------------------------------#
        #   Preprocess the ground truth boxes
        # ---------------------------------------------------#
        nL = len(box)
        labels_out = np.zeros((nL, 6))
        if nL:
            # ---------------------------------------------------#
            #   Normalize the boxes to be between 0 and 1
            # ---------------------------------------------------#
            box[:, [0, 2]] = box[:, [0, 2]] / self.input_shape[1]
            box[:, [1, 3]] = box[:, [1, 3]] / self.input_shape[0]
            # ---------------------------------------------------#
            #   The first two indices are the center of the box
            #   The next two indices are the width and height of the box
            #   The last index is the class of the box
            # ---------------------------------------------------#
            box[:, 2:4] = box[:, 2:4] - box[:, 0:2]
            box[:, 0:2] = box[:, 0:2] + box[:, 2:4] / 2

            labels_out[:, 1] = box[:, -1]
            labels_out[:, 2:] = box[:, :4]

        return image_rgb, image_nir, labels_out

    def rand(self, a=0, b=1):
        return np.random.rand() * (b - a) + a

    def get_random_data(self, annotation_line, input_shape, jitter=.3, hue=.1, sat=0.7, val=0.4, random=True):
        line = annotation_line.split()
        # ------------------------------#
        #   读取RGB和NIR图像
        # ------------------------------#
        image_rgb = Image.open(line[0])  # RGB图像路径
        image_nir = Image.open(line[1])  # NIR图像路径

        image_rgb = cvtColor(image_rgb)
        image_nir = cvtColor(image_nir)

        iw, ih = image_rgb.size
        h, w = input_shape
        box = np.array([np.array(list(map(int, box.split(',')))) for box in line[2:]])

        if not random:
            scale = min(w / iw, h / ih)
            nw = int(iw * scale)
            nh = int(ih * scale)
            dx = (w - nw) // 2
            dy = (h - nh) // 2

            image_rgb = image_rgb.resize((nw, nh), Image.BICUBIC)
            image_nir = image_nir.resize((nw, nh), Image.BICUBIC)

            new_image_rgb = Image.new('RGB', (w, h), (128, 128, 128))
            new_image_nir = Image.new('RGB', (w, h), (128, 128, 128))

            new_image_rgb.paste(image_rgb, (dx, dy))
            new_image_nir.paste(image_nir, (dx, dy))

            image_rgb_data = np.array(new_image_rgb, np.float32)
            image_nir_data = np.array(new_image_nir, np.float32)

            if len(box) > 0:
                np.random.shuffle(box)
                box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
                box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
                box[:, 0:2][box[:, 0:2] < 0] = 0
                box[:, 2][box[:, 2] > w] = w
                box[:, 3][box[:, 3] > h] = h
                box_w = box[:, 2] - box[:, 0]
                box_h = box[:, 3] - box[:, 1]
                box = box[np.logical_and(box_w > 1, box_h > 1)]

            return image_rgb_data, image_nir_data, box

        # ------------------------------------------#
        #   Resize the images and apply distortion
        # ------------------------------------------#
        new_ar = iw / ih * self.rand(1 - jitter, 1 + jitter) / self.rand(1 - jitter, 1 + jitter)
        scale = self.rand(0.25, 2)
        if new_ar < 1:
            nh = int(scale * h)
            nw = int(nh * new_ar)
        else:
            nw = int(scale * w)
            nh = int(nw / new_ar)

        # 进行图像缩放
        image_rgb = image_rgb.resize((nw, nh), Image.BICUBIC)
        image_nir = image_nir.resize((nw, nh), Image.BICUBIC)

        # ------------------------------------------#
        #   Add gray bars to the excess parts of the image
        # ------------------------------------------#
        dx = int(self.rand(0, w - nw))
        dy = int(self.rand(0, h - nh))

        # 创建新图像并将缩放后的RGB和NIR图像粘贴到新图像上
        new_image_rgb = Image.new('RGB', (w, h), (128, 128, 128))
        new_image_nir = Image.new('RGB', (w, h), (128, 128, 128))

        new_image_rgb.paste(image_rgb, (dx, dy))
        new_image_nir.paste(image_nir, (dx, dy))

        image_rgb_data = np.array(new_image_rgb, np.uint8)
        image_nir_data = np.array(new_image_nir, np.uint8)

        # ------------------------------------------#
        #   Flip the images
        # ------------------------------------------#
        flip = self.rand() < .5
        if flip:
            image_rgb_data = np.fliplr(image_rgb_data)
            image_nir_data = np.fliplr(image_nir_data)

        # ---------------------------------#
        #   Adjust the ground truth boxes
        # ---------------------------------#
        if len(box) > 0:
            np.random.shuffle(box)
            box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
            box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
            if flip:
                box[:, [0, 2]] = w - box[:, [2, 0]]
            box[:, 0:2][box[:, 0:2] < 0] = 0
            box[:, 2][box[:, 2] > w] = w
            box[:, 3][box[:, 3] > h] = h
            box_w = box[:, 2] - box[:, 0]
            box_h = box[:, 3] - box[:, 1]
            box = box[np.logical_and(box_w > 1, box_h > 1)]

        return image_rgb_data, image_nir_data, box

    # ------------------------------------------#
    #   Stitch and add gray bars to multimodal images
    # ------------------------------------------#
    def merge_bboxes(self, bboxes, cutx, cuty):
        merge_bbox = []
        for i in range(len(bboxes)):
            for box in bboxes[i]:
                tmp_box = []
                x1, y1, x2, y2 = box[0], box[1], box[2], box[3]

                if i == 0:
                    if y1 > cuty or x1 > cutx:
                        continue
                    if y2 >= cuty and y1 <= cuty:
                        y2 = cuty
                    if x2 >= cutx and x1 <= cutx:
                        x2 = cutx

                if i == 1:
                    if y2 < cuty or x1 > cutx:
                        continue
                    if y2 >= cuty and y1 <= cuty:
                        y1 = cuty
                    if x2 >= cutx and x1 <= cutx:
                        x2 = cutx

                if i == 2:
                    if y2 < cuty or x2 < cutx:
                        continue
                    if y2 >= cuty and y1 <= cuty:
                        y1 = cuty
                    if x2 >= cutx and x1 <= cutx:
                        x1 = cutx

                if i == 3:
                    if y1 > cuty or x2 < cutx:
                        continue
                    if y2 >= cuty and y1 <= cuty:
                        y2 = cuty
                    if x2 >= cutx and x1 <= cutx:
                        x1 = cutx
                tmp_box.append(x1)
                tmp_box.append(y1)
                tmp_box.append(x2)
                tmp_box.append(y2)
                tmp_box.append(box[-1])
                merge_bbox.append(tmp_box)
        return merge_bbox

    def get_random_data_with_Mosaic(self, annotation_line, input_shape, jitter=0.3, hue=.1, sat=0.7, val=0.4):
        h, w = input_shape
        min_offset_x = self.rand(0.3, 0.7)
        min_offset_y = self.rand(0.3, 0.7)

        image_datas_rgb = []
        image_datas_nir = []
        box_datas = []
        index = 0
        for line in annotation_line:
            line_content = line.split()
            image_rgb = Image.open(line_content[0])
            image_nir = Image.open(line_content[1])

            image_rgb = cvtColor(image_rgb)
            image_nir = cvtColor(image_nir)

            iw, ih = image_rgb.size
            box = np.array([np.array(list(map(int, box.split(',')))) for box in line_content[2:]])

            flip = self.rand() < .5
            if flip and len(box) > 0:
                image_rgb = image_rgb.transpose(Image.FLIP_LEFT_RIGHT)
                image_nir = image_nir.transpose(Image.FLIP_LEFT_RIGHT)
                box[:, [0, 2]] = iw - box[:, [2, 0]]

            new_ar = iw / ih * self.rand(1 - jitter, 1 + jitter) / self.rand(1 - jitter, 1 + jitter)
            scale = self.rand(.4, 1)
            if new_ar < 1:
                nh = int(scale * h)
                nw = int(nh * new_ar)
            else:
                nw = int(scale * w)
                nh = int(nw / new_ar)

            image_rgb = image_rgb.resize((nw, nh), Image.BICUBIC)
            image_nir = image_nir.resize((nw, nh), Image.BICUBIC)

            if index == 0:
                dx = int(w * min_offset_x) - nw
                dy = int(h * min_offset_y) - nh
            elif index == 1:
                dx = int(w * min_offset_x) - nw
                dy = int(h * min_offset_y)
            elif index == 2:
                dx = int(w * min_offset_x)
                dy = int(h * min_offset_y)
            elif index == 3:
                dx = int(w * min_offset_x)
                dy = int(h * min_offset_y) - nh

            new_image_rgb = Image.new('RGB', (w, h), (128, 128, 128))
            new_image_nir = Image.new('RGB', (w, h), (128, 128, 128))

            new_image_rgb.paste(image_rgb, (dx, dy))
            new_image_nir.paste(image_nir, (dx, dy))

            image_rgb_data = np.array(new_image_rgb)
            image_nir_data = np.array(new_image_nir)

            index += 1
            box_data = []
            if len(box) > 0:
                np.random.shuffle(box)
                box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
                box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
                box[:, 0:2][box[:, 0:2] < 0] = 0
                box[:, 2][box[:, 2] > w] = w
                box[:, 3][box[:, 3] > h] = h
                box_w = box[:, 2] - box[:, 0]
                box_h = box[:, 3] - box[:, 1]
                box = box[np.logical_and(box_w > 1, box_h > 1)]
                box_data = np.zeros((len(box), 5))
                box_data[:len(box)] = box

            image_datas_rgb.append(image_rgb_data)
            image_datas_nir.append(image_nir_data)
            box_datas.append(box_data)

        cutx = int(w * min_offset_x)
        cuty = int(h * min_offset_y)

        new_image_rgb = np.zeros([h, w, 3])
        new_image_nir = np.zeros([h, w, 3])

        new_image_rgb[:cuty, :cutx, :] = image_datas_rgb[0][:cuty, :cutx, :]
        new_image_rgb[cuty:, :cutx, :] = image_datas_rgb[1][cuty:, :cutx, :]
        new_image_rgb[cuty:, cutx:, :] = image_datas_rgb[2][cuty:, cutx:, :]
        new_image_rgb[:cuty, cutx:, :] = image_datas_rgb[3][:cuty, cutx:, :]

        new_image_nir[:cuty, :cutx, :] = image_datas_nir[0][:cuty, :cutx, :]
        new_image_nir[cuty:, :cutx, :] = image_datas_nir[1][cuty:, :cutx, :]
        new_image_nir[cuty:, cutx:, :] = image_datas_nir[2][cuty:, cutx:, :]
        new_image_nir[:cuty, cutx:, :] = image_datas_nir[3][:cuty, cutx:, :]

        new_image_rgb = np.array(new_image_rgb, np.uint8)
        new_image_nir = np.array(new_image_nir, np.uint8)
        # ---------------------------------#
        #   Apply color domain transformation
        # ---------------------------------#
        r = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1
        # ---------------------------------#
        #   Move the images to HSV
        # ---------------------------------#
        hue_rgb, sat_rgb, val_rgb = cv2.split(cv2.cvtColor(new_image_rgb, cv2.COLOR_RGB2HSV))
        hue_nir, sat_nir, val_nir = cv2.split(cv2.cvtColor(new_image_nir, cv2.COLOR_RGB2HSV))
        dtype_rgb = new_image_rgb.dtype
        dtype_nir = new_image_nir.dtype
        # ---------------------------------#
        #   Apply transformations
        # ---------------------------------#
        x_rgb = np.arange(0, 256, dtype=dtype_rgb)
        x_nir = np.arange(0, 256, dtype=dtype_nir)
        lut_hue_rgb = ((x_rgb * r[0]) % 180).astype(dtype_rgb)
        lut_hue_nir = ((x_nir * r[0]) % 180).astype(dtype_nir)
        lut_sat_rgb = np.clip(x_rgb * r[1], 0, 255).astype(dtype_rgb)
        lut_sat_nir = np.clip(x_nir * r[1], 0, 255).astype(dtype_nir)
        lut_val_rgb = np.clip(x_rgb * r[2], 0, 255).astype(dtype_rgb)
        lut_val_nir = np.clip(x_nir * r[2], 0, 255).astype(dtype_nir)

        new_image_rgb = cv2.merge((cv2.LUT(hue_rgb, lut_hue_rgb), cv2.LUT(sat_rgb, lut_sat_rgb), cv2.LUT(val_rgb, lut_val_rgb)))
        new_image_nir = cv2.merge((cv2.LUT(hue_nir, lut_hue_nir), cv2.LUT(sat_nir, lut_sat_nir), cv2.LUT(val_nir, lut_val_nir)))
        new_image_rgb = cv2.cvtColor(new_image_rgb, cv2.COLOR_HSV2RGB)
        new_image_nir = cv2.cvtColor(new_image_nir, cv2.COLOR_HSV2RGB)


        new_boxes = self.merge_bboxes(box_datas, cutx, cuty)

        return new_image_rgb, new_image_nir, new_boxes

    def get_random_data_with_MixUp(self, image_rgb_1, image_nir_1, box_1, image_rgb_2, image_nir_2, box_2):
        new_image_rgb = np.array(image_rgb_1, np.float32) * 0.5 + np.array(image_rgb_2, np.float32) * 0.5
        new_image_nir = np.array(image_nir_1, np.float32) * 0.5 + np.array(image_nir_2, np.float32) * 0.5
        if len(box_1) == 0:
            new_boxes = box_2
        elif len(box_2) == 0:
            new_boxes = box_1
        else:
            new_boxes = np.concatenate([box_1, box_2], axis=0)
        return new_image_rgb, new_image_nir, new_boxes


def yolo_dataset_collate(batch):
    images_rgb = []
    images_nir = []
    bboxes = []
    for i, (img_rgb, img_nir, box) in enumerate(batch):
        images_rgb.append(img_rgb)
        images_nir.append(img_nir)
        box[:, 0] = i
        bboxes.append(box)

    images_rgb = torch.from_numpy(np.array(images_rgb)).type(torch.FloatTensor)
    images_nir = torch.from_numpy(np.array(images_nir)).type(torch.FloatTensor)
    bboxes = torch.from_numpy(np.concatenate(bboxes, 0)).type(torch.FloatTensor)
    return images_rgb, images_nir, bboxes
