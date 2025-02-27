import os
import xml.etree.ElementTree as ET

from PIL import Image
from tqdm import tqdm

from utils.utils import get_classes
from utils.utils_map import get_coco_map, get_map
from yolo_mul import YOLO

if __name__ == "__main__":
    map_mode = 0  # Mode for map generation (0: default, 1: predict, 2: ground truth, 3: map, 4: COCO map)
    classes_path = 'model_data/voc_classes.txt'  # Path to the classes file
    MINOVERLAP = 0.5  # Minimum overlap for evaluation
    confidence = 0.001  # Confidence threshold for predictions
    nms_iou = 0.5  # IoU threshold for Non-Maximum Suppression
    score_threhold = 0.5  # Score threshold for filtering predictions
    map_vis = False  # Flag to visualize images
    VOCdevkit_path = '/data/csl/python_csl/yolov8_s/VOCdevkit'  # Path to the VOCdevkit dataset
    map_out_path = 'map_out'  # Output path for the map results

    image_ids = open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Main/test.txt")).read().strip().split()

    if not os.path.exists(map_out_path):
        os.makedirs(map_out_path)
    if not os.path.exists(os.path.join(map_out_path, 'ground-truth')):
        os.makedirs(os.path.join(map_out_path, 'ground-truth'))
    if not os.path.exists(os.path.join(map_out_path, 'detection-results')):
        os.makedirs(os.path.join(map_out_path, 'detection-results'))
    if not os.path.exists(os.path.join(map_out_path, 'images-optional')):
        os.makedirs(os.path.join(map_out_path, 'images-optional'))

    class_names, _ = get_classes(classes_path)

    if map_mode == 0 or map_mode == 1:
        print("Load model.")
        yolo = YOLO(confidence=confidence, nms_iou=nms_iou)
        print("Load model done.")

        print("Get predict result.")
        for image_id in tqdm(image_ids):
            image_path_rgb = os.path.join(VOCdevkit_path, "VOC2007/JPEGImages_rgb/" + image_id + ".png")
            image_path_nir = os.path.join(VOCdevkit_path, "VOC2007/JPEGImages_nir/" + image_id + ".png")
            image_rgb = Image.open(image_path_rgb)
            image_nir = Image.open(image_path_nir)
            if map_vis:
                image_rgb.save(os.path.join(map_out_path, "images-optional/" + image_id + "_rgb.jpg"))
                image_nir.save(os.path.join(map_out_path, "images-optional/" + image_id + "_nir.jpg"))
            yolo.get_map_txt(image_id, image_rgb, image_nir, class_names, map_out_path)
        print("Get predict result done.")

    if map_mode == 0 or map_mode == 2:
        print("Get ground truth result.")
        for image_id in tqdm(image_ids):
            with open(os.path.join(map_out_path, "ground-truth/" + image_id + ".txt"), "w") as new_f:
                root = ET.parse(os.path.join(VOCdevkit_path, "VOC2007/Annotations/" + image_id + ".xml")).getroot()
                for obj in root.findall('object'):
                    difficult_flag = False
                    if obj.find('difficult') != None:
                        difficult = obj.find('difficult').text
                        if int(difficult) == 1:
                            difficult_flag = True
                    obj_name = obj.find('name').text
                    if obj_name not in class_names:
                        continue
                    bndbox = obj.find('bndbox')
                    left = bndbox.find('xmin').text
                    top = bndbox.find('ymin').text
                    right = bndbox.find('xmax').text
                    bottom = bndbox.find('ymax').text

                    if difficult_flag:
                        new_f.write("%s %s %s %s %s difficult\n" % (obj_name, left, top, right, bottom))
                    else:
                        new_f.write("%s %s %s %s %s\n" % (obj_name, left, top, right, bottom))
        print("Get ground truth result done.")

    if map_mode == 0 or map_mode == 3:
        print("Get map.")
        get_map(MINOVERLAP, True, score_threhold=score_threhold, path=map_out_path)
        print("Get map done.")

    if map_mode == 4:
        print("Get map.")
        get_coco_map(class_names=class_names, path=map_out_path)
        print("Get map done.")