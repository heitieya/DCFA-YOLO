import time
import cv2
import numpy as np
from PIL import Image
import os
from tqdm import tqdm

from yolo_mul import YOLO


def load_image_pair(rgb_path, nir_path):  # Load RGB and NIR image paths
    rgb_image = Image.open(rgb_path)  # Open the RGB image
    nir_image = Image.open(nir_path)  # Open the NIR image
    return rgb_image, nir_image  # Return both images


if __name__ == "__main__":
    yolo = YOLO()  # Initialize YOLO object

    mode = "fps"  # Set mode for processing

    rgb_img_path = "path/to/your/rgb_image.jpg"  # Path to the RGB image
    nir_img_path = "path/to/your/nir_image.jpg"  # Path to the NIR image

    test_interval = 100  # Interval for FPS calculation
    fps_rgb_image_path = "logs/JPEGImages_nir_c/220401143702.png"  # Path for RGB image in FPS mode
    fps_nir_image_path = "logs/JPEGImages_nir_c/220401143702.png"  # Path for NIR image in FPS mode

    dir_origin_path_rgb = "D:/A_Python/A_P/face_detect/yolov5-pytorch-main/img_rgb/"  # Directory for original RGB images
    dir_origin_path_nir = "D:/A_Python/A_P/face_detect/yolov5-pytorch-main/img_nir/"  # Directory for original NIR images
    dir_save_path = "img_out/"  # Directory to save output images

    heatmap_save_path = "model_data/heatmap_vision.png"  # Path to save the heatmap image

    if mode == "predict":
        try:
            rgb_image, nir_image = load_image_pair(rgb_img_path, nir_img_path)
        except:
            print(f'Error opening images: {rgb_img_path} or {nir_img_path}')
        else:
            r_image = yolo.detect_image(rgb_image, nir_image)
            r_image.show()

    elif mode == "fps":
        rgb_img, nir_img = load_image_pair(fps_rgb_image_path, fps_nir_image_path)
        tact_time = yolo.get_FPS(rgb_img, nir_img, test_interval)
        print(str(tact_time) + ' seconds, ' + str(1 / tact_time) + 'FPS, @batch_size 1')

    elif mode == "dir_predict":
        img_names = os.listdir(dir_origin_path_rgb)
        for img_name in tqdm(img_names):
            if img_name.lower().endswith(
                    ('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
                rgb_image_path = os.path.join(dir_origin_path_rgb, img_name)
                nir_image_path = os.path.join(dir_origin_path_nir, img_name)

                if not os.path.exists(nir_image_path):
                    print(f"NIR image not found for {img_name}")
                    continue

                rgb_image, nir_image = load_image_pair(rgb_image_path, nir_image_path)
                r_image = yolo.detect_image(rgb_image, nir_image)

                if not os.path.exists(dir_save_path):
                    os.makedirs(dir_save_path)
                r_image.save(os.path.join(dir_save_path, img_name.replace(".jpg", ".png")), quality=95, subsampling=0)

    elif mode == "heatmap":
        while True:
            rgb_img = input('Input RGB image filename:')
            nir_img = input('Input NIR image filename:')
            try:
                rgb_image, nir_image = load_image_pair(rgb_img, nir_img)
            except:
                print('Open Error! Try again!')
                continue
            else:
                yolo.detect_heatmap(rgb_image, nir_image, heatmap_save_path)

    else:
        raise AssertionError(
            "Please specify the correct mode: 'predict', 'video', 'fps', 'heatmap', 'export_onnx', 'dir_predict'.")