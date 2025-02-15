以下是完善后的 README.md 文件内容：
DCFA-YOLO: A Dual-Channel Cross-Feature-Fusion Attention YOLO Network for Cherry Tomato Bunch Detection
This repository contains the implementation of the DCFA-YOLO network, as described in the paper DCFA-YOLO: A Dual-Channel Cross-Feature-Fusion Attention YOLO Network for Cherry Tomato Bunch Detection. This model leverages multi-modal information from RGB and depth images to improve detection performance, particularly for agricultural applications such as cherry tomato bunch detection.
DCFA-YOLO: YOLOv8-based Multi-modal Object Detection Model
Supports Adam and SGD optimizers
Supports EMA (Exponential Moving Average)
Supports heatmap visualization
Requirements
Python 3.7+
PyTorch 1.7.1+ (recommended for AMP mixed-precision training)
CUDA 10.2+
OpenCV
NumPy
Quick Start
Install Dependencies
bash复制
pip install -r requirements.txt
Training Steps
1. Data Preparation
Prepare VOC-format dataset
Place RGB images in VOCdevkit/VOC2007/JPEGImages_rgb
Place Depth images in VOCdevkit/VOC2007/JPEGImages_nir
Place annotation files in VOCdevkit/VOC2007/Annotations
2. Data Preprocessing
Modify parameters in voc_annotation_mul.py and run:
Python复制
python voc_annotation_mul.py
3. Start Training
Python复制
python train_mul.py
Inference
Modify model path and class file path in yolo_mul.py
Run prediction script:
Python复制
python predict_mul.py
Model Evaluation
VOC Dataset Evaluation
Modify model path and class file path in yolo_mul.py
Run evaluation script:
Python复制
python get_map_mul.py
Multi-modal Input Specification
This project supports RGB and Depth dual-modal input with the following requirements:
RGB images: Standard 3-channel color images, stored in JPEGImages_rgb
Depth images: Single-channel grayscale images, stored in JPEGImages_nir
Image dimensions must match
File names must strictly correspond (e.g., 001.jpg and 001.png)
Script Description
All core scripts have been modified for multi-modal input, including:
voc_annotation_mul.py: Multi-modal data preprocessing
train_mul.py: Multi-modal training script
predict_mul.py: Multi-modal inference script
yolo_mul.py: Core implementation of multi-modal model
get_map_mul.py: Multi-modal evaluation script
References
YOLOv8-Pytorch
Ultralytics YOLOv8
License
This project is licensed under the MIT License. See LICENSE file for details.
Contact Us
For any questions, please contact us via:
Email: 2450096004@mails.szu.edu.cn
GitHub Issues: New Issue
