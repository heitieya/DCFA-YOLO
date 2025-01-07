import os
import random
import xml.etree.ElementTree as ET
import numpy as np
from utils.utils import get_classes

#--------------------------------------------------------------------------------------------------------------------------------#
#   annotation_mode is used to specify the content calculated when this file runs
#   annotation_mode 0 represents the entire label processing process, including obtaining txt files in VOCdevkit/VOC2007/ImageSets and the training files 2007_train.txt, 2007_val.txt
#   annotation_mode 1 represents obtaining txt files in VOCdevkit/VOC2007/ImageSets
#   annotation_mode 2 represents obtaining training files 2007_train.txt, 2007_val.txt
#--------------------------------------------------------------------------------------------------------------------------------#
annotation_mode = 2

#-------------------------------------------------------------------#
#   Used to generate target information for 2007_train.txt and 2007_val.txt
#   It should be consistent with the classes_path used for training and prediction
#   If there is no target information in the generated 2007_train.txt
#   It is because the classes are not set correctly
#   Only effective when annotation_mode is 0 and 2
#-------------------------------------------------------------------#
classes_path = 'model_data/voc_classes.txt'

#--------------------------------------------------------------------------------------------------------------------------------#
#   trainval_percent is used to specify the ratio of (training set + validation set) to the test set, by default (training set + validation set): test set = 9:1
#   train_percent is used to specify the ratio of the training set to the validation set in (training set + validation set), by default training set: validation set = 9:1
#   Only effective when annotation_mode is 0 and 1
#--------------------------------------------------------------------------------------------------------------------------------#
trainval_percent = 0.9
train_percent = 0.9

#-------------------------------------------------------#
#   Points to the folder where the VOC dataset is located
#-------------------------------------------------------#
VOCdevkit_path = 'VOCdevkit'

# 修改这里的路径为多模态路径
VOCdevkit_sets = [('2007', 'train'), ('2007', 'val')]
classes, _ = get_classes(classes_path)

#-------------------------------------------------------#
#   Count the number of targets
#-------------------------------------------------------#
photo_nums = np.zeros(len(VOCdevkit_sets))
nums = np.zeros(len(classes))

def convert_annotation(year, image_id, list_file):
    in_file = open(os.path.join(VOCdevkit_path, 'VOC%s/Annotations/%s.xml' % (year, image_id)), encoding='utf-8')
    tree = ET.parse(in_file)
    root = tree.getroot()

    for obj in root.iter('object'):
        difficult = 0
        if obj.find('difficult') is not None:
            difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (int(float(xmlbox.find('xmin').text)), int(float(xmlbox.find('ymin').text)),
             int(float(xmlbox.find('xmax').text)), int(float(xmlbox.find('ymax').text)))
        list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))
        nums[classes.index(cls)] += 1

if __name__ == "__main__":
    random.seed(0)
    if " " in os.path.abspath(VOCdevkit_path):
        print(os.path.abspath(VOCdevkit_path))
        raise ValueError("The folder path where the dataset is stored and the image name cannot contain spaces, please modify the path.")

    if annotation_mode == 0 or annotation_mode == 1:
        print("Generate txt in ImageSets.")
        xmlfilepath = os.path.join(VOCdevkit_path, 'VOC2007/Annotations')
        saveBasePath = os.path.join(VOCdevkit_path, 'VOC2007/ImageSets/Main')
        temp_xml = os.listdir(xmlfilepath)
        total_xml = []
        for xml in temp_xml:
            if xml.endswith(".xml"):
                total_xml.append(xml)

        num = len(total_xml)
        tv = int(num * trainval_percent)
        tr = int(tv * train_percent)
        trainval = random.sample(list(range(num)), tv)
        train = random.sample(trainval, tr)

        print("train and val size", tv)
        print("train size", tr)

        ftrainval = open(os.path.join(saveBasePath, 'trainval.txt'), 'w')
        ftest = open(os.path.join(saveBasePath, 'test.txt'), 'w')
        ftrain = open(os.path.join(saveBasePath, 'train.txt'), 'w')
        fval = open(os.path.join(saveBasePath, 'val.txt'), 'w')

        for i in range(num):
            name = total_xml[i][:-4] + '\n'
            if i in trainval:
                ftrainval.write(name)
                if i in train:
                    ftrain.write(name)
                else:
                    fval.write(name)
            else:
                ftest.write(name)

        ftrainval.close()
        ftrain.close()
        fval.close()
        ftest.close()
        print("Generate txt in ImageSets done.")

    if annotation_mode == 0 or annotation_mode == 2:
        print("Generate 2007_train.txt and 2007_val.txt for train.")
        type_index = 0
        for year, image_set in VOCdevkit_sets:
            image_ids = open(os.path.join(VOCdevkit_path, 'VOC%s/ImageSets/Main/%s.txt' % (year, image_set)), encoding='utf-8').read().strip().split()
            list_file = open('%s_%s.txt' % (year, image_set), 'w', encoding='utf-8')
            for image_id in image_ids:
        
                list_file.write('%s/VOC%s/JPEGImages_rgb/%s.jpg' % (os.path.abspath(VOCdevkit_path), year, image_id))
                list_file.write(' %s/VOC%s/JPEGImages_nir/%s.jpg' % (os.path.abspath(VOCdevkit_path), year, image_id))

                convert_annotation(year, image_id, list_file)
                list_file.write('\n')
            photo_nums[type_index] = len(image_ids)
            type_index += 1
            list_file.close()

        print("Generate 2007_train.txt and 2007_val.txt for train done.")

        def printTable(List1, List2):
            for i in range(len(List1[0])):
                print("|", end=' ')
                for j in range(len(List1)):
                    print(List1[j][i].rjust(int(List2[j])), end=' ')
                    print("|", end=' ')
                print()

        str_nums = [str(int(x)) for x in nums]
        tableData = [classes, str_nums]
        colWidths = [0] * len(tableData)

        for i in range(len(tableData)):
            for j in range(len(tableData[i])):
                if len(tableData[i][j]) > colWidths[i]:
                    colWidths[i] = len(tableData[i][j])

        printTable(tableData, colWidths)

        if photo_nums[0] <= 500:
            print("The number of training sets is less than 500, it is recommended to increase the training epochs to meet the sufficient number of gradient descent steps.")

        if np.sum(nums) == 0:
            print("No targets found, please check classes_path or dataset label names.")
