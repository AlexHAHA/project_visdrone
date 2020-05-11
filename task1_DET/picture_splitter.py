"""
将数据集的图片进行分割，获取较小尺寸的图片
"""
import os
import cv2
import numpy as np

class ImageSplitter(object):
    """
    split the image in dataset into smaller images
    label format should be yolov3
    """
    def __init__(self, path_images, path_labels, size=(416,416), img_fmt='.jpg'):
        self.path_images = path_images
        self.path_labels = path_labels
        self.outimg_size = size

        self.file_images = os.listdir(self.path_images)
        self.file_labels = os.listdir(self.path_labels)
        self.file_log = 'log.txt'
        self.f_log = open(self.file_log, 'w')

    def process():
        for f_image in self.file_images:
            name = os.path.basename(f_image)
            f_label = os.path.join(self.path_labels, name.replace('.jpg', '.txt').replace('.png', '.txt'))
            if f_label not in self.file_labels:
                self.flog.write(f'Label not exist cooresponding to: {name}')
                continue

            bboxes = self.__load_txt(f_label)
            img = cv2.imread(f_image)
            img_h, img_w = img.shape[:2]
            #
            while len(labels) >0:
                res_bboxes = np.zeros((1,4))
                #
                ref_bbox = bboxes[0]
                bboxes = np.delete(bboxes, 0, axis=0)
                cx, cy = ref_bbox[0]*img_w, ref_bbox[1]*img_h
                bw, bh = ref_bbox[2]*img_w, ref_bbox[3]*img_h
                leftup    = int(int(cx - bw/2), int(cy - bh/2))
                rightdown = int(int(cx + bw/2), int(cy + bh/2))

                for i,bbox in enumerate(bboxes):
                    if is_contain(bbox, leftup, rightdown)


    def is_contain(x):

            


    def __load_txt(self, file_path):
        '''
        使用numpy加载txt，只提取bbox信息
        Return:
            - 二维array
        '''
        datas = np.loadtxt(file_path)
        if len(datas.shape) == 1:
            datas = np.expand_dims(datas, 0)
        return datas[:,1:]

    def __del__(self):
        self.f_log.close()

