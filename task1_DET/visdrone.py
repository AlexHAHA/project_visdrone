"""
功能：
1、将visdrone的annotation转为yolov3的label
2、对visdrone的图片进行annotation叠加，观看效果
3、将所有的图片路径存放在一个.txt中
"""
import os
import numpy as np
import torch
import cv2
import pickle as pkl
from tqdm import tqdm


def generate_colors():
    '''
    加载调色板
    '''
    current_path = os.getcwd()
    #print(current_path)
    pallete_path = os.path.abspath(os.path.join(current_path,".."))
    pallete_file = os.path.join(pallete_path,"resources\\pallete")
    colors = pkl.load(open(pallete_file, "rb"))
    colors = np.array(colors)
    return colors

class VisDrone(object):
    '''
    VisDrone数据集转换，bbox叠加显示
    '''
    def __init__(self):
        #self.path_base        = r"D:\CETCA_DeepLearning\Task 1 - Object Detection in Images\VisDrone2019-DET-train"
        #self.path_base        = r"H:\deepLearning\dataset\visdrone\Task 1 - Object Detection in Images\VisDrone2019-DET-train"
        self.path_base        = r"H:\deepLearning\dataset\visdrone\Task 1 - Object Detection in Images\VisDrone2019-DET-val"
        self.path_images      = os.path.join(self.path_base, "images")
        self.path_annotations = os.path.join(self.path_base, "annotations") 

        self.colors = generate_colors()
        self.file_imgs = os.listdir(self.path_images)
        self.file_anns = os.listdir(self.path_annotations)

    def read_annotation(self, anno_file):
        #读取文件获取标签
        lbls = None
        with open(anno_file) as f:
            lbls = f.readlines()

        #将标签存放为nparray
        labels = np.zeros((len(lbls), 8))
        for i,lbl in enumerate(lbls):
            lbl = lbl.rstrip(', \n')
            res = [int(num) for num in lbl.split(',')]
            labels[i,:] = np.array(res)
        return labels

    def add_annotation_layer(self, img_name, ann_name):
        img_path        = os.path.join(self.path_images, img_name)
        annotation_path = os.path.join(self.path_annotations, ann_name)
        img = cv2.imread(img_path)

        #读取文件获取标签
        lbls = None
        with open(annotation_path) as f:
            lbls = f.readlines()

        #将标签存放为nparray
        labels = np.zeros((len(lbls), 8))
        for i,lbl in enumerate(lbls):
            res = [int(num) for num in lbl.split(',')]
            labels[i,:] = np.array(res)

        for i,lbl in enumerate(labels):
            p1 = (int(lbl[0]), int(lbl[1]))
            p2 = (int(lbl[0]+lbl[2]), int(lbl[1]+lbl[3]))
            #print(p1,p2)
            c = int(lbl[5])
            color = tuple(self.colors[c])
            #print(type(color[0])
            #print(color)
            #
            if lbl[4]>=1:
                pass
            img = cv2.rectangle(img, p1, p2, (int(color[0]),int(color[1]),int(color[2])), 2)
        return img

    def ppt(self, num=10, delay=1000):
        '''
        将图片叠加annotation后进行幻灯片播放
        '''
        for i in range(num):
            image = self.add_annotation_layer(self.file_imgs[i],self.file_anns[i])
            cv2.imshow('visdrone',image)
            cv2.waitKey(delay)

    def visdrone2yolo(self, num=10):
        '''
        将annotation转成yolo格式
        '''
        path_labels = os.path.join(self.path_base, "labels")
        if not os.path.exists(path_labels):
            os.mkdir(path_labels)

        # 对所有annotation进行转换
        if num == -1:
            num = len(self.file_anns)

        for i in tqdm(range(num)):
            try:
                anno_file = self.file_anns[i]
                img_file  = self.file_imgs[i]
                #获取图片大小
                img = cv2.imread(os.path.join(self.path_images, img_file))
                img_height,img_width = img.shape[:2]
                #读取visdrone的annotation
                labels = self.read_annotation(os.path.join(self.path_annotations, anno_file))
                labels_str = ""
                for lbl in labels:
                    #如果bbox没有包含目标则忽略
                    if lbl[4]<1:
                        continue
                    #如果bbox宽度或高度为0则忽略
                    if lbl[2] == 0 or lbl[3] == 0:
                        print(f"img:{img_file}包含无效bbox，高度或宽度为0")
                        continue

                    c_x = lbl[0]+lbl[2]/2
                    c_y = lbl[1]+lbl[3]/2
                    lbl_str = "{} {:.4f} {:.4f} {:.4f} {:.4f}\n".format(int(lbl[5]),c_x/img_width,c_y/img_height,
                                                                        lbl[2]/img_width,lbl[3]/img_height)
                    labels_str += lbl_str
                #新建yolo的label文件，并写入yolo格式的label
                with open(os.path.join(path_labels, anno_file),'w') as anno_f:
                    anno_f.write(labels_str)
                #print(f"finished {anno_file}")
            except Exception as e:
                print(f"to_yolo error:{e}")
                print(f"{anno_file}")

    def generate_file(self, txt_name):
        '''
        将所有数据集内的图片名写入txt文件
        '''
        contents = ""
        with open(os.path.join(self.path_base, txt_name),'w') as f:
            contents = ""
            for img in self.file_imgs:
                contents = f"{os.path.join(self.path_images, img)}\n"
                f.write(contents)

if __name__ == '__main__':
    visdrone = VisDrone()
    #visdrone.visdrone2yolo(-1)
    #visdrone.generate_file('train.txt')
    #visdrone.generate_file('valid.txt')
    visdrone.ppt()

