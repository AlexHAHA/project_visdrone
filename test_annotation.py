import os
import numpy as np
import torch
import cv2
import pickle as pkl

def generate_colors():
    '''
    加载调色板
    '''
    colors = pkl.load(open("resources/pallete", "rb"))
    colors = np.array(colors)
    return colors


class VisDrone(object):
    '''
    VisDrone数据集转换，bbox叠加显示
    '''
    def __init__(self):
        self.path_base        = r"D:\CETCA_DeepLearning\Task 1 - Object Detection in Images\VisDrone2019-DET-train"
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
            res = [int(num) for num in lbl.split(',')]
            labels[i,:] = np.array(res)
        return labels


    def add_annotation_layer(img_name, ann_name):
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
            print(p1,p2)
            c = int(lbl[5])
            color = tuple(colors[c])
            #print(type(color[0])
            #print(color)
            if lbl[4]>=1:
                img = cv2.rectangle(img, p1, p2, (int(color[0]),int(color[1]),int(color[2])), 2)
        return img

    def show(self, num=10, delay=1):
        image = self.add_annotation_layer(self.file_imgs[0],self.file_anns[0])
        cv2.imshow(image)
        cv2.waitKey(1000)

    def to_yolo(self, num=10):
        '''
        将annotation转成yolo格式
        '''
        path_labels = os.path.join(self.path_base, "labels")
        if not os.path.exists(path_labels):
            os.mkdir(path_labels)

        img_width  = 960
        img_height = 540
        for i in range(num):
            try:
                anno_file = self.file_anns[i]
                #读取visdrone的annotation
                labels = self.read_annotation(os.path.join(self.path_annotations, anno_file))
                labels_str = ""
                for lbl in labels:
                    c_x = lbl[0]+lbl[2]/2
                    c_y = lbl[1]+lbl[3]/2
                    lbl_str = "{} {:.4f} {:.4f} {:.4f} {:.4f}\n".format(int(lbl[5]),c_x/img_width,c_y/img_height,
                                                                        lbl[2]/img_width,lbl[3]/img_height)
                    labels_str += lbl_str
                #新建yolo的label文件，并写入yolo格式的label
                with open(os.path.join(path_labels, anno_file),'w') as anno_f:
                    anno_f.write(labels_str)
                print(f"finished {anno_file}")
            except Exception as e:
                print(f"to_yolo error:{e}")
                print(f"{anno_file}")
                

if __name__ == '__main__':
    visdrone = VisDrone()
    visdrone.to_yolo(3)

