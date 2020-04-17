"""
对验证集进行检测，并将检测结果保存，后续使用mAP进行计算。
"""
from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *

import os
import sys
import time
import shutil
import datetime
import argparse

from PIL import Image

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import cv2

def generate_colors():
    '''
    加载调色板
    '''
    current_path = os.getcwd()
    #print(current_path)
    pallete_path = os.path.abspath(os.path.join(current_path,"..\\.."))
    pallete_file = os.path.join(pallete_path,"resources\\pallete")
    colors = pkl.load(open(pallete_file, "rb"))
    colors = np.array(colors)
    return colors

class YoloDetect():
    def __init__(self, model_def,
                       data_config,
                       file_pretrained_weights,
                       path_detection_results,
                       img_size=416,
                ):
        self.frame         = None
        self.detections    = [None]
        #
        data_config      = parse_data_config(data_config)
        self.valid_path  = data_config["valid"]
        self.class_names = load_classes(data_config["names"])
        self.path_detection_results = path_detection_results

        self.class_colors = generate_colors()
        self.img_size   = img_size
        self.conf_thres = 0.8
        self.nms_thres  = 0.4
        # choose GPU or CPU
        self.Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        try:
            self.model = Darknet(model_def).to(self.device)
            self.model.apply(weights_init_normal)
        except Exception as e:
            pass

        if file_pretrained_weights:
            # load user pretrained weights(checkpoint)
            if file_pretrained_weights.endswith(".pth"):
                self.model.load_state_dict(torch.load(file_pretrained_weights))
            # load others pretrained weights
            else:
                self.model.load_darknet_weights(file_pretrained_weights)
        self.model.eval()

    def prepare_image(self, img, in_dim):
        '''
        图片预处理：将原图的整幅图片进行padding后缩放，作为神经网络的输入
        '''
        original_img = img
        img_t   = img[:,:,::-1].transpose((2,0,1)).copy()
        img_t = torch.from_numpy(img_t).float().div(255.0)
        #print(img_t.dtype)
        img_t,_ = pad_to_square(img_t,0)
        img_t = resize(img_t, in_dim)
        img_t = img_t.unsqueeze(0)
        return img_t,original_img

    def add_layer1(self, img, detections, in_dim):
        '''
        直接在原始图片上叠加目标识别和分类效果
        img:       原始图片
        detection: yolov3输出的bounding box
        in_dim:    yolov3输出图片大小
        '''
        h,w = img.shape[:2]
        #print(h,w)
        detections = detections[0].numpy()
        #print(detections.shape)
        diff_dim = int(np.abs(h - w)/2)
        scaler = (h / in_dim) if h>w else (w / in_dim)
        #print('scaler is: ',scaler)
        #print(detections)
        for i in range(detections.shape[0]):
            detect = detections[i]
            label = self.class_names[int(detect[-1])]
            if h > w:
                p1 = (int(detect[0] * scaler) - diff_dim, int(detect[1] * scaler))
                p2 = (int(detect[2] * scaler) - diff_dim, int(detect[3] * scaler))
            else:
                p1 = (int(detect[0] * scaler), int(detect[1] * scaler) - diff_dim)
                p2 = (int(detect[2] * scaler), int(detect[3] * scaler) - diff_dim)
            cv2.rectangle(img, p1, p2, self.class_colors[int(detect[-1])],4)
            cv2.putText(img, label, p1, cv2.FONT_HERSHEY_SIMPLEX, 2.0,
                                    self.class_colors[int(detect[-1])], 2)
        return img

    def add_layer2(self, img, detections, in_dim):
        '''
        直接在网络输入图片上叠加目标识别和分类效果
        '''
        img = img.cpu().squeeze(0).numpy().transpose((1,2,0))
        img = img[:,:,::-1].copy()
        detections = detections[0].numpy()
        for i in range(detections.shape[0]):
            detect = detections[i]
            label = dict_names_of_class[int(detect[-1])]
            p1 = (int(detect[0]), int(detect[1]))
            p2 = (int(detect[2]), int(detect[3]))
            
            cv2.rectangle(img, p1, p2, dict_colors_of_class[int(detect[-1])],2)
            cv2.putText(img, label, p1, cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                                    dict_colors_of_class[int(detect[-1])])
        return img

    def detect(self,frame):
        '''
        基础识别功能函数，输入frame，输出识别结果
        '''
        #print(f"detect.py,detect: {frame.shape}")
        img, original_img = self.prepare_image2(frame, self.img_size)
        #cv2.imshow('prepare img', img)
        #cv2.waitKey(200)
        input_img = Variable(img.type(self.Tensor))

        detections = None
        with torch.no_grad():
            detections = self.model(input_img)
            #print(f"detect.py,detect: detections.size {detections.size()}")
            #print(f"detect.py,detect: {detections[0]}")
            detections = non_max_suppression(detections, self.conf_thres, self.nms_thres)
        return detections

    def detect_picfile(self, pic_file):
        '''
        识别单张图片
        '''        
        process_img = None
        frame       = cv2.imread(pic_file)
        #
        prev_time   = time.time()
        detections  = self.detect(frame)
        #print(detections)
        if detections[0] is not None:
            process_img = self.add_layer1(frame, detections, self.img_size)
            #cv2.imshow('img', process_img)
            img_show = cv2.cvtColor(process_img, cv2.COLOR_BGR2RGB)
        else:
            img_show = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # fill results
        results                  = {}
        results['fps']           = 1 / (time.time()-prev_time)
        results['detections']    = detections
        results['processed_img'] = img_show
        results['height'], results['width'] = img_show.shape[:2]

    def write_detections_tofile(detections, dr_file):
        pass

    def evalute1(self, imgs_folder):
        """
        对图片进行检测并保存结果至detect_results文件夹中
        """
        for img_file in self.valid_path:
            img_name = os.path.basename(img_file)
            dr_file  = os.path.join(self.path_detection_results, 
                                    img_name.replace('.png','.txt').replace('.jpg','.txt'))
            img = cv2.imread(img_file)
            


    def detect_picfolder(self, pic_path):
        '''
        识别文件夹内的多张图片，一张一张地处理图片
        '''
        self.model.eval()
        dataloader = DataLoader(
                    ImageFolder(pic_path, img_size=self.img_size),
                    batch_size=1,
                    shuffle=False,
                    num_workers=0,
        )
        imgs           = []
        img_detections = []
        prev_time      = time.time()
        for batch_i, (img_paths, input_imgs) in enumerate(dataloader):
            print(f'batch_i:{batch_i}')
            frame = cv2.imread(img_paths[0])
            img_show = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            cv2.imshow('frame', frame)
            #cv2.waitKey(200)
            # fill results
            results = {}
            results['processed_img'] = img_show
            results['height'], results['width'] = img_show.shape[:2]
            
            '''
            # Configure input
            input_imgs = Variable(input_imgs.type(self.Tensor))
            # Get detections
            with torch.no_grad():
                detections = self.model(input_imgs)
                detections = non_max_suppression(detections, self.conf_thres, self.nms_thres)
                print(detections)
            # Log progress
            current_time = time.time()
            fps          = 1/(current_time-prev_time)
            prev_time    = current_time
            # Save image and detections
            imgs.extend(img_paths[0])
            img_detections.extend(detections)
            '''

path_weights     = r"H:\deepLearning\dataset\visdrone\Task 1 - Object Detection in Images\VisDrone2019-DET-train\yolov3\yolov3-tiny_99.pth"
path_class_names = r"H:\deepLearning\dataset\visdrone\Task 1 - Object Detection in Images\VisDrone2019-DET-train\yolov3\classes.names"
path_model_def   = r"config\yolov3-tiny.cfg"
path_data_config = r"config\vis_drone.data"
path_detection_results = r"H:\deepLearning\dataset\visdrone\Task 1 - Object Detection in Images\VisDrone2019-DET-val\detection_results"

if __name__ == "__main__":
    if os.path.exists(path_detection_results):
        shutil.rmtree(path_detection_results)
    else:
        os.mkdir(path_detection_results)

    yolodetect = YoloDetect(model=path_model_def,
                            data_config=path_data_config,
                            path_detection_results=path_detection_results,
                            file_pretrained_weights=path_weights)
    

    

