from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *

import os
import sys
import time
import datetime
import argparse

from PIL import Image

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import cv2
import av

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator

class YoloDetect():
    def __init__(self, model="yolov3-tiny",
                       class_names="yolov3/data/coco.names",
                       file_pretrained_weights="yolov3/weights/yolov3-tiny.weights",
                       img_size=608,
                ):
        self.flag_detect   = False
        self.flag_captured = False
        self.frame         = None
        self.detections    = [None]
        #
        self.class_names  = load_classes(class_names)
        self.class_colors = generate_colors(len(self.class_names))
        self.img_size   = img_size
        self.conf_thres = 0.8
        self.nms_thres  = 0.4
        # choose GPU or CPU
        self.Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 视频流fps > 识别速度(fps)，通过抽帧采样后在识别
        self.detect_interval = 5
        self.detect_cnt = 0
        #
        model_def = ""
        try:
            model_def = "yolov3/config/{0}.cfg".format(model)
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

    def prepare_image1(self, img, in_dim):
        '''
        图片预处理：将原图的右下角图片抠出来，输入到神经网络中
        '''
        original_img = img
        # Convert opencv numpy to tensor
        # tensor dim:(batch, channel[RGB], height, weight)
        # opencv dim:(height,weight,channel[BGR])
        img_ = img[-in_dim:,-in_dim:,::-1].transpose((2,0,1)).copy()
        img_ = torch.from_numpy(img_).float().div(255.0).unsqueeze(0)
        return img_,original_img
    def prepare_image2(self, img, in_dim):
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
    def add_layer2(self, img, detections, in_dim):
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

    def add_layer3(self, img, detections, in_dim):
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

    def set_detectflag(self, flag):
        self.flag_detect = flag

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

    def detect_picfile(self, pic_file, fun_callback):
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
            process_img = self.add_layer2(frame, detections, self.img_size)
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
        fun_callback(results)        

    def detect_picfolder1(self, pic_path, fun_callback):
        '''
        识别文件夹内的多张图片，一张一张地处理图片
        未解决：使用dataloader与fun_callback好像不能同时用
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
            fun_callback(results)
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
            #fun_callback(fps)
            # Save image and detections
            imgs.extend(img_paths[0])
            img_detections.extend(detections)
            '''

    def set_callback_function(self,fun_callback):
        self.fun_callback = fun_callback

    def set_video_file(self, video_file, decode='h264'):
        '''
        video_file = 0:       open webcam
        video_file = "../..": open video file according path
        '''
        if decode == 'h264':
            self.cap = cv2.VideoCapture(video_file)
        elif decode == 'h265':
            self.container = av.open(video_file)
            self.av_lst = self.container.decode(video=0)
            #stream = self.container.streams.video[0]
            #total_frame = float(self.container.streams.video[0].average_rate)

    def video_parser(self):
        '''
        使用opencv解析H.264视频流
        '''
        ret, frame = self.cap.read()   
        #cv2.imshow('frame', frame)  
        cv2.waitKey(1)
        self.detect_cnt += 1
        if ret:
            if self.detect_cnt % self.detect_interval == 0:
                self.frame = frame.copy()
                self.flag_captured = True
            else:
                self.flag_captured = False

    def detect_video(self):   
        '''
        使用检测器识别视频的单帧图片
        '''
        results            = {}     # 处理结果作为回调函数的参数，返回

        if self.frame is None:
            return
        #
        img = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)

        if self.flag_detect:
            if self.flag_captured:
                prev_time   = time.time()
                self.detections  = self.detect(img)
                results['fps']       = 1 / (time.time()-prev_time)
            else:
                img_show       = img
                results['fps'] = 0
            # 将识别结果叠加至图像
            if self.detections[0] is not None:
                process_img = self.add_layer2(img, self.detections, self.img_size)
                #cv2.imshow('img', process_img)
                img_show = process_img
            else:
                img_show = img
        else:
            img_show = img
            results['fps'] = 0

        results['processed_img'] = img_show
        results['height'], results['width'] = img_show.shape[:2]
        self.fun_callback(results)     

    def detect_video2(self):
        '''
        使用pyav解析H.265视频流
        '''
        frame = self.av_lst.__next__()
        frame = np.array(frame.to_image()) 
        frame =  cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        #cv2.imshow('frame', frame)
        cv2.waitKey(1)
        results            = {}     # 处理结果作为回调函数的参数，返回
        #
        if not self.flag_captured:
            img_show       = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results['fps'] = 0
        else:
            #
            prev_time   = time.time()
            detections  = self.detect(frame)
            if detections[0] is not None:
                process_img = self.add_layer2(frame, detections, self.img_size)
                #cv2.imshow('img', process_img)
                img_show = cv2.cvtColor(process_img, cv2.COLOR_BGR2RGB)
            else:
                img_show = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results['fps']       = 1 / (time.time()-prev_time)

        results['processed_img'] = img_show
        results['height'], results['width'] = img_show.shape[:2]
        self.fun_callback(results)  

    def evaluate(self, model, path, iou_thres, conf_thres,
                       nms_thres, img_size, batch_size):
        '''

        '''
        model.eval()
        # Get dataloader
        dataset = ListDataset(path, img_size=img_size, augment=False, multiscale=False)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=1, collate_fn=dataset.collate_fn
        )

        Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

        labels = []
        sample_metrics = []  # List of tuples (TP, confs, pred)
        for batch_i, (_, imgs, targets) in enumerate(tqdm.tqdm(dataloader, desc="Detecting objects")):

            # Extract labels
            labels += targets[:, 1].tolist()
            # Rescale target
            targets[:, 2:] = xywh2xyxy(targets[:, 2:])
            targets[:, 2:] *= img_size

            imgs = Variable(imgs.type(Tensor), requires_grad=False)

            with torch.no_grad():
                outputs = model(imgs)
                outputs = non_max_suppression(outputs, conf_thres=conf_thres, nms_thres=nms_thres)

            sample_metrics += get_batch_statistics(outputs, targets, iou_threshold=iou_thres)

        # Concatenate sample statistics
        true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
        precision, recall, AP, f1, ap_class = ap_per_class(true_positives, pred_scores, pred_labels, labels)

        return precision, recall, AP, f1, ap_class
    

