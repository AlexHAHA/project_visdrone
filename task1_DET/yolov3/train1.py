from __future__ import division

from models import *
from yolov3.test import evaluate
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *

from terminaltables import AsciiTable

import os
import sys
import time
import datetime
import argparse

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim

from PyQt5.QtCore import QTimer
class YoloTrain():
    def __init__(self,fun_callback,     #回调函数
                      file_data_config, #训练集
                      model,            #yolo模型结构
                      file_pretrained_weights,
                      epochs=100,                 #number of epochs
                      batch_size=8,               #size of each image batch
                      img_size=416,               #size of each image dimension
                      n_cpu=8,                    #number of cpu threads to use during batch generation
                      gradient_accumulations=2,   #number of gradient accums before step
                      checkpoint_interval=2,
                      ):
        self.fun_callback = fun_callback
        # 使用QTimer，避免主界面卡死
        self.timer = QTimer()

        self.epochs = epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gradient_accumulations = gradient_accumulations
        self.checkpoint_interval    = checkpoint_interval
        #print(file_data_config)
        self.data_config = parse_data_config(file_data_config)
        self.train_path  = self.data_config["train"]
        self.valid_path  = self.data_config["valid"]
        self.class_names = load_classes(self.data_config["names"])
        
        #print(f"train1.py {self.data_config}")

        #
        model_def = ""
        try:
            if model == "yolov3-tiny_xj1":
                model_def = "yolov3/config/yolov3-tiny_xj1.cfg"
            elif model == "yolov3_xj1":
                model_def = "yolov3/config/yolov3_xj1.cfg"
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
        # get dataloader
        dataset = ListDataset(self.train_path, augment=True, multiscale=True)
        self.dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=n_cpu,
            pin_memory=True,
            collate_fn=dataset.collate_fn,
        )
        self.optimizer = torch.optim.Adam(self.model.parameters())

        self.metrics_dict = {}
        self.metrics = [
            "grid_size",
            "loss",
            "x",
            "y",
            "w",
            "h",
            "conf",
            "cls",
            "cls_acc",
            "recall50",
            "recall75",
            "precision",
            "conf_obj",
            "conf_noobj",
        ]
        print('loaded weights')
        
    def train(self):
        results            = {}     # 处理结果作为回调函数的参数，返回
        for epoch in range(self.epochs):
            self.model.train()
            start_time = time.time()
            for batch_i, (_,imgs,targets) in enumerate(self.dataloader):
                batches_done = len(self.dataloader) * epoch + batch_i

                imgs = Variable(imgs.to(self.device))
                targets = Variable(targets.to(self.device), requires_grad=False)

                # loss
                loss, outputs = self.model(imgs, targets)
                loss.backward()

                if batches_done % self.gradient_accumulations:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                #print(f"train.py:loss {loss}")
                # Log progress
                self.log_str = "---- [Epoch %d/%d, Batch %d/%d] ----\n" % (epoch, self.epochs, batch_i, len(self.dataloader))
                print(self.log_str)
                results['epoch'] = epoch
                results['batch'] = batch_i
                results['loss']  = loss
            
                # call back function
                self.fun_callback(results)
            if epoch % self.checkpoint_interval == 0:
                torch.save(self.model.state_dict(), r"C:\Users\admin\CETCA_deeplearning\tutorial_pyqt5\CETCA_dp\yolov3\checkpoints\yolov3_ckpt_%d.pth" % epoch)
                print(f'epoch:{epoch}')
    def train_test(self):
        results = {}
        for i in range(10):       
            results['epoch'] = i
            results['batch'] = i
            results['loss']  = i
            self.fun_callback(results)
            time.sleep(0.1)
