"""
将数据集的图片进行分割，获取较小尺寸的图片
"""
import os
import shutil
import math
import numpy as np

import cv2
from tqdm import tqdm
class ImageSplitter(object):
    """
    split the image in dataset into smaller images
    label format should be yolov3
    """
    def __init__(self, path_images, path_labels, path_out, size=(416,416), img_fmt='.jpg'):
        '''
        size = (width, height)
        '''
        self.path_images = path_images
        self.path_labels = path_labels
        self.outimg_size = size

        self.file_images = os.listdir(self.path_images)
        self.file_labels = os.listdir(self.path_labels)
        self.path_out    = path_out
        self.path_out_images = os.path.join(path_out, 'images')
        self.path_out_labels = os.path.join(path_out, 'labels')

        if os.path.exists(self.path_out):
            shutil.rmtree(self.path_out)
        os.mkdir(self.path_out)
        os.mkdir(self.path_out_images)
        os.mkdir(self.path_out_labels)
            
        self.file_log = os.path.join(path_out, 'log.txt')
        #self.f_log = open(self.file_log, 'w')

    def process(self):
        '''
        图片分割处理，根据bboxes分布位置，将其分割为若干个符合out image size的图片
        '''
        for f_image in tqdm(self.file_images):
            name = f_image
            #print(f"{f_image}")
            f_label = os.path.join(self.path_labels, f_image.replace('.jpg', '.txt').replace('.png', '.txt'))
            f_image = os.path.join(self.path_images, f_image)
            #print(f_label)
            if not os.path.exists(f_label):
                #self.f_log.write(f'Label not exist cooresponding to: {name}')
                print(f"{f_label} not exits")
                continue

            bboxes = self.__load_txt(f_label)
            img    = cv2.imread(f_image)
            img_h, img_w = img.shape[:2]
            #print(f"img.shape={img.shape}")
            bboxes       = self.unit_normlized2pixel(bboxes, img_w, img_h)
            bboxes       = self.ccwh_2_xyxy(bboxes, img_w, img_h)
            #print(bboxes)
            bboxes       = self.sort(bboxes, img_w, img_h)
            #print(bboxes)
            #self.splitter1(name.split('.')[0], img, bboxes)
            self.splitter2(name.split('.')[0], img, bboxes)

    def splitter1(self, name, img, bboxes):
        '''
        以bboxes的第一个bbox为中心，分割出大小为out image size的图片，并将位于该范围内的所有bbox保存
        '''
        img_h, img_w         = img.shape[:2]
        out_img_w, out_img_h = self.outimg_size

        bboxes_copy  = bboxes.copy()
        idx = 0
        #
        while len(labels) > 0:
            res_bboxes = np.zeros((1,4))
            #
            ref_bbox = bboxes[0]
            cx, cy = ref_bbox[0], ref_bbox[1]
            bw, bh = ref_bbox[2], ref_bbox[3]
            #
            if bw > self.outimg_size[0] or bh > self.outimg_size[1]:
                print(f"Error: image-{name} has a big bbox{(bw,bh)} larger than outimage size")
            
            # splitter image corner
            leftup    = int(max(int(cx - out_img_w/2), 0), max(int(cy - out_img_h/2), 0))
            rightdown = int(min(int(cx + math.ceil(out_img_w/2)), img_w), min(int(cy + math.ceil(out_img_h/2)), img_h))

            if leftup[0] == 0:
                rightdown[0] = out_img_w
            if leftup[1] == 0:
                rightdown[1] = out_img_h
            if rightdown[0] == img_w:
                leftup[0] = img_w - out_img_w
            if rightdown[1] == img_h:
                leftup[1] = img_h - out_img_h
            # 截取区域
            out_img = img[leftup[1]:rightdown[1], leftup[0]:rightdown[0]]
            # 根据截取区域，选择在该区域的所有bbox
            out_bboxes, msk = self.get_bboxes(bboxes_copy, leftup, rightdown)
            # 去掉已经选取过的bbox
            bboxes = np.delete(bboxes, msk, axis=0)

            # 输出结果并保存至相应文件夹
            self.output(f"{name}_{idx}", out_img, out_bboxes)
            idx += 1

    def splitter2(self, name, img, bboxes, grid=(2,2), overlap=0.1):
        '''
        将整个图片分割为特定行数和列数的图片，且保证一定的重合度
        Argments:
            -grid:   (img_num_row, img_num_col)
            -overlap: percent of overlapping
        '''
        img_h, img_w = img.shape[:2]
        step_h = img_h // 2
        step_w = img_w // 2
        outimg_h = int(step_h * (1 + overlap))
        outimg_w = int(step_w * (1 + overlap))
        #print(f"h:{outimg_h}, w:{outimg_w}")
        for i in range(grid[0]):
            for j in range(grid[1]):
                left  = j*step_w
                up    = i*step_h
                right = j*step_w + outimg_w
                down  = i*step_h + outimg_h
                # if rightdown corner outside image,then adjust leftup corner
                if right > img_w:
                    right = img_w
                    left  = right - outimg_w
                if down > img_h:
                    down = img_h
                    up   = down - outimg_h
                #
                out_img    = img[up:down, left:right]
                #print(f"{i}_{j}:{left,up,right,down}")
                out_bboxes,_ = self.get_bboxes(bboxes, (left,up), (right,down))
                # sub the offset of selected bboxes
                out_bboxes[:, [1,3]] -= left
                out_bboxes[:, [2,4]] -= up
                #print(out_bboxes)
                if len(out_bboxes) > 0:
                    # set annotation as yolo type
                    out_bboxes = self.xyxy_2_ccwh(out_bboxes, outimg_w, outimg_h)
                    out_bboxes = self.unit_pixel2normlized(out_bboxes, outimg_w, outimg_h)
                    # 输出结果并保存至相应文件夹
                    self.output(f"{name}_{i}_{j}", out_img, out_bboxes)

    def get_bboxes(self, bboxes, leftup, rightdown):
        msk = []
        for i, bbox in enumerate(bboxes):
            if bbox[1] >= leftup[0] and bbox[2] >= leftup[1] and bbox[3] < rightdown[0] and bbox[4] < rightdown[1]:
                msk.append(i)
        return bboxes[msk], msk

    def output(self, name, img, bboxes):
        file_img   = os.path.join(self.path_out_images, name+'.jpg')
        file_label = os.path.join(self.path_out_labels, name+'.txt')

        cv2.imwrite(file_img, img)
        #np.savetxt(file_label, bboxes, fmt="%.5f")
        with open(file_label, 'w') as f:
            for bbox in bboxes:
                #print(bbox)
                line = "{:d} {:.5f} {:.5f} {:.5f} {:.5f}\n".format(int(bbox[0]), 
                                                                bbox[1], bbox[2], bbox[3], bbox[4])
                #print(line)
                f.write(line)

    def unit_normlized2pixel(self, bboxes, img_w, img_h):
        '''
        anotation unit transfer from normlized to pixel
        '''
        bboxes[:, 1] *= img_w
        bboxes[:, 3] *= img_w
        bboxes[:, 2] *= img_h
        bboxes[:, 4] *= img_h
        bboxes = bboxes.astype(np.int32)
        return bboxes
        
    def unit_pixel2normlized(self, bboxes, img_w, img_h):
        #print(f"bboxes:{bboxes}, img_w:{img_w}")        
        bboxes = bboxes.astype(np.float32)
        bboxes[:, 1] /= img_w
        bboxes[:, 3] /= img_w
        bboxes[:, 2] /= img_h
        bboxes[:, 4] /= img_h
        #bboxes = bboxes.astype(np.float32)
        return bboxes

    def ccwh_2_xyxy(self, bboxes, img_w, img_h):
        '''
        annotation converter: cx,cy,w,h to x1,y1,x2,y2
        '''
        x1 = bboxes[:, 1] - bboxes[:, 3] // 2
        y1 = bboxes[:, 2] - bboxes[:, 4] // 2
        x2 = bboxes[:, 1] + bboxes[:, 3] // 2
        y2 = bboxes[:, 2] + bboxes[:, 4] // 2 
        bboxes[:, 1] = x1
        bboxes[:, 2] = y1
        bboxes[:, 3] = x2
        bboxes[:, 4] = y2
        return bboxes

    def xyxy_2_ccwh(self, bboxes, img_w, img_h):
        '''
        '''
        cx = (bboxes[:,1] + bboxes[:,3]) // 2
        cy = (bboxes[:,2] + bboxes[:,4]) // 2
        w  = bboxes[:,3] - bboxes[:,1]
        h  = bboxes[:,4] - bboxes[:,2]
        bboxes[:,1] = cx
        bboxes[:,2] = cy
        bboxes[:,3] = w
        bboxes[:,4] = h
        return bboxes

    def sort(self, bboxes, img_w, img_h):
        '''
        对图片内的所有bbox以其中心点进行排序，中心点y坐标越小越靠前，其次x坐标越小越靠前。
        '''
        #bboxes.sort(axis=0)
        bboxes_list = list(bboxes)
        bboxes_list.sort(key=lambda x:x[1]*img_w + x[2])
        bboxes = np.array(bboxes_list)
        return bboxes


    def __load_txt(self, file_path):
        '''
        使用numpy加载txt，只提取bbox信息
        Return:
            - 二维array
        '''
        datas = np.loadtxt(file_path)
        if len(datas.shape) == 1:
            datas = np.expand_dims(datas, 0)
        return datas

    def __del__(self):
        #self.f_log.close()
        pass

if __name__ == '__main__':
    path_images = r"D:\CETCA_DeepLearning\CETCA_UAVDataSet\dataset_htswinter_ruanjiangongchengbu2\images"
    path_labels = r"D:\CETCA_DeepLearning\CETCA_UAVDataSet\dataset_htswinter_ruanjiangongchengbu2\labels"
    path_out    = r"D:\CETCA_DeepLearning\CETCA_UAVDataSet\dataset_htswinter_ruanjiangongchengbu2\outputs"
    imgsplitter = ImageSplitter(path_images, path_labels, path_out)
    imgsplitter.process()