"""
mAP calculator
"""
import os,sys
import glob,shutil
import json

import cv2
import numpy as np
import matplotlib.pyplot as plt


# 图片坐标系
'''
    0,0 ------> x (width)
     |
     |  (Left,Top)
     |      *_________
     |      |         |
            |         |
     y      |_________|
  (height)            *
                (Right,Bottom)
'''
class MAP_Calculator(object):
    """
    folder(path_base):
    |--ground_truth
    |--detection_results
    |--images
    """
    MINOVERLAP = 0.5
    def __init__(self):
        self.path_base = r"H:\deepLearning\dataset\visdrone\Task 1 - Object Detection in Images\VisDrone2019-DET-train\mAP"
        self.path_gt   = os.path.join(self.path_base, 'input', 'ground_truth')
        self.path_dr   = os.path.join(self.path_base, 'input', 'detection_results')
        self.path_img  = os.path.join(self.path_base, 'input', 'images')
        self.file_class_names = os.path.join(self.path_base, 'input', 'class.names')

        if not os.path.exists(self.path_gt):
            print("Error: no ground_truth folder exists")
        if not os.path.exists(self.path_dr):
            print("Error: no detection_results folder exists")
        if not os.path.exists(self.path_img):
            print("Error: no iamges folder exists")

        #创建output文件夹
        self.path_output = os.path.join(self.path_base, 'output')
        self.path_output_temp   = os.path.join(self.path_output, 'temp')
        self.path_output_class  = os.path.join(self.path_output, 'class')
        if os.path.exists(self.path_output):
            shutil.rmtree(self.path_output)
        os.mkdir(self.path_output)
        os.mkdir(self.path_output_temp)
        os.mkdir(self.path_output_class)

        # 记录每类的目标数量
        self.gt_counter_per_class = {}
        # 记录出现每类目标的图片数量，即一个类别共出现在多少张图片中
        self.gt_counter_images_per_class = {}
        # ground_truth中出现的所有目标的类别的idx
        self.gt_classes_idx = []
        
        self.class_names = self.load_classnames()
        #print(self.class_names)
        self.ap_dictionary = {}

        # 
        self.count_true_positives = {}

    def load_classnames(self):
        with open(self.file_class_names,'r') as f:
            lines = f.readlines()
            class_names = [line.rstrip() for line in lines]
        return class_names

    def load_gt_data(self):
        '''
        从ground_truth文件夹加载数据，得到：
        self.gt_counter_images_per_class
        '''
        #self.files_gt = os.listdir(self.path_gt)
        self.files_gt = glob.glob(self.path_gt + r'\*.txt')
        self.files_gt.sort()

        # 用于存放ground_truth信息，便于后续AP计算时使用
        self.gt_files_infos = {}
        for txt_file in self.files_gt:
            file_id     = os.path.basename(txt_file).split(".txt",1)[0]
            txt_path_gt = os.path.join(self.path_gt, txt_file)
            txt_path_dr = os.path.join(self.path_dr, txt_file)
            # 判断与gt同名的dr文件是否存在
            if not os.path.exists(txt_path_dr):
                print(f"Error: corespond file {txt_file} not exist in detection_results folder")
                continue
            #
            already_seen_classes = []
            #
            lines = np.loadtxt(txt_path_gt)
            # 记录每个gt_file的信息
            infos = []
            for line in lines:
                class_idx = int(line[0])
                if class_idx in self.gt_counter_per_class:
                    self.gt_counter_per_class[class_idx] += 1
                else:
                    self.gt_counter_per_class[class_idx] = 1

                if class_idx not in already_seen_classes:
                    if class_idx in self.gt_counter_images_per_class:
                        self.gt_counter_images_per_class[class_idx] += 1
                    else:
                        self.gt_counter_images_per_class[class_idx] = 1
                    already_seen_classes.append(class_idx)
                infos.append({'idx':class_idx, 'bbox':list(line[1:]), 'used':False})
            #print(f"file={txt_file}")
            #print(self.gt_counter_images_per_class)

            self.gt_files_infos[file_id] = infos
        print(f'gt info={self.gt_files_infos}')
        self.gt_classes_idx = list(self.gt_counter_per_class.keys())
        self.n_classes      = len(self.gt_classes_idx)

    def load_dr_data(self):
        '''
        从detection_results文件夹中加载数据，获取每个类别所有目标的信息(confidence,bbox)，并按照confidence由大到小
        对目标信息进行排序，并将每个类别的目标信息存放到同名的.json文件中。
        '''
        self.files_dr = glob.glob(self.path_dr + r"\*.txt")
        self.files_dr.sort()

        #把每个类别的所有目标的信息汇总起来
        for class_idx in self.gt_classes_idx:
            bounding_boxes = []
            for txt_file in self.files_dr:
                #print(f'txt_file={os.path.basename(txt_file)}')
                file_id = os.path.basename(txt_file).split(".txt",1)[0]
                txt_path_dr = os.path.join(self.path_dr, txt_file)
                txt_path_gt = os.path.join(self.path_gt, txt_file)

                if not os.path.exists(txt_path_gt):
                    print(f"Error: corespond file {txt_file} not exist in ground_truth folder")
                    continue
                #
                lines = np.loadtxt(txt_path_dr)
                for line in lines:
                    tmp_class_name_idx = int(line[0])
                    confidence         = float(line[1])
                    bbox               = list(line[2:])
                    #print(f'conf:={type(confidence)},bbox={type(bbox)}')

                    if tmp_class_name_idx == class_idx:
                        bounding_boxes.append({"confidence":confidence, "file_id":file_id, "bbox":bbox})
            #
            #print(f'{class_idx}:len is {len(bounding_boxes)}')
            # 根据bbox的confidence（置信度）对detection_results进行排序
            bounding_boxes.sort(key=lambda x:x['confidence'], reverse=True)
            with open(os.path.join(self.path_output_temp,f'{class_idx}_{self.class_names[class_idx]}_dr.json'), 'w') as f:
                json.dump(bounding_boxes, f)

    def ap_calculate(self):
        '''
         Calculate the AP for each class
         1. 图片中的每个gt_bbox只能被匹配一次，也就是说若有多个同类目标的dr_bbox都与一个gt_bbox匹配了，那么只有第一个匹配上的dr_bbox的tp=True，
            所以在调试的时候非常注意，每次本函数运行前都需要重新运行一下load_gt_data()以重新加载self.gt_files_infos变量。
        '''
        sum_AP = 0.0
        # open file to store the output
        with open(os.path.join(self.path_output, 'output.txt'), 'w') as output_file:
            output_file.write("# AP and precision/recall per class\n")
            # 
            for class_idx in self.gt_classes_idx:
                self.count_true_positives[class_idx] = 0
                # Load detection-results of that class
                dr_file = os.path.join(self.path_output_temp, f'{class_idx}_{self.class_names[class_idx]}_dr.json')
                dr_data = json.load(open(dr_file))

                # Assign detection-results to ground-truth objects
                # nd代表class_idx类别的目标数量
                nd = len(dr_data)
                tp = [0] * nd
                fp = [0] * nd
                for idx, detection in enumerate(dr_data):
                    file_id = detection['file_id']
                    # open ground-truth with that file_id
                    gt_file = os.path.join(self.path_gt, file_id+'.txt')

                    #gt_datas = np.loadtxt(gt_file)
                    gt_datas = self.gt_files_infos[file_id]

                    ovmax = -1
                    gt_match = -1
                    # load detected object bounding-box
                    bbox_dr = detection['bbox']

                    #for gt_data in gt_datas()
                    for gt_data in gt_datas:
                        # look for a class index match
                        if gt_data['idx'] == class_idx:
                            #bbox_gt = gt_data[1:]
                            bbox_gt = gt_data['bbox']
                            bi = [max(bbox_dr[0],bbox_gt[0]), max(bbox_dr[1],bbox_gt[1]),
                                  max(bbox_dr[2],bbox_gt[2]), max(bbox_dr[3],bbox_gt[3])]
                            iw = bi[2] - bi[0] + 1
                            ih = bi[3] - bi[1] + 1
                            if iw > 0 and ih > 0:
                                ua = (bbox_dr[2]-bbox_dr[0]+1) * (bbox_dr[3]-bbox_dr[1]+1) +\
                                     (bbox_gt[2]-bbox_gt[0]+1) * (bbox_gt[3]-bbox_gt[1]+1) -\
                                     iw * ih
                                ov = iw*ih / ua
                                if ov > ovmax:
                                    ovmax = ov 
                                    # 由于gt_data是dict类型，后续改变gt_match就会对gt_data进行改变
                                    gt_match = gt_data
                    # set minimum overlap
                    min_overlap = self.MINOVERLAP

                    # 计算TP,FP
                    if ovmax >= min_overlap:
                        if not bool(gt_match['used']):
                            gt_match['used'] = True
                            tp[idx] = 1
                            self.count_true_positives[class_idx] += 1
                            print(f'match')
                        else:
                            pass
                            fp[idx] = 1
                            print(f'repeat match')
                    else:
                        fp[idx] = 1
                        if ovmax > 0:
                            print('insufficient overlap')
                        else:
                            print(f'ovmax={ovmax}')

                # compute precision/recall
                cumsum = 0
                for idx, val in enumerate(fp):
                    fp[idx] += cumsum
                    cumsum += val
                cumsum = 0
                for idx, val in enumerate(tp):
                    tp[idx] += cumsum
                    cumsum += val
                print(f'tp={tp}')
                print(f'fp={fp}')
                rec = tp[:]
                for idx, val in enumerate(tp):
                    rec[idx] = float(tp[idx]) / self.gt_counter_per_class[class_idx]
                prec = tp[:]
                for idx, val in enumerate(tp):
                    prec[idx] = float(tp[idx]) / (fp[idx] + tp[idx])

                ap, mrec, mprec = self.ap(rec[:], prec[:])
                sum_AP += ap

                #
                text = "{0:.2f}%".format(ap*100) + " = " + self.class_names[class_idx] + " AP"
                rounded_prec = ['%.2f' % elem for elem in prec]
                rounded_rec  = ['%.2f' % elem for elem in rec]
                output_file.write(text + "\n Precision: " + str(rounded_prec) + 
                                        "\n Recall: " + str(rounded_rec))
                output_file.write("\n")
                self.ap_dictionary[class_idx] = ap

    def ap(self, rec, prec):
        '''
        #Params
        rec: recall, list type
        prec: precision, list type
        '''
        rec.insert(0, 0.0)
        rec.append(1.0)
        mrec = rec[:]
        prec.insert(0, 0.0)
        prec.append(0.0)
        mpre = prec[:]

        # This part makes the precision monotonically decreasing（单调递减）
        # (goes from the end to the beginning)
        for i in range(len(mpre)-2, -1, -1):
            mpre[i] = max(mpre[i], mpre[i+1])

        # This part creates a list of indexes where the recall changes
        i_list = []
        for i in range(1, len(mrec)):
            if mrec[i] != mrec[i-1]:
                i_list.append(i) # if it was matlab would be i + 1
        # The Average Precision (AP) is the area under the curve
        # (numerical integration)
        ap = 0.0
        for i in i_list:
            ap += ((mrec[i]-mrec[i-1])*mpre[i])
        return ap, mrec, mpre

    def draw_ap(self):
        pass
        

if __name__ == '__main__':
    mc = MAP_Calculator()
