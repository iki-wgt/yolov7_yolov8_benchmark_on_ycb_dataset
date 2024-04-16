import cv2
import os 
import torch
import sys
import random
import argparse
import os
import platform
import sys
from pathlib import Path

import torch
from numpyencoder import NumpyEncoder
sys.path.append("/root/src/python/")
import param_singleton
import time

from yolov9_utils.utils.torch_utils import select_device
from yolov9_utils.models.common import DetectMultiBackend
from yolov9_utils.utils.general import (LOGGER,Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from yolov9_utils.utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams

from yolov9_utils.utils.plots import Annotator, colors, save_one_box

class Yolov9Prediction:
    def __init__(self): 

        self.param_singleton = param_singleton.ParamSingleton()
        print(id(self.param_singleton))
        
        # inits params
        self.docker_dataset_path = self.param_singleton.get_docker_dataset_path()
        self.docker_single_test_dataset_paths = self.param_singleton.get_docker_single_test_dataset_path()
        self.docker_model_path = self.param_singleton.get_docker_model_path()
        self.docker_predictions_path = self.param_singleton.get_docker_predictions_path()
        self.docker_evaluation_path = self.param_singleton.get_docker_evaluation_path()
        self.dataset_scenes = self.param_singleton.get_dataset_scenes()
        self.model_name = self.param_singleton.get_model_name()
        self.save_predictions = self.param_singleton.get_save_predictions()

        self.model_path = os.path.join(self.docker_model_path,self.model_name,"weights","best.pt")

        print("--- Loading Model ---")
        self.__load_model()
    
            
    def __load_model(self):
        self.device = select_device('0')
        self.model= DetectMultiBackend(self.model_path,device=self.device)
        self.stride, self.names, self.pt = self.model.stride, self.model.names, self.model.pt

    def set_weight_name(self, weight_name):
        self.model_path = os.path.join(self.param_singleton.get_docker_model_path(),self.model_name,"weights", weight_name)
        self.__load_model()
        print(self.model_path)

    def __save_predictions(self, dataset_scene, obj_scene, im0s, norm_det_per_image, path, names):
        # Build prediction structure
        label_path = os.path.join(self.docker_predictions_path,self.model_name,dataset_scene,'labels',obj_scene)
        image_path = os.path.join(self.docker_predictions_path,self.model_name,dataset_scene,'images',obj_scene)
        img_name = path.split('/')[-1:][0]
        if not os.path.isdir(image_path):
            os.makedirs(image_path)
            os.makedirs(label_path)

        # write predicted images
        cv2.imwrite(os.path.join(image_path,img_name),im0s)

        # Write labels (id,conf,x_center,y_center,widht,height)
        with open(os.path.join(label_path,f"{img_name.split('.')[0]}.txt"),'w') as f :
            for norm_det in norm_det_per_image:
                id = norm_det[0]
                conf = norm_det[1]
                x,y,w,h = norm_det[2]
                f.write(f"{id} {conf} {x} {y} {w} {h}\n")

        self.__create_classes_txt(names)

    def __predict_per_dataset_scene(self,dataset_scene, names, conf_thres, iou_thres):
        
        imgsz = check_img_size((640, 640), s=self.stride)  # check image size
        bs = 1
        
        dataset_scene_path = os.path.join(self.docker_dataset_path,dataset_scene,'labeled_images','images')
        obj_scenes = sorted(os.listdir(dataset_scene_path))
        half = self.device.type != 'cpu'
        images = []
        times = []
        for obj_scene in obj_scenes: 
            dataset = LoadImages(os.path.join(dataset_scene_path,obj_scene), img_size=imgsz, stride=self.stride, auto=self.pt)
            self.model.warmup(imgsz=(1 if self.pt or self.model.triton else bs, 3, *imgsz))  # warmup

            seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
            for path, im, im0s, vid_cap, s in dataset:
                with dt[0]:
                    im = torch.from_numpy(im).to(self.model.device)
                    im = im.half() if self.model.fp16 else im.float()  # uint8 to fp16/32
                    im /= 255  # 0 - 255 to 0.0 - 1.0
                    if len(im.shape) == 3:
                        im = im[None]  # expand for batch dim
                with dt[1]:
                    pred = self.model(im, augment=False, visualize=False)
 
                with dt[2]:
                    pred = non_max_suppression(pred, conf_thres, iou_thres)

                # Process predictions
                for i, det in enumerate(pred):  # per image
                    seen += 1
                    p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
                    gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                    annotator = Annotator(im0, line_width=1, example=str(names))

                    if len(det):
                        # Rescale boxes from img_size to im0 size
                        det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
                        # Print results
                        for c in det[:, 5].unique():
                            n = (det[:, 5] == c).sum()  # detections per class
                            s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                        norm_det_per_image = []

                        # Write results
                        for *xyxy, conf, cls in reversed(det):
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            norm_det_per_image.append([int(cls), float(conf), xywh])
                            c = int(cls)
                            label = f'{names[c]} {conf:.2f}'
                            annotator.box_label(xyxy, label, color=colors(c,True))
                
                im0 = annotator.result()
                self.__save_predictions(dataset_scene, obj_scene, im0, norm_det_per_image, path, names)


            # if half:
            #     model.half() # possible if cuda available
            # for path, img, im0s, vid_cap in dataset:
            #     start_time = time.time()
            #     img = torch.from_numpy(img).to(self.device)
            #     img = img.half() if half else img.float()  # uint8 to fp16/32
            #     img/=255.0
            #     if img.ndimension() == 3:
            #             img = img.unsqueeze(0)
            #     with torch.no_grad(): # no grad calcs
            #             pred = model(img)[0]
            #     det_per_image = non_max_suppression(pred,conf_thres=conf_thres,iou_thres=iou_thres) 

            #     norm_det_per_image = []
            #     for i, det in enumerate(det_per_image):
            #             gn = torch.tensor(im0s.shape)[[1, 0, 1, 0]]
            #             det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0s.shape).round()
            #             for *xyxy, conf, cls in reversed(det):
            #                 label = f'{names[int(cls)]} {conf:.2f}'
            #                 xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
            #                 norm_det_per_image.append([int(cls), float(conf), xywh])
            #                 plot_one_box(xyxy, im0s, label=label, color=colors[int(cls)], line_thickness=1)
            #     end_time = time.time()
            #     times.append((end_time-start_time)*1000)
            #     # print(f"Prediction time: {(end_time-start_time)*1000}")
            #     images.append(cv2.cvtColor(im0s,cv2.COLOR_BGR2RGB))
            #     if self.save_predictions:
            #         self.__save_predictions(dataset_scene, obj_scene, im0s, norm_det_per_image, path, names)

        
    
    def __create_classes_txt(self,names):
        with open(os.path.join(self.docker_predictions_path,self.model_name,"classes.txt"),'w') as f :
            for name in names:
                f.write(f'{name}\n')
    
    def __write_pred_time(self,pred_time):
        print("Average prediction time: ",pred_time)
        with open(os.path.join(self.docker_predictions_path,self.model_name,"pred_time.txt"),'w') as f :
            f.write(f'Prediction_Time: {pred_time} ms\n')
    
    def __save_images(self, dataset_name, path, im0s):
        images_path = os.path.join(self.docker_predictions_path, self.model_name, dataset_name,'images')
        img_name = path.split('/')[-1:][0]
                
        if not os.path.isdir(images_path):
            os.makedirs(images_path)

        cv2.imwrite(os.path.join(images_path,img_name),im0s)

    def __save_labels(self, dataset_name, path, norm_det_per_image, names):
        # Build prediction structure
        label_path = os.path.join(self.docker_predictions_path,self.model_name,dataset_name,'labels')
        img_name = path.split('/')[-1:][0]
        if not os.path.isdir(label_path):
            os.makedirs(label_path)

        # Write labels (id,conf,x_center,y_center,widht,height)
        with open(os.path.join(label_path,f"{img_name.split('.')[0]}.txt"),'w') as f :
            for norm_det in norm_det_per_image:
                id = norm_det[0]
                conf = norm_det[1]
                x,y,w,h = norm_det[2]
                f.write(f"{id} {conf} {x} {y} {w} {h}\n")
    

    def predict(self):
        print("--- Starting YOLOV9 Prediction---")
        # Inspired by: https://github.com/WongKinYiu/yolov7/blob/3b41c2cc709628a8c1966931e696b14c11d6db0c/detect.py
        # model = self.model.get_model()
        names = self.model.names if hasattr(self.model, 'module') else self.model.names
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
        all_images = []
        conf_thres = self.param_singleton.get_conf_thres()
        iou_thres = self.param_singleton.get_iou_thres()
        pred_times = []

        # Predict special dataset
        for dataset_scene in self.dataset_scenes:
            self.__predict_per_dataset_scene(dataset_scene, names, conf_thres, iou_thres)
            # all_images += images
            # pred_times.append(pred_time)

        # Predict for testdataset
        # self.__predict_specific_test_dataset(self.param_singleton.get_single_test_dataset_name(), self.model, conf_thres, iou_thres, names, colors)

        if self.save_predictions:
            self.__create_classes_txt(names)
            print("The predictions are saved in: ",os.path.join(self.docker_predictions_path,self.model_name))
            # self.__write_pred_time(sum(pred_times)/len(pred_times))

        # self.plots_utils.plot_images(all_images)
