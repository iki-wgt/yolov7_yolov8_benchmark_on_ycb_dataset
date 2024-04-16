# Created by Vijay Rajagopal
import os
from review_object_detection_metrics.src.evaluators.tube_evaluator import TubeEvaluator
import logging
from review_object_detection_metrics.src.utils import converter
from review_object_detection_metrics.src.utils.enumerators import (BBFormat, BBType, CoordinatesType,
                                   MethodAveragePrecision)
import review_object_detection_metrics.src.evaluators.coco_evaluator as coco_evaluator
import review_object_detection_metrics.src.evaluators.pascal_voc_evaluator as pascal_voc_evaluator
import review_object_detection_metrics.src.utils.converter as converter
import review_object_detection_metrics.src.utils.general_utils as general_utils

class MAP_Metric():

    def __init__(self):
        self.anno_gt_path = None
        self.anno_det_path = None
        self.img_gt_path = None
        self.format_gt = None
        self.format_det = None
        self.coor_det = None
        self.metric = None
        self.classnames_path = None
        self.threshold = None
        self.plot_flag = None
        self.save_path = None

    def set_anno_gt_path(self, path:str) -> None:
        """
        Directory or file of the ground-truth annotations
        
        @param path 
        """
        self.anno_gt_path = path

    def set_anno_det_path(self, path:str) -> None:
        """
        Directory or file of the detection annotations
        
        @param path 
        """
        self.anno_det_path = path

    def set_img_gt_path_(self, path:str) -> None:
        """
        Directory of the gt images
        
        @param path 
        """
        self.img_gt_path = path
    def set_format_gt(self,format:str) -> None:
        """
        Takes in keyword that represents the format that the ground truth
        will be in (Available formats: coco, voc, imagenet, labelme, openimg, yolo, absolute, cvat, tube)        
        
        @param format: available coco, voc, imagenet, labelme, openimg, yolo, absolute, cvat, tube
        """
        self.format_gt = format
    def set_format_det(self, format:str) -> None:
        """
        Takes in keyword that represents the format that the detection will
        be in (Available inputs: coco, xyrb (x,y,x2,y2), or xywh (x,y,width,height), xcycwh (x_center, y_center, w, h))
        
        @param: format available: coco, xyrb (x,y,x2,y2), or xywh (x,y,width,height), xcycwh (x_center, y_center, w, h)
        """
        self.format_det = format
    def set_coord_det(self, coordinate_system:str) -> None:
        """
        Argument is associated with format_det and represents 
        the coordinate system of the detection files. Abolute or relative
        (Available inputs: abs, rel)
        
        @param coordinate_system can be rel or abs
        """
        self.coor_det = coordinate_system

    def set_metric(self, metric_type:str) -> None:
        """
        Takes in keyword that represents the metric that will be used (Available options: coco, voc2007, voc2012)

        @param metric_type
        """
        self.metric = metric_type
    def set_classnames_path(self,path:str) -> None:
        """
        The path of the file that contains the class names for this detection 
        (NOTE: required if ground truth format is specified as yolo or if detection files have class IDs)

        @param path to the classes.txt    
        """
        self.classnames_path = path

    def set_threshold(self, threshold=0.5)->None:
        """
        A decimal from 1 to 0 representing the IoU threshold for a given metric
 
        @param threshold IOU threshold, default 0.5
        """
        self.threshold = threshold
        
    def set_plot_flag(self, plot:bool)-> None:
        """
        A flag for creating a Precision x Recall graph for the given metric that it is being used in

        @param plot
        """
        self.plot_flag = plot

    def set_save_path(self,path:str)->None:
        """
        An optional parameter used to specify where the visual output will be saved to. Required when plot is specified
        
        @param path where the output should be safed
        """
        self.save_path = path

    def __verifyArgs(self)->None:
        if not os.path.exists(self.anno_gt_path):
            raise Exception('--anno_gt path does not exist!')
        if not os.path.exists(self.anno_det_path):
            raise Exception('--anno_det path does not exist!')
        if self.threshold > 1 or self.threshold < 0:
            raise Exception('Incorrect range for threshold (0-1)')
        if self.plot_flag and self.save_path == '':
            raise Exception("Precision-Recall graph specified but no save path given!")
        if self.format_gt == 'yolo' and self.classnames_path == '':
            raise Exception("The ground truth format specified (%s), requires a name file. Specify with --name."%self.format_gt)
        if 'tube' == self.format_gt != self.format_det:
            raise Exception("Spatio-Temporal Tube AP specified in one format parameter but not other!")

        if self.img_gt_path == '':
            logging.warning("Image path not specified. Assuming path is same as ground truth annotations.")
            self.img_gt_path = self.anno_gt_path

        if self.classnames_path == '':
            logging.warning("Names property empty so assuming detection format is class_id based.")

        if not os.path.exists(self.save_path):
            logging.warning("save-path directory %s is not found. Attempting to create folder..."%(self.save_path))
            try:
                os.mkdir(self.save_path)
            except:
                logging.error("Could not create directory! Exiting")
                raise Exception()


    def calculate_mAP(self):
        self.__verifyArgs()

            # collect ground truth labels:
        if self.format_gt == 'coco':
            gt_anno = converter.coco2bb(self.anno_gt_path)
        elif self.format_gt == 'voc':
            gt_anno = converter.vocpascal2bb(self.anno_gt_path)
        elif self.format_gt == 'imagenet':
            gt_anno = converter.imagenet2bb(self.anno_gt_path)
        elif self.format_gt == 'labelme':
            gt_anno = converter.labelme2bb(self.anno_gt_path)
        elif self.format_gt == 'openimg':
            gt_anno = converter.openimage2bb(self.anno_gt_path, self.img_gt_path)
        elif self.format_gt == 'yolo':
            gt_anno = converter.yolo2bb(self.anno_gt_path, self.img_gt_path, self.classnames_path)
        elif self.format_gt == 'absolute':
            gt_anno = converter.text2bb(self.anno_gt_path, img_dir=self.img_gt_path)
        elif self.format_gt == 'cvat':
            gt_anno = converter.cvat2bb(self.anno_gt_path)
        elif self.format_gt == 'tube':
            logging.warning("Spatio-Temporal Tube AP specified. Loading ground truth and detection results at same time...")
            tube = TubeEvaluator(self.anno_gt_path, self.anno_det_path)
        else:
            raise Exception("%s is not a valid ground truth annotation format. Valid formats are: coco, voc, imagenet, labelme, openimg, yolo, absolute, cvat"%self.anno_gt_path)

        # collect detection truth labels:
        if self.format_det == 'coco':
            logging.warning("COCO detection format specified. Ignoring 'coord_det'")
            det_anno = converter.coco2bb(self.anno_det_path, bb_type=BBType.DETECTED)
        elif self.format_det == 'tube':
            pass
        else:
            if self.format_det == 'xywh':
                # x,y,width, height
                BB_FORMAT = BBFormat.XYWH
            elif self.format_det == 'xyrb':
                # x,y,right,bottom
                BB_FORMAT = BBFormat.XYX2Y2
            elif self.format_det == 'xcycwh':
                # x center y center
                BB_FORMAT = BBFormat.YOLO
            else:
                raise Exception("%s is not a valid detection annotation format"%self.format_det)

            if self.coor_det == 'abs':
                COORD_TYPE = CoordinatesType.ABSOLUTE
                if self.format_det == 'xcycwh':
                    logging.warning("format_det defined as YOLO (x_center, y_center, width, height) but coordinates defined as absolute!\
                        This will probably give incorrect results!")
            elif self.coor_det == 'rel':
                COORD_TYPE = CoordinatesType.RELATIVE
            else:
                raise Exception("%s is not a valid detection coordinate format"%self.coor_det)
            det_anno = converter.text2bb(self.anno_det_path, bb_type=BBType.DETECTED, bb_format=BB_FORMAT, type_coordinates=COORD_TYPE, img_dir=self.img_gt_path)

            # If the gt specified requires names, then switch id based to name:
            if self.classnames_path != '':
                det_anno = general_utils.replace_id_with_classes(det_anno, self.classnames_path)

        # print out results of annotations loaded:
        print("%d ground truth bounding boxes retrieved"%(len(gt_anno)))
        print("%d detection bounding boxes retrieved"%(len(det_anno)))


        # COCO (101-POINT INTERPOLATION)
        if self.metric == 'coco':
            logging.info("Running metric with COCO metric")
            print("Running metric with COCO metric")
            coco_sum = coco_evaluator.get_coco_summary(gt_anno, det_anno)
            coco_out = coco_evaluator.get_coco_metrics(gt_anno, det_anno, iou_threshold=self.threshold)
            value_only = tuple([float(_i[1]) for _i in coco_sum.items()])
            print(value_only)
            print( ('\nCOCO metric:\n'
                    'AP [.5:.05:.95]: %f\n'
                    'AP50: %f\n'
                    'AP75: %f\n'
                    'AP Small: %f\n'
                    'AP Medium: %f\n'
                    'AP Large: %f\n'
                    'AR1: %f\n'
                    'AR10: %f\n'
                    'AR100: %f\n'
                    'AR Small: %f\n'
                    'AR Medium: %f\n'
                    'AR Large: %f\n\n'%value_only) )

            print("Class APs:")
            for item in coco_out.items():
                if item[1]['AP'] != None:
                    print("%s AP50: %f"%(item[0], item[1]['AP']))
                else:
                    logging.warning('AP for %s is None'%(item[0]))

            if self.plot_flag:
                logging.warning("Graphing precision-recall is not supported!")

            return coco_sum, coco_out

        # 11-POINT INTERPOLATION
        elif self.metric == 'voc2007':
            print("Running metric with VOC2012 metric, using the 11-point interpolation approach")

            voc_sum = pascal_voc_evaluator.get_pascalvoc_metrics(gt_anno, det_anno, iou_threshold=self.threshold, method=MethodAveragePrecision.ELEVEN_POINT_INTERPOLATION)
            print("mAP: %f"%(voc_sum['mAP']))
            print("Class APs:")
            for class_item in voc_sum['per_class'].items():
                if class_item[1]['AP'] != None:
                    print("%s AP: %f"%(class_item[0], class_item[1]['AP']))
                else:
                    logging.warning('AP for %s is None'%(class_item[0]))

            if self.plot_flag:
                pascal_voc_evaluator.plot_precision_recall_curve(voc_sum['per_class'], mAP=voc_sum['mAP'], savePath=self.save_path, showGraphic=False)
            return voc_sum

        # EVERY POINT INTERPOLATION
        elif self.metric == 'voc2012':
            print("Running metric with VOC2012 metric, using the every point interpolation approach")

            voc_sum = pascal_voc_evaluator.get_pascalvoc_metrics(gt_anno, det_anno, iou_threshold=self.threshold)
            print("mAP: %f"%(voc_sum['mAP']))
            print("Class APs:")
            for class_item in voc_sum['per_class'].items():
                if class_item[1]['AP'] != None:
                    print("%s AP: %f"%(class_item[0], class_item[1]['AP']))
                else:
                    logging.warning('AP for %s is None'%(class_item[0]))

            if self.plot_flag:
                pascal_voc_evaluator.plot_precision_recall_curves(voc_sum['per_class'],  savePath=self.save_path, showGraphic=False)
                pascal_voc_evaluator.plot_precision_recall_curve(voc_sum['per_class'], mAP=voc_sum['mAP'], savePath=self.save_path, showGraphic=True)
            return voc_sum

        # SST METRIC
        elif self.metric == 'tube':
            tube_out = tube.evaluate()
            per_class, mAP = tube_out
            print("mAP: %f"%(mAP))
            print("Class APs:")
            for name, class_obj in per_class.items():
                print("%s AP: %f"%(name, class_obj['AP'])) 
            return tube_out
        else:
            # Error out for incorrect metric format
            raise Exception("%s is not a valid metric (coco, voc2007, voc2012)"%(self.format_gt))