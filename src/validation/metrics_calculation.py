import os 
import shutil
import sys
import json
from numpyencoder import NumpyEncoder
sys.path.append("/root/src/python/")
import param_singleton


class MetricsCalcualtion:
    def __init__(self): 
        self.param_singleton = param_singleton.ParamSingleton()
        print(id(self.param_singleton))
        # Get docker paths
        self.docker_evaluation_path = self.param_singleton.get_docker_evaluation_path()
        self.docker_predictions_path = self.param_singleton.get_docker_predictions_path() 
        self.docker_dataset_path = self.param_singleton.get_docker_dataset_path()

       
        # Get dataset scenes
        self.dataset_scenes = self.param_singleton.get_dataset_scenes()


        # Get Model Name
        self.model_name = self.param_singleton.get_model_name()


    def __prep_calculation(self): 
        """
        Prepare datas and create dir structure to calculate mAP
        """
        des_datas_path = os.path.join(self.docker_evaluation_path,'datas_to_predict')

        # create data path to calculate mAP
        if os.path.isdir(des_datas_path):
            shutil.rmtree(des_datas_path)

        os.makedirs(os.path.join(des_datas_path,'gt','images'))
        os.makedirs(os.path.join(des_datas_path,'gt','labels'))
        os.makedirs(os.path.join(des_datas_path,'pred'))

        shutil.copy(os.path.join(self.docker_predictions_path,self.model_name,'classes.txt'), des_datas_path)

        # Copy ground_truth and pred datas
        for dataset_scene in self.dataset_scenes:
            # gt_paths
            gt_images_path = os.path.join(self.docker_dataset_path,dataset_scene,'labeled_images','images')
            gt_labels_path = os.path.join(self.docker_dataset_path,dataset_scene,'labeled_images','labels')
            # pred_paths
            pr_labels_path = os.path.join(self.docker_predictions_path,self.model_name,dataset_scene,'labels')

            sub_scenes = sorted(os.listdir(os.path.join(self.docker_dataset_path,dataset_scene,'labeled_images','images')))
            for sub_scene in sub_scenes:
                gt_images = sorted(os.listdir(os.path.join(gt_images_path,sub_scene)))        
                gt_labels = sorted(os.listdir(os.path.join(gt_labels_path,sub_scene)))
                prefix = f"{sub_scene}"
                for label ,image in zip(gt_labels,gt_images):
                    shutil.copy(os.path.join(gt_images_path,sub_scene,image), os.path.join(des_datas_path,'gt','images',f"{prefix}_{image}"))
                    shutil.copy(os.path.join(gt_labels_path,sub_scene,label), os.path.join(des_datas_path,'gt','labels',f"{prefix}_{label}"))
                    shutil.copy(os.path.join(pr_labels_path,sub_scene,label), os.path.join(des_datas_path,'pred',f"{prefix}_{label}"))

    def calculate_metrics(self, map_return=None):
        print("--- Starting Metrics Calculations OWN DATASET ---")

        self.__prep_calculation()
        # 1. Calculate mAP
        from review_object_detection_metrics.map_calculation import MAP_Metric
        metric = MAP_Metric()

        metric.set_anno_gt_path(os.path.join(self.docker_evaluation_path,'datas_to_predict','gt','labels'))
        metric.set_anno_det_path(os.path.join(self.docker_evaluation_path,'datas_to_predict','pred'))
        metric.set_img_gt_path_(os.path.join(self.docker_evaluation_path,'datas_to_predict','gt','images'))

        metric.set_format_gt('yolo')
        metric.set_format_det('xcycwh')
        metric.set_coord_det('rel')
        metric.set_metric('coco')
        metric.set_classnames_path(os.path.join(self.docker_evaluation_path,'datas_to_predict','classes.txt'))
        # metric.set_classnames_path(os.path.join("/root/model_evaluation/predictions/22_13_11_100_40_ycbm_ycbv_yolov8x_01",'classes.txt'))
        metric.set_threshold(0.5)
        metric.set_plot_flag(False)
        metric.set_save_path(os.path.join(self.docker_evaluation_path,'CLI'))
        coco_metric_res, metrics_per_object = metric.calculate_mAP();
        metric.set_metric('voc2012')
        # voc_metric_res = metric.calculate_mAP();
        # print(voc_metric_res)
        print(f"--- MAP: {coco_metric_res['AP']} ")

        # Check if json file exist
        if not os.path.isfile(os.path.join(self.docker_evaluation_path,'models_evaluation','models_evaluation.json')):
            with open(os.path.join(self.docker_evaluation_path,'models_evaluation','models_evaluation.json'), "w") as f:
                f.write(json.dumps({'models':{}},cls=NumpyEncoder))

        if os.path.isfile(os.path.join(self.docker_evaluation_path,'models_evaluation','models_evaluation.json')):
            with open(os.path.join(self.docker_evaluation_path,'models_evaluation','models_evaluation.json'), "r") as f:
                 current_json_content = json.load(f)
            current_json_content['models'][self.model_name] = {"own_dataset":{'general':coco_metric_res,'per_object': metrics_per_object}}
            with open(os.path.join(self.docker_evaluation_path,'models_evaluation','models_evaluation.json'), "w") as f:
                f.write(json.dumps(current_json_content,cls=NumpyEncoder))
        else:
            print("Error: models_evaluation.json not found and cant create one")                    

        print(f"--- MAP: {coco_metric_res['AP']} ")

        if map_return:
            return coco_metric_res['AP']

    def calculate_special_metrics(self, anno_gt_path, img_gt_path, anno_det_path, map_return=None,):
        print("--- Starting Metrics Calculations SPECIAL DATASET ---")

        # self.__prep_calculation()
        # 1. Calculate mAP
        from review_object_detection_metrics.map_calculation import MAP_Metric
        metric = MAP_Metric()

        ## FOR SPECIAL DATASET PATH
        metric.set_anno_gt_path(anno_gt_path)
        metric.set_anno_det_path(anno_det_path)
        metric.set_img_gt_path_(img_gt_path)

        metric.set_format_gt('yolo')
        metric.set_format_det('xcycwh')
        metric.set_coord_det('rel')
        metric.set_metric('coco')
        metric.set_classnames_path(os.path.join(self.docker_evaluation_path,'datas_to_predict','classes.txt'))
        # metric.set_classnames_path(os.path.join("/root/model_evaluation/predictions/22_13_11_100_40_ycbm_ycbv_yolov8x_01",'classes.txt'))
        metric.set_threshold(0.5)
        metric.set_plot_flag(False)
        metric.set_save_path(os.path.join(self.docker_evaluation_path,'CLI'))
        coco_metric_res, metrics_per_object = metric.calculate_mAP();
        metric.set_metric('voc2012')
        # voc_metric_res = metric.calculate_mAP();
        # print(voc_metric_res)
        print(f"--- MAP: {coco_metric_res['AP']} ")

        # Check if json file exist
        if not os.path.isfile(os.path.join(self.docker_evaluation_path,'models_evaluation','models_evaluation.json')):
            with open(os.path.join(self.docker_evaluation_path,'models_evaluation','models_evaluation.json'), "w") as f:
                f.write(json.dumps({'models':{}},cls=NumpyEncoder))

        if os.path.isfile(os.path.join(self.docker_evaluation_path,'models_evaluation','models_evaluation.json')):
            with open(os.path.join(self.docker_evaluation_path,'models_evaluation','models_evaluation.json'), "r") as f:
                 current_json_content = json.load(f)
                 curr_dic = current_json_content['models'][self.model_name]
                 curr_dic['special_dataset'] = {'general':coco_metric_res,'per_object': metrics_per_object}
            # current_json_content['models'][self.model_name] = {"special_dataset": {'general':coco_metric_res,'per_object': metrics_per_object}}
            current_json_content['models'][self.model_name] = curr_dic
            with open(os.path.join(self.docker_evaluation_path,'models_evaluation','models_evaluation.json'), "w") as f:
                f.write(json.dumps(current_json_content,cls=NumpyEncoder))
        else:
            print("Error: models_evaluation.json not found and cant create one")                    

        print(f"--- MAP: {coco_metric_res['AP']} ")

        if map_return:
            return coco_metric_res['AP']
        ## TODO Plot Confusion Matrix
        # 1. Get all ids of the frame of gt and prediction
        # 2. if same id, check if IOU >0 else it is a backround FP else it is right
        # 3. if not same id, check if the ID has some IOU with other id 
        