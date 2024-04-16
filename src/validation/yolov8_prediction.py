from PIL import Image
from ultralytics import YOLO
import os 
import shutil
import sys
import glob 
import math

sys.path.append("/root/src/python/")
import param_singleton

class Yolov8Prediction:

    def __init__(self):
        self.param_singleton = param_singleton.ParamSingleton() 
        
        self.model_name = self.param_singleton.get_model_name()
        self.model_path = self.param_singleton.get_docker_model_path()
        self.model_path = os.path.join(self.model_path,self.model_name,"weights", "best.pt")
        print("--- Loading Yolov8 model---")
        self.model = YOLO(self.model_path)  # load model

        self.docker_dataset_path = self.param_singleton.get_docker_dataset_path()
        self.docker_single_test_dataset_paths = self.param_singleton.get_docker_single_test_dataset_path()
        self.dataset_scenes = self.param_singleton.get_dataset_scenes()
        self.predictions_path = self.param_singleton.get_docker_predictions_path()

        self.pred_model_path = os.path.join(self.predictions_path,self.model_name)
    
    def set_weight_name(self, weight_name):
        self.model_path = os.path.join(self.param_singleton.get_docker_model_path(),self.model_name,"weights", weight_name)
        self.model = YOLO(self.model_path)  # load model
        print(self.model_path)

    def __start_prediction(self):
        print("--- Starting Predictions---")
        speeds = []
        for dataset_scene in self.dataset_scenes:
            conf_thres = self.param_singleton.get_conf_thres()
            iou_thres = self.param_singleton.get_iou_thres()

            dataset_scene_path = os.path.join(self.docker_dataset_path,dataset_scene,'labeled_images','images')
            obj_scenes = sorted(os.listdir(dataset_scene_path))

            for obj_scene in obj_scenes:
                images_path = os.path.join(self.pred_model_path,dataset_scene,'images')
                labels_path = os.path.join(self.pred_model_path,dataset_scene,'labels',obj_scene)
                # if not os.path.isdir(labels_path):
                os.makedirs(labels_path, exist_ok=True)
                # images = os.listdir(os.path.join(dataset_scene_path,obj_scene))
                images = glob.glob(os.path.join(dataset_scene_path,obj_scene,'*'))
                results = self.model.predict(images, 
                                            project =images_path, 
                                            name= obj_scene, 
                                            save=True, 
                                            conf = conf_thres, 
                                            iou = iou_thres)

                for idx , r in enumerate(results):
                    boxes = r.boxes
                    names = r.names   
                    boxes.cls.tolist()
                    boxes.conf.tolist()
                    boxes.xywhn.tolist()
                    name = images[idx].split('/')[-1].split('.')[0] + '.txt'
                    path = os.path.join(labels_path, name)
                    with open(path,'w') as f :
                        for idx, cls in enumerate(boxes.cls.tolist()):
                            conf = boxes.conf.tolist()[idx]
                            x,y,w,h = boxes.xywhn.tolist()[idx]
                            f.write(f"{int(cls)} {conf} {x} {y} {w} {h}\n")

            speeds.append(results[0].speed['inference'])

        print("Average speed: ", sum(speeds)/len(speeds))    
        
        return names, sum(speeds)/len(speeds)
    
    def __start_specific_dataset_predition(self):
        
        print("--- Starting Specific Predictions---")
        speeds = []
        conf_thres = self.param_singleton.get_conf_thres()
        iou_thres = self.param_singleton.get_iou_thres()

        dataset_name = self.param_singleton.get_single_test_dataset_name()

        dataset_scene_path = os.path.join(self.docker_single_test_dataset_paths, dataset_name, 'images')

        labels_path = os.path.join(self.pred_model_path, dataset_name,'labels')
        # if not os.path.isdir(labels_path):
        os.makedirs(labels_path, exist_ok=True)
        # images = os.listdir(os.path.join(dataset_scene_path,obj_scene))
        images = glob.glob(os.path.join(dataset_scene_path, "*"))

        print(f"Predicting on {len(images)} images")
        img_per_run = 30
        iterations = math.ceil((len(images) / img_per_run))
        print(f"NUM OF ITERATIONS {iterations}")
        for i in range(iterations):
            sub_img = images[i*img_per_run:(i+1)*img_per_run]
            results = self.model.predict(sub_img, 
                                        project = labels_path, 
                                        name= dataset_name, 
                                        save=False, 
                                        conf = conf_thres, 
                                        iou = iou_thres)

            for idx , r in enumerate(results):
                boxes = r.boxes
                names = r.names   
                boxes.cls.tolist()
                boxes.conf.tolist()
                boxes.xywhn.tolist()
                name = sub_img[idx].split('/')[-1].split('.')[0] + '.txt'
                path = os.path.join(labels_path, name)
                with open(path,'w') as f :
                    for idx, cls in enumerate(boxes.cls.tolist()):
                        conf = boxes.conf.tolist()[idx]
                        x,y,w,h = boxes.xywhn.tolist()[idx]
                        f.write(f"{int(cls)} {conf} {x} {y} {w} {h}\n")

        speeds.append(results[0].speed['inference'])

        print("Average speed: ", sum(speeds)/len(speeds))    
        
        return names, sum(speeds)/len(speeds)

    def __start_train_prediction(self):
        
        print("--- Starting Predictions---")
        speeds = []
        conf_thres = self.param_singleton.get_conf_thres()
        iou_thres = self.param_singleton.get_iou_thres()

        # dataset_scene_path = os.path.join(self.docker_dataset_path,dataset_scene,'labeled_images','images')
        # obj_scenes = sorted(os.listdir(dataset_scene_path))

        images_path = os.path.join(self.pred_model_path,'images')
        labels_path = os.path.join(self.pred_model_path,'labels')
        # if not os.path.isdir(labels_path):
        os.makedirs(labels_path, exist_ok=True)
#       
        # images = os.listdir(os.path.join(dataset_scene_path,obj_scene))
        images = glob.glob(os.path.join('/root/train_datasets/ycbv_ycbm_01_dataset/train/images/*'))
        print(f"Predicting on {len(images)} images")
        img_per_run = 50
        iterations = math.ceil((len(images) / img_per_run))
        print(f"NUM OF ITERATIONS {iterations}")
        for i in range(iterations):
            sub_img = images[i*img_per_run:(i+1)*img_per_run]
            print(f"RANGE {i*img_per_run} - {(i+1) *img_per_run}")
            results = self.model.predict(sub_img, 
                                        project = images_path, 
                                        name= "prediction", 
                                        save=False, 
                                        conf = conf_thres,
                                        half=True,  
                                        iou = iou_thres)
#           
            print(f"Result length {len(results)}")
            for idx , r in enumerate(results):
                boxes = r.boxes
                names = r.names   
                boxes.cls.tolist()
                boxes.conf.tolist()
                boxes.xywhn.tolist()
                name = sub_img[idx].split('/')[-1].split('.')[0] + '.txt'
                path = os.path.join(labels_path, name)
                with open(path,'w') as f :
                    for idx, cls in enumerate(boxes.cls.tolist()):
                        conf = boxes.conf.tolist()[idx]
                        x,y,w,h = boxes.xywhn.tolist()[idx]
                        f.write(f"{int(cls)} {conf} {x} {y} {w} {h}\n")

                speeds.append(results[0].speed['inference'])

        print("Average speed: ", sum(speeds)/len(speeds))    
        
        return names, sum(speeds)/len(speeds)

    def __create_classes_txt(self, names):
        with open(os.path.join(self.pred_model_path,"classes.txt"),'w') as f :
            for name in names.values():
                f.write(f'{name}\n')
    
    def __pre_processing(self):

        if os.path.isdir(self.pred_model_path):
            shutil.rmtree(self.pred_model_path)
        
        # Create model dir to save predictions
        os.makedirs(self.pred_model_path, exist_ok=True)
    
    def __write_pred_time(self,pred_time):
        with open(os.path.join(self.pred_model_path,"pred_time.txt"),'w') as f :
            f.write(f'Prediction_Time: {pred_time} ms\n')

    def predict(self):
        self.__pre_processing()
        names, pred_time = self.__start_prediction()
        names, pred_time = self.__start_specific_dataset_predition()
        self.__create_classes_txt(names)
        self.__write_pred_time(pred_time)

    # def train_predict(self):
    #     self.__pre_processing()
    #     names, pred_time = self.__start_train_prediction()
    #     self.__create_classes_txt(names)
    #     self.__write_pred_time(pred_time)

if __name__ == "__main__":
    yolov8_prediction = Yolov8Prediction()
    yolov8_prediction.predict()
