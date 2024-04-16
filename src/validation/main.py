from metrics_calculation import MetricsCalcualtion
# from yolov8_prediction import Yolov8Prediction
from yolov9_prediction import Yolov9Prediction
from param_singleton import ParamSingleton
from model_comparison import ModelComparison
import os

def main():

    param_singleton = ParamSingleton()
    # param_singleton.set_model_name('32_19_02_100_40_ycbv_ycbm_yolov7x_01') 
    param_singleton.set_model_name('yolov9-ycb_v16') 
    param_singleton.set_single_test_dataset_name("combined_testdataset")
    
    yolov7_prediction = Yolov9Prediction()
    yolov7_prediction.predict()

    # yolov8_prediction = Yolov8Prediction()
    # yolov8_prediction.predict()

    anno_gt_path = os.path.join(param_singleton.get_docker_single_test_dataset_path(), param_singleton.get_single_test_dataset_name(), "labels")    
    img_gt_path = os.path.join(param_singleton.get_docker_single_test_dataset_path(), param_singleton.get_single_test_dataset_name(), "images")    
    anno_det_path = os.path.join(param_singleton.get_docker_predictions_path(),param_singleton.get_model_name(), param_singleton.get_single_test_dataset_name(),'labels')

    metrics_calculation = MetricsCalcualtion()
    metrics_calculation.calculate_metrics()
    # metrics_calculation.calculate_special_metrics(anno_gt_path= anno_gt_path, img_gt_path=img_gt_path, anno_det_path=anno_det_path)

    # model_comparion = ModelComparison()
    # model_comparion.compare_models()


if __name__ == "__main__":
    main()