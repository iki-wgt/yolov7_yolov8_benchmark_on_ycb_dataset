from metrics_calculation import MetricsCalcualtion
# from yolov7_prediction import Yolov7Prediction
from yolov8_prediction import Yolov8Prediction
from param_singleton import ParamSingleton
from model_comparison import ModelComparison

def main():

    param_singleton = ParamSingleton()
    # param_singleton.set_model_name("01_07_04_300_16_ycbv_ycbm_yolov7pt_01") # v7 model
    # param_singleton.set_model_name("02_07_04_300_16_ycbv_ycbm_yolov7pt_02") # v7 model
    # param_singleton.set_model_name("03_07_04_10_55_ycbv_ycbm_yolov7pt_03") # v7 model
    # param_singleton.set_model_name("04_08_04_10_40_ycbv_ycbm_yolov7pt_01") # v7 model
    # param_singleton.set_model_name("05_11_04_10_40_ycbv_ycbm_yolov7-tinypt_01-tiny") # v7 model
    # param_singleton.set_model_name("06_11_04_50_40_ycbv_ycbm_08_04_10_40_ycbv_ycbm_yolov7pt_01_01") # v7 model
    # param_singleton.set_model_name("07_12_04_10_40_ycbv_ycbm_yolov7pt_01-adam") # v7 model
    # param_singleton.set_model_name("08_13_04_10_40_ycbv_ycbm_11_04_10_40_ycbv_ycbm_yolov7-tinypt_01-tiny_02-tiny") # v7 model
    # param_singleton.set_model_name("09_13_04_10_40_ycbv_ycbm_yolov7xpt_01_x_model") # v7 model
    # param_singleton.set_model_name("10_15_04_10_40_ycbv_ycbm_nopretrain_01") # v7 model
    # param_singleton.set_model_name("11_25_08_10_40_ycbv_ycbm_yolov8x_01_x_model") # v8 model 
    # param_singleton.set_model_name('12_04_09_10_40_ycbv_yolov7x_01_x_model') # v7 model
    # param_singleton.set_model_name('13_06_09_10_40_ycbv_yolov8x_01') # v8 model
    # param_singleton.set_model_name('14_07_09_10_40_ycbm_yolov8x_01') # v8 model
    # param_singleton.set_model_name('15_08_09_10_40_ycbm_yolov7x_01') # v7 model
    # param_singleton.set_model_name('16_11_09_20_40_ycbv_ycbm_yolov8x_01') # v8 model
    # param_singleton.set_model_name('17_12_09_5_40_ycbm_ycbv_yolov8x_01') # v8 model
    # param_singleton.set_model_name('18_13_09_10_20_ycbm_ycbv_yolov8x_01') # v8 model
    # param_singleton.set_model_name('19_13_09_10_40_ycbv_ycbm_yolov8m_01') # v8 model
    # param_singleton.set_model_name('20_13_09_10_40_ycbv_ycbm_yolov8n_01') # v8 model
    param_singleton.set_model_name('22_13_11_100_40_ycbm_ycbv_yolov8x_01') # v8 model 
    # param_singleton.set_model_name('23_17_11_100_40_ycbv_yolov8x_01') # v8 model 
    # param_singleton.set_model_name('24_21_11_100_40_ycbm_yolov8x_01') # v8 model 
    # param_singleton.set_model_name('25_09_12_100_40_ycbm_ycbv_yolov7x_01') # v7 model 

    # yolov7_prediction = Yolov7Prediction()
    # yolov7_prediction.predict()

    # yolov8_prediction = Yolov8Prediction()
    # yolov8_prediction.set_weight_name(f"best.pt")
    # yolov8_prediction.train_predict()
    
    metrics_calculation = MetricsCalcualtion()
    mAP = metrics_calculation.calculate_metrics(map_return=True)
    #mAPs.append(mAP)
    print(mAP)
    # mAPs = []
    # for i in range (0,100):
    # for i in range (1,100):
        # 
        # weight_name = f"epoch_{i:03}.pt"
        # yolov7_prediction = Yolov7Prediction()
        # yolov7_prediction.set_weight_name(weight_name)
        # yolov7_prediction.predict()
# 
        # yolov8_prediction = Yolov8Prediction()
        # yolov8_prediction.set_weight_name(f"epoch{i}.pt")
        # yolov8_prediction.predict()
# 
        # metrics_calculation = MetricsCalcualtion()
        # mAP = metrics_calculation.calculate_metrics(map_return=True)
        # mAPs.append(mAP)
# 
    # print(mAPs)
    # 
    # max_mAP = max(mAPs)
    # index_max = max(range(len(mAPs)), key=mAPs.__getitem__)    
    # print(f"The best epoch is {index_max+1}.pt with a MAP of {max_mAP}")
    # model_comparion = ModelComparison()
    # model_comparion.compare_models()


if __name__ == "__main__":
    main()