## Performance comparison of YOLOv7 and YOLOv8 using the YCB datasets YCB-M and YCB-Video

This repo is an implementation of the paper "Performance comparison of YOLOv7 and YOLOv8 using the YCB datasets YCB-M and YCB-Video" and can be found into the `doc/paper` directory.

The trained models can be found [here](https://drive.google.com/drive/folders/1_rSO1G0Ve8FNOtPmyeTobopHP1VXrbYy?usp=sharing)

## Getting Started
This application is running in docker. 
There are four docker images available:
1. yolov7_training
2. yolov7_validation
3. yolov8_training
4. yolov8_validation

### Build and Run an image

1. To run one of these images, first build the image with following command (yolov7_training as example): 
```bash
docker-compose -f docker/yolov7/v7_training_docker-compose.yml build
```
2. After that run the image with following command
```bash
docker-compose -f docker/yolov7/v7_training_docker-compose.yml up -d 
```
*Important Hint: Please mount some volumes if necessary, if for example you need a dataset in the yolov7 training image.* 

## Datasets

### YCB-Video and YCB-M dataset split
To split the two datasets like I did in the paper, follow these steps:
1. Download the YCB-Video and YCB-M Dataset
2. Build and run the docker image of the yolov7_validation as described above.
3. Add the two datasets as volume mount in the validation dataset compose.
4. Adjust the `self.docker_training_dataset_path` in which both dataset are stored in `src/validation/param_singleton.py` script.
3. In the directory `/root/src/validation` are two scripts called `ycbm_dataset.py` and `ycbv_dataset.py` these are used to split the datasets. 
4. Before starting you have to adjust the paths in the inits of these scripts, e.g. where the splits has to be stored.
5. Run the scripts.

Optionaly the splitted datasets can be found here: (Link rein, wenn ich dafür Speicher bekomme) 

### Own Created dataset
The own created test dataset can be found here: (Link rein, wenn ich dafür Speicher bekomme) 

## YOLOv7 training
1. Run the yolov7_training image
2. Mount the Training and Validation dataset if necessary
3. Exec into the image and run
```bash
python3 train.py --workers 48 --device 0 --batch-size 40 --epochs 100 --img 640 640 --data data/custom_data.yaml --hyp data/hyp.scratch.custom.yaml --cfg cfg/training/yolov7x.yaml --name new_model_name --weights yolov7x.pt --patience 10 --save_period 1
```
*Hint: adjust paths where the training and validation datas are in data/custom_data.yaml*

## YOLOv8 training
1. Run the yolov8_training image
2. Mount the Training and Validation dataset if necessary
3. Exec into the image and run   
```bash
yolo mode=train task=detect model=train model=yolov8x.pt data=config/custom.yaml epochs=100 imgsz=640 batch=40 workers=48 device=0 patience=10 save_period=1 plots=True name=new_model_name
```
*Hint: adjust paths where the training and validation datas are in config/custom.yaml*

## Validation
1. Run the yolov7 or yolov8 validation image (depends, which has to be evaluated)
2. Mount all datasets (YCB-M, YCB-Video and own created) into the docker image.
3. Adjust the paths in `param_singleton.py` and add the `model_name` and the correspoding test_dataset (YCB-M, YCB-Video or combination) into the `main.py`.   
4. Run the `main.py` script with following command 
```bash
python3 main.py
```
*Hint: If there are some problems, please write an issue.*

## Authors
Samuel Hafner (@hafners)

## Acknowledgment 
Thanks to the contributors of the repo [review_object_detection_metrics](https://github.com/rafaelpadilla/review_object_detection_metrics), who made the mAP calculation easy. 
