# How to train

use python3.9

## Download custom images
To download custom datas see this [video](https://www.youtube.com/watch?v=-QWxJ0j9EY8)

## Annotate images
Annotate the images in yolo format

For every image there should be an txt file with the annotations. 
Use for example labelImg to annotate images

Format:
```bash
class_id center_x center_y width height
```
*Attention: format is in percentage. Top left of the image is (0,0) and bottom right (1,1)*
If you have the top left and the bottom right corner of the bbox calculate like this:

```python
image_width = 640
image_height = 480
bbox[187.27,  46.85, 440.31, 165.08]

widht = abs(bbox[0] - bbox[2]) / image_width 
height  = abs(bbox[1] - bbox[3]) / image_height

x_center = (bbox[0] / image_widht) + widht / 2
y_center = (bbox[1] / image_height) + height / 2

```


## Prepare Images to train

Add the images and labels for training into *data/train/images* and *data/train/labels*
Add the images and labels for validation into *data/val/images* and data/val/labels'

*Hint: If the directories are not there run the build_training_template.sh script*


## Install the requirements 

**Note: Please check your cuda version and install the right torch and torchvision version into the requirements_gpu.txt**
For that 
1. Go to this [pytorch](https://pytorch.org/get-started/previous-versions/) website and select the right version.
2. Add these into the *requirements_gpu.txt* in the given format!
3. Run the requirements.txt and the requirements\_gpu.txt with

```bash
pip3 install -r requirements.txt
```
and 

```bash
pip3 install -r requirements_gpu.txt
```

## Change the configs

### custom_data.yaml
Change the *custom_data.yaml* in the data directory
1. Change the number of classes
2. Add the classes refered to the id. *First class_name == id 0 etc.*

### yolov7_custom.yaml
Change into the *yolov7_custom.yaml* into the *cfg/training* directory the number of classes

*Hint: This is possible with all yolov7 model types given into the cfg/taining directory*

## Download Pretrained Model

Depending on which model type you choose from the chapter above, download the pretrained weights from the 
[yolov7 repo](https://github.com/WongKinYiu/yolov7) and store them in the root of this repo


## Start Training
Run this command into the terminal of the root of this directory

```bash
python3 train.py --workers 1 --device 0 --batch-size 8 --epochs 100 --img 640 640 --data data/custom_data.yaml --hyp data/hyp.scratch.custom.yaml --cfg cfg/training/yolov7_custom.yaml --name yolov7_custom --weights yolov7.pt 
```
 
| Parameter      | Description | Hints | 
| ----------- | ----------- | ----------- | 
| workers      | maximum number of dataloader workers       | |
| device   | cuda device, i.e. 0 or 0,1,2,3 or cpu        | | 
| batch-size   | total batch size for all GPUs       | The more gpu storage you have the bigger the batch_size | 
| epochs   | how many epochs to train        | | 
| img-size   | [train, test] image sizes        | Depending on which model you train the image size change. E.g for YOLOv7-W6 the img-size = 1280. For more informations: [yolov7 repo](https://github.com/WongKinYiu/yolov7)| 
| data   | Path to the custom_data.yaml you created      | | 
| hyp   | hyperparameters path       |
| cfg   | Path to the yolov7_custom.yaml you created       | | 
| name   | Name of the yolov7 output model        | | 
| weights   | Path to pretrained weights        | | 

**Optional:**
| Parameter      | Description | Hints | 
| ----------- | ----------- | ----------- | 
| adam      | use Adam optimizer instead of SGD       | |



## After Trainig
After the Training the yolov7 model is stored into the directory *runs/traing/yolov7-custom/weights/best.pt*

## Start Prediction
To test the model run
```bash
python3 detect.py --weights best.pt --conf 0.5 --img-size 640 --source 1.jpg --view-img --no-trace 
```
