import cv2
import numpy as np
import matplotlib.pyplot as plt 

class IOU:

    def __init__(self,images_path):
        self.images = []
        self.images_path= images_path


    def calc_mIOU(self,gt,val):
        
        ious = []

        for idx in range(len(gt['annotations'])):
            img_name = gt['images'][idx]['path'].split('/')[-1]

            bbox_val = np.array(val['annotations'][idx]['bbox']).astype(int)
            bbox_gt = np.array(gt['annotations'][idx]['bbox']).astype(int)

            # resize
            bbox_val[2] = bbox_val[0] + bbox_val[2]
            bbox_val[3] = bbox_val[1] + bbox_val[3]

            bbox_gt[2] = bbox_gt[0] + bbox_gt[2]
            bbox_gt[3] = bbox_gt[1] + bbox_gt[3]


            bbox_val = bbox_val + np.random.randint(50,100)

            iou = self.__bb_intersection_over_union(bbox_val, bbox_gt)
            ious.append(iou)

            image = cv2.imread(f'{self.images_path}/{img_name}')
            image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

            cv2.rectangle(image, tuple(bbox_gt[:2]), 
            	tuple(bbox_gt[2:]), (0, 255, 0), 10)
            cv2.rectangle(image, tuple(bbox_val[:2]), 
            	tuple(bbox_val[2:]), (0, 0, 255), 10)
            cv2.putText(image,str(iou)[0:5],(bbox_gt[0],bbox_gt[1]),fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=5, color=(0,255,0), thickness=5)

            image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

            cv2.imwrite(f"{self.images_path}/new_{img_name}", image)

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            self.images.append(image)

        print(f"mIOU: {np.sum(ious)/len(ious)} ")

    def plot_images(self):
        fig = plt.figure(figsize=(30, 30))
        columns = 4
        rows = 2
        for i in range(1,len(self.images)):
            fig.add_subplot(rows, columns, i)
            if i < len(self.images):
                plt.axis('off')
                plt.imshow(self.images[i])

        plt.show()


    def __bb_intersection_over_union(self,boxA, boxB):
        # determine the (x, y)-coordinates of the intersection rectangle
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        # compute the area of intersection rectangle
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        # compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = interArea / float(boxAArea + boxBArea - interArea)
        # return the intersection over union value
        return iou
