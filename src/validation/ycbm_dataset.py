import os 
import uuid
from glob import glob
import shutil
from param_singleton import ParamSingleton
import copy

class YCBMDataset:

    def __init__(self): 
        self.param_singleton = ParamSingleton()
        self.ycbm_path = os.path.join(self.param_singleton.get_docker_train_dataset_path(), 'YCB-M Dataset')
        
        self.cameras = next(os.walk(self.ycbm_path))[1]
        self.camera_paths = [os.path.join(self.ycbm_path, camera) for camera in self.cameras]

        self.train_path = os.path.join(self.param_singleton.get_docker_train_datasets(), 'ycbm_02_dataset', 'train')
        self.val_path = os.path.join(self.param_singleton.get_docker_train_datasets(), 'ycbm_02_dataset', 'val')
        self.test_path = os.path.join(self.param_singleton.get_docker_single_test_dataset_path(), 'ycbm_test_dataset')


    def __create_dir_structure(self):

        if os.path.exists(self.train_path) and os.path.exists(self.val_path) and os.path.exists(self.test_path):
            shutil.rmtree(self.train_path, ignore_errors=True)
            shutil.rmtree(self.val_path, ignore_errors=True)
            shutil.rmtree(self.test_path, ignore_errors=True)

        os.makedirs(os.path.join(self.train_path,'images'), exist_ok=True)
        os.makedirs(os.path.join(self.train_path,'labels'), exist_ok=True)
        os.makedirs(os.path.join(self.val_path,'images'), exist_ok=True)
        os.makedirs(os.path.join(self.val_path,'labels'), exist_ok=True)
        os.makedirs(os.path.join(self.test_path,'images'), exist_ok=True)
        os.makedirs(os.path.join(self.test_path,'labels'), exist_ok=True)

    

    def start_train_val_split(self):

        self.__create_dir_structure()

        camera_ann_img_dir = {}
        for camera_name, camera_path in zip(self.cameras,self.camera_paths):
            print(camera_name, camera_path)
            
            self.image_dirs = os.path.join(camera_path, f'{camera_name}_dataset', camera_name)
            self.image_dirs = sorted(glob(self.image_dirs+'/*/trajectory/*.jpg'))
            
            self.ann_dirs = os.path.join(camera_path, f'{camera_name}_annotations', camera_name)
            self.ann_dirs = sorted(glob(self.ann_dirs+'/*/trajectory/*.txt'))

            camera_ann_img_dir[camera_name] = [self.image_dirs, self.ann_dirs]

        num_datas = 0
        for cam_key, cam_value in camera_ann_img_dir.items():
            num_datas += len(cam_value[0])
        
        test_split = 0.1
        val_split = 0.1
        
        print(f"NUM_DATA{num_datas}")

        # Split train and test
        for cam_key, cam_value in camera_ann_img_dir.items():
            tmp_test_split = int(len(cam_value[0]) * test_split)

            print(f"on dic {cam_key} are {len(cam_value[0])} images for training and therefore {tmp_test_split} images for test and {len(cam_value[0]) - tmp_test_split} for training")
            
            cam_imgs = cam_value[0]
            cam_anns = cam_value[1]
            split_ratio = int(len(cam_imgs)/tmp_test_split)
            
            test_imgs = cam_imgs[::split_ratio]            
            test_anns = cam_anns[::split_ratio]            

            print(len(test_imgs))            
            print(len(test_anns))

            train_val_imgs = copy.deepcopy(cam_imgs)
            train_val_anns = copy.deepcopy(cam_anns)

            del train_val_imgs[::split_ratio]
            del train_val_anns[::split_ratio]

            print(len(train_val_imgs))
            print(len(train_val_anns))

            check_img =  any(item in train_val_imgs for item in test_imgs)
            check_ann =  any(item in train_val_anns for item in test_anns)

            if check_img or check_ann:
                print(f"ON {cam_key} the list contains same element in train and test ")
                return

            tmp_ann_split = int(((len(train_val_imgs)) * val_split))
            split_ratio = int(len(train_val_imgs)/tmp_ann_split)

            print(f"on dic {cam_key} are {len(train_val_imgs)} images for training and therefore {tmp_ann_split} images for val and {len(train_val_imgs) - tmp_ann_split} for training")

            val_imgs = train_val_imgs[::split_ratio]
            val_anns = train_val_anns[::split_ratio]

            train_imgs = copy.deepcopy(train_val_imgs)
            train_anns = copy.deepcopy(train_val_anns)

            del train_imgs[::split_ratio]
            del train_anns[::split_ratio]

            check_img =  any(item in train_imgs for item in val_imgs)
            check_ann =  any(item in train_anns for item in val_anns)

            if check_img or check_ann:
                print(f"ON {cam_key} the list contains same element in train and val ")
                return

            for i in range(len(train_imgs)):
                image_dir = train_imgs[i]
                ann_dir = train_anns[i]

                new_uuid  = uuid.uuid4()
                new_png_name = f"{cam_key}_{new_uuid}.jpg"
                new_ann_name = f"{cam_key}_{new_uuid}.txt"

                shutil.copyfile(image_dir, os.path.join(self.train_path, 'images', new_png_name))
                shutil.copyfile(ann_dir, os.path.join(self.train_path, 'labels', new_ann_name))
   

            for i in range(len(val_imgs)):
                image_dir = val_imgs[i]
                ann_dir = val_anns[i]

                new_uuid  = uuid.uuid4()
                new_png_name = f"{cam_key}_{new_uuid}.jpg"
                new_ann_name = f"{cam_key}_{new_uuid}.txt"

                shutil.copyfile(image_dir, os.path.join(self.val_path, 'images', new_png_name))
                shutil.copyfile(ann_dir, os.path.join(self.val_path, 'labels', new_ann_name))

            for i in range(len(test_imgs)):
                image_dir = test_imgs[i]
                ann_dir = test_anns[i]

                new_uuid  = uuid.uuid4()
                new_png_name = f"{cam_key}_{new_uuid}.jpg"
                new_ann_name = f"{cam_key}_{new_uuid}.txt"

                shutil.copyfile(image_dir, os.path.join(self.test_path, 'images', new_png_name))
                shutil.copyfile(ann_dir, os.path.join(self.test_path, 'labels', new_ann_name))
   
if __name__ == "__main__":
    ycbv_dataset = YCBMDataset()
    ycbv_dataset.start_train_val_split()