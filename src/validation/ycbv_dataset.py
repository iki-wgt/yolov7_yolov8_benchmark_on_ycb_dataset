import os 
import uuid
import shutil
import copy
from param_singleton import ParamSingleton


class YCBVDataset:

    def __init__(self): 
        self.param_singleton = ParamSingleton()
        self.ycbv_path = os.path.join(self.param_singleton.get_docker_train_dataset_path(), 'YCB_video_dataset/YCB_Dataset')
        self.ycb_images_path = os.path.join(self.ycbv_path, 'data')
        self.train_val_txt_path =  os.path.join(self.ycbv_path, 'yolo')
        self.train_path = os.path.join(self.param_singleton.get_docker_train_datasets(), 'ycbv_02_dataset', 'train')
        self.val_path = os.path.join(self.param_singleton.get_docker_train_datasets(), 'ycbv_02_dataset', 'val')
        self.test_path = os.path.join(self.param_singleton.get_docker_single_test_dataset_path(), 'ycbv_test_dataset')

        print(os.listdir(self.ycbv_path))
        print(os.listdir(self.train_val_txt_path))

    def __create_dir_structure(self):

        if os.path.exists(self.train_path) and os.path.exists(self.val_path) and os.path.exists(self.test_path):
            shutil.rmtree(self.train_path)
            shutil.rmtree(self.val_path)
            shutil.rmtree(self.test_path)
        
        os.makedirs(os.path.join(self.train_path,'images'), exist_ok=True)
        os.makedirs(os.path.join(self.train_path,'labels'), exist_ok=True)
        os.makedirs(os.path.join(self.val_path,'images'), exist_ok=True)
        os.makedirs(os.path.join(self.val_path,'labels'), exist_ok=True)
        os.makedirs(os.path.join(self.test_path,'images'), exist_ok=True)
        os.makedirs(os.path.join(self.test_path,'labels'), exist_ok=True)

    def __copy_train_val_files(self,paths):
        dir_img_ann_dic = {}

        for path in paths: 
            path = path.split('\n')[0]
            dir_name = path.split('/')[-2]
            image_name = path.split('/')[-1]
            ann_name = image_name.split('.')[0] + '.txt'
            
            if dir_name in dir_img_ann_dic:
                dir_img_ann_dic[dir_name].append([image_name, ann_name])
            else:
                dir_img_ann_dic[dir_name] = [[image_name,ann_name]]
        
        train_dir = {}
        val_dir = {}

        for key, values in dir_img_ann_dic.items():
            
            val_split = int(len(values) * 0.2)
            print(f"on dic {key} are {len(values)} images for training and therefore {val_split} images for val and {len(values) - val_split} for training")
            sorted_values = sorted(values, key=lambda x: x[0])
            split_ratio = int(len(values)/val_split)
            val_data = sorted_values[::split_ratio] # Get every ration data as val datas
            train_data = copy.deepcopy(sorted_values)
            del train_data[::split_ratio] # remove val datas

            # Check if val and train contain similar elements 
            check =  any(item in val_data for item in train_data)
            if check == True:
                print(f"ON {key} the list contains same element in train and val ")
                return
            
            train_dir[key] = train_data
            val_dir[key] = val_data

        for train_k , train_values in train_dir.items():
            for train_value  in train_values:

                new_uuid  = uuid.uuid4()
                new_png_name = f"{train_value[0].split('/')[-1].split('.')[0]}_{new_uuid}.png"
                new_ann_name = f"{train_value[1].split('/')[-1].split('.')[0]}_{new_uuid}.txt"

                shutil.copyfile(os.path.join(self.ycb_images_path,train_k,train_value[0]), os.path.join(self.train_path, 'images', new_png_name))
                shutil.copyfile(os.path.join(self.ycb_images_path,train_k,train_value[1]), os.path.join(self.train_path, 'labels', new_ann_name))

        for val_k , val_values in val_dir.items():
            for val_value  in val_values :

                new_uuid  = uuid.uuid4()
                new_png_name = f"{val_value[0].split('/')[-1].split('.')[0]}_{new_uuid}.png"
                new_ann_name = f"{val_value[1].split('/')[-1].split('.')[0]}_{new_uuid}.txt"

                shutil.copyfile(os.path.join(self.ycb_images_path,val_k,val_value[0]), os.path.join(self.val_path, 'images', new_png_name))
                shutil.copyfile(os.path.join(self.ycb_images_path,val_k,val_value[1]), os.path.join(self.val_path, 'labels', new_ann_name))

    def __copy_test_files(self, paths):

        print(f"# test datas {len(paths)}" )
        for path in paths: 
            path = path.split('\n')[0]
            dir_name = path.split('/')[-2]
            image_name = path.split('/')[-1]
            ann_name = image_name.split('.')[0] + '.txt'

            new_uuid  = uuid.uuid4()
            new_png_name = f"{image_name.split('/')[-1].split('.')[0]}_{new_uuid}.png"
            new_ann_name = f"{ann_name.split('/')[-1].split('.')[0]}_{new_uuid}.txt"

            shutil.copyfile(os.path.join(self.ycb_images_path,dir_name,image_name), os.path.join(self.test_path, 'images', new_png_name))
            shutil.copyfile(os.path.join(self.ycb_images_path,dir_name,ann_name), os.path.join(self.test_path, 'labels', new_ann_name))

    def start_train_val_split(self):
        self.__create_dir_structure()

        with open(os.path.join(self.train_val_txt_path, 'train.txt'), 'r') as f:
            train_paths = f.readlines()

        with open(os.path.join(self.train_val_txt_path, 'test.txt'), 'r') as f:
            test_paths = f.readlines()

        self.__copy_train_val_files(train_paths)
        self.__copy_test_files(test_paths)

if __name__ == "__main__":
    ycbv_dataset = YCBVDataset()
    ycbv_dataset.start_train_val_split()