class ParamSingletonMeta(type):
    """
    The Singleton class can be implemented in different ways in Python. Some
    possible methods include: base class, decorator, metaclass. We will use the
    metaclass because it is best suited for this purpose.
    """

    _instances = {}

    def __call__(cls, *args, **kwargs):
        """
        Possible changes to the value of the `__init__` argument do not affect
        the returned instance.
        """
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]


class ParamSingleton(metaclass=ParamSingletonMeta):

    def __init__(self): 
        
        # Docker Paths
        self.docker_dataset_path = '/root/model_evaluation/test_dataset'
        self.docker_single_test_dataset_paths = '/root/model_evaluation/new_test_datasets'
        self.docker_model_path = '/root/model_evaluation/yolov7_ycb_models'
        self.docker_predictions_path = '/root/model_evaluation/predictions'
        self.docker_evaluation_path = '/root/model_evaluation/evaluation'
        self.docker_train_dataset_path = '/root/datasets'
        self.docker_train_datasets = '/root/train_datasets'

        # Model name
        self.model_name = '12_04_10_40_ycbv_ycbm_yolov7pt_01-adam'       

        # Dataset test scenes
        self.dataset_scenes=['couch_table','table', 'shelf']
        self.single_test_dataset_name = ""
        # Save Predictions 
        self.save_predictions = True
        self.plot_images = False

        # prediction params
        self.conf_thres = 0.5
        self.iou_thres = 0.45

    def get_docker_dataset_path(self):
        return self.docker_dataset_path
    
    def get_docker_model_path(self):
        return self.docker_model_path
    
    def get_docker_predictions_path(self):
        return self.docker_predictions_path
    
    def get_docker_evaluation_path(self):
        return self.docker_evaluation_path
    
    def get_docker_train_dataset_path(self):
        return self.docker_train_dataset_path
    
    def get_docker_single_test_dataset_path(self):
        return self.docker_single_test_dataset_paths
    
    def get_docker_train_datasets(self):
        return self.docker_train_datasets

    def get_dataset_scenes(self):
        return self.dataset_scenes
    
    def get_model_name(self):
        return self.model_name
    
    def get_save_predictions(self):
        return self.save_predictions
    
    def get_plot_images(self):
        return self.plot_images
    
    def get_conf_thres(self):
        return self.conf_thres
    
    def get_iou_thres(self):
        return self.iou_thres

    def set_model_name(self,model_name):
        self.model_name = model_name
    
    def set_single_test_dataset_name(self, dataset_name):
        self.single_test_dataset_name = dataset_name
    
    def get_single_test_dataset_name(self):
        return self.single_test_dataset_name 