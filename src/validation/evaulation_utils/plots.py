import matplotlib.pyplot as plt 
import matplotlib
import numpy as np
from math import ceil
import param_singleton 

class Plots:
    def __init__(self): 
        self.param_singleton = param_singleton.ParamSingleton()
        self.__plot_images = self.param_singleton.get_plot_images()


    def plot_images(self,images, num_of_plots=12):
        matplotlib.use('TkAgg')
        
        if not self.__plot_images:
            print("Plot images is set to false")
            return
        
        if num_of_plots > len(images):
            print("Num of plots are larger then images")
            return 
        fig = plt.figure(figsize=(60,30),facecolor='black')
        random_idx = np.random.randint(1,len(images)-1,num_of_plots)
        columns = 4
        rows = ceil(num_of_plots/columns)
        for i, img_idx in enumerate(random_idx):
             fig.add_subplot(rows, columns, i+1)
             plt.imshow(images[img_idx])
             plt.axis('off')
        plt.show()
