import os
import json
import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np
import sys
sys.path.append("/root/src/python/")
from evaulation_utils.plots import Plots
import param_singleton

class ModelComparison:

    def __init__(self): 
        self.plots_utils = Plots()
        self.param_singleton = param_singleton.ParamSingleton()
        print(id(self.param_singleton))

        self.docker_evaluation_path = self.param_singleton.get_docker_evaluation_path()

    def __build_table_plots(self,df, save_fig_path):
        col_colours,row_colours = ['#999999' for x in df.columns], ['#999999' for x in df.index] 
        max_per_columns = df.max(axis='rows')

        cell_colors = np.full(df.values.shape,'w',dtype='str')
        for cls in df.columns:
            cell_colors[np.where(df.values == [max_per_columns[cls]])] = 'g'

        fig, ax = plt.subplots()
        table = ax.table(cellText=df.values,cellColours=cell_colors,colLabels=df.columns, colColours=col_colours,rowLabels=df.index,rowColours=row_colours,loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(7,10)
        ax.axis('off')

        fig.savefig(save_fig_path, bbox_inches='tight', pad_inches=0)


    def compare_models(self): 
        print("--- Starting Comparing Models---")

        json_model_eval_path = os.path.join(self.docker_evaluation_path,"models_evaluation","models_evaluation.json")
        
        with open(json_model_eval_path) as f:
            json_model_eval = json.load(f)

        general_dic = {}
        per_object_dic = {}

        for model_name in json_model_eval['models'].keys():
            general_dic[model_name] = json_model_eval['models'][model_name]["general"]
            obj_names = json_model_eval['models'][model_name]["per_object"].keys()
            per_object_dic[model_name] = {}
            for obj_name in obj_names:
                if obj_name == 'sugar_box':
                    continue
                
                per_object_dic[model_name][obj_name] = format(json_model_eval['models'][model_name]["per_object"][obj_name]['AP'],'.5f')

        df_general = pd.DataFrame.from_dict(general_dic)
        df_per_obj = pd.DataFrame.from_dict(per_object_dic)

        df_general_T = df_general.T
        df_per_object_T = df_per_obj.T
        
        model_eval_path = os.path.join(self.docker_evaluation_path,"models_evaluation","models_evaluation.png")
        model_eval_obj_path = os.path.join(self.docker_evaluation_path,"models_evaluation","models_evaluation_per_object.png")
        
        self.__build_table_plots(df_general_T, model_eval_path)
        self.__build_table_plots(df_per_object_T, model_eval_obj_path)  

        print(f"--- The results are saved in the following path:{model_eval_path} ---")