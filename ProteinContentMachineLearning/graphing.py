""" RESEARCH TOPIC: The classification of different proteins into class labels based on its nutritional values.

    Part 3: Graphing

    

"""

import re
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

column_list = ['C18:1 (g)', 'Total monounsaturated fatty acids, equated \n(g)', 'Sodium (Na) \n(mg)', 'Available carbohydrate, without sugar alcohols \n(g)', 'Available carbohydrate, with sugar alcohols \n(g)', 'Selenium (Se) \n(ug)', 'Total folates \n(ug)', 'Dietary folate equivalents \n(ug)', 'Vitamin E \n(mg)', 'Tryptophan \n(mg/gN)', 'Total long chain omega 3 fatty acids, equated \n(%T)', 'C18 (g)', 'Total trans fatty acids, imputed \n(mg)', 'C16:1 (g)', 'C22:5w3 (mg)', 'C22:6w3 (mg)', 'Total long chain omega 3 fatty acids, equated \n(mg)']
df = pd.read_excel('wrangled.xlsx')
df.fillna(0)

def graph_scatter(column_names: list, classification_codes: list, class_names: list, save_to_path: str) -> None:
    #   1. Create a scatter plot, setting the x-values as the 'Classification' values,
    #      and y-values as the respective column value

    
    new_name = [re.sub(r'\n.*', '', name) for name in column_names]
    column_num = range(len(column_names))
    for num in column_num:

        fig, ax = plt.subplots(figsize=(10,10))
        
        ax.scatter(x=df['Classification'], y=df[column_names[num]])
        #   Add custom labels for each classification_codes
        plt.xticks(classification_codes, class_names)
        #   Define X and Y axes labels
        plt.xlabel("Classification")
        plt.ylabel(column_names[num])
        plt.title(f'{column_names[num]} for each Classification')

        plt.show()
        #   Save figure once completed
        plt.savefig(save_to_path + new_name[num] + '.png')
        plt.close()


def graph_boxes(column_names: list, classification_codes: list, class_names: list, save_to_path: str) -> None:
    #   1. Creates subplots of boxplots for each nutrient in the column_names against the food classes

    # Creates a list of filenames to save for later
    new_name = [re.sub(r'\n.*', '', name) for name in column_names]


   
    
    column_num = range(len(column_names))
    for num in column_num:
        fig, ax = plt.subplots(figsize=(10,10))
        #   Identify data for the x and y axes
        sns.boxplot(data=df, x='Classification', y=column_names[num], showfliers=False)
        #   Create labels for the x-axis
        plt.xticks(classification_codes, class_names)
        plt.title(f'{column_names[num]} for each Classification')
        plt.show()
        #   Save figure to location
        plt.savefig(save_to_path + new_name[num] + '.png')
        plt.close()


