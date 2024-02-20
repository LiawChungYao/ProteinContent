""" RESEARCH TOPIC: The classification of different proteins into class labels based on its nutritional values.

    To run this file, type in the terminal 'python main.py'

"""

import re
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import wrangling
import supervisedlearning as sl
import graphing.py as graph

warnings.filterwarnings(action='ignore')

# The raw dataset file
filename = 'nutrient-file-release2-jan22.xlsx'

# Dataset directory
    #      15000s -> Fish
    #      17000s -> Eggs
    #      18000s -> Meat
    #      22204s -> Nuts
    #      25000s -> Legumes
    #      34000s -> Crocodile

# STAGE 1: WRANGLING
# 
# Wrangling the dataset into known classifications [Fish, Eggs, Meat, Nuts, Legumes] and unknown classification [Crocodile]
# The data is saved, so the raw data can be assessed later for graphing
wrangling.wrangle('nutrient-file-release2-jan22.xlsx',1,[15, 17, 18, 22204, 25],'wrangled.xlsx')
wrangling.wrangle('nutrient-file-release2-jan22.xlsx',1, [34], 'unclassified.xlsx')

#----------------------------------------------------------------------------------#
#   STAGE 2: MODELLING
#
# Predetermined constant values
random = 100 # Used as input for mutual information and cross-validation [If script is run again, it would produce the same result]
f2f_threshold = 0.8 # Feature to feature threshold [If 2 features have a high correlation, one will be discarded to reduce redundancy]

# Load the wrangled dataset
df = pd.read_excel('wrangled.xlsx')
df = df.reset_index(drop=True)

# Remove all columns with an average of Zero
# If a column has an average of 0, as the entire dataset is filled with positive values, assumption is that all values are 0
column_means = df.iloc[:, 3:].mean()
columns_to_remove = column_means[column_means == 0].index
for column in columns_to_remove:
    df.pop(column)

# Seperating dataset into class_label and features
class_label = df['Classification'] # This is the Classification
features = df.iloc[: , 3:] # Features (Stripping the first 3 columns)
features = features.fillna(0) 

# If 2 features have a higher correlation than f2f_threshold, one of them will be removed
features = sl.feature_reduction(features,f2f_threshold) 

# Correlation between each feature and the class_label
correlation_scores = sl.feature_correlation(features,class_label,random)

# Analysis of the best threshold
best_threshold = sl.selected_threshold(correlation_scores,features, class_label, random)

selected_features = sl.strippedfeatures((best_threshold[2]/100), correlation_scores)

# Import the unclassified file, and let the model predict which class_label it falls under
unclassified = pd.read_excel('unclassified.xlsx')
unclassified = unclassified.reset_index(drop=True)
unclassified = unclassified.iloc[: , 3:]
unclassified = unclassified.fillna(0)
unclassified = unclassified[selected_features]

print(sl.Test(selected_features, features, class_label, unclassified))
print(selected_features)

#---------------------------------------------------------------------------------------------
# Graphing
#
# Using the list of columns selected during Stage 2 to graph each nutrient content against each food class
column_list = selected_features

# Creates scatter plots
graph.graph_scatter(column_list, [0,1,2,3,4], ['Seafood','Eggs','Meat','Nuts','Legumes'], '/home/scatter_graphs/')

# Creates box plots
graph.graph_boxes(column_list, [0,1,2,3,4], ['Seafood','Eggs','Meat','Nuts','Legumes'], '/home/box_plots/')
