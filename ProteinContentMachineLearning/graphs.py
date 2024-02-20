"""
    This file will produce graphs that exhibit the amount of nutrients in each classification. 
"""


# Features that should be graphed out
# ['C18:1 (g)', 'Total monounsaturated fatty acids, equated \n(g)', 'Sodium (Na) \n(mg)', 'Available carbohydrate, without sugar alcohols \n(g)', 'Available carbohydrate, with sugar alcohols \n(g)', 'Selenium (Se) \n(ug)', 'Total folates \n(ug)', 'Dietary folate equivalents \n(ug)', 'Vitamin E \n(mg)', 'Tryptophan \n(mg/gN)', 'Total long chain omega 3 fatty acids, equated \n(%T)', 'C18 (g)', 'Total trans fatty acids, imputed \n(mg)', 'C16:1 (g)', 'C22:5w3 (mg)', 'C22:6w3 (mg)', 'Total long chain omega 3 fatty acids, equated \n(mg)']

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_excel('wrangled.xlsx')

def graph():

# read in file 
    

#summary df 
    summ_df = df['Classification', 'Protein (g)'].groupby('Classification')
    print(summ_df.head())
