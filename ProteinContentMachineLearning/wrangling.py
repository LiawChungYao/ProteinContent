""" RESEARCH TOPIC: The classification of different proteins into class labels based on its nutritional values.

    Part 1: Data Wrangling

    To run this file, type in the terminal 'python wrangling.py'

"""

# 1) We are not filtering the selected columns because then there will be no use for feature Selection
# 2) Before filling in the Nan values to 0, see whether if the other values are relatively significant (As in if almost the whole column is 0
# then we probably wont use that compile
# 3) We need to make the data wrangling section a function that either returns the dataframe or saves a excel file     
# 4) Can you strip the classification column to only contain the first 2 values. this way I can have a more streamlined answer sheet

# 5) Can you save crocodile into another File

import pandas as pd
import re

def wrangle(filename: str, sheet_number: int, row_filter: list, new_filename: str) -> None:
    #   1. Convert excel into a pd dataframe
    df = pd.read_excel(filename, sheet_name=sheet_number)

    #   2. Filter and remove rows unrelated to the foods we want to classify.
    #      15000s -> Fish
    #      17000s -> Eggs
    #      18000s -> Meat
    #      22204s -> Nuts
    #      25000s -> Legumes
    #      34000s -> Crocodile
    row_filters = r''
    for number in range(len(row_filter)):
        if len(str(row_filter[number])) == 5:
            pattern = r'(^{first},*{second})'.format(first=str(row_filter[number])[:2], second=str(row_filter[number])[2:])   
            row_filters = "".join([row_filters, pattern])
            
            
        else:
            pattern = r'(^{code},*\d{{3}})'.format(code=row_filter[number])
            row_filters = "".join([row_filters, pattern])
            
        if row_filter[-1] != row_filter[number]:
            row_filters = row_filters + "|"
        
        
    # print(row_filters)
   
    mask = df['Classification'].astype(str).str.contains(row_filters, regex=True)
    df = df[mask]

    
    #   3. Replace classification codes with new classification codes
    df['Classification'] = df['Classification'].apply(lambda x: str(x)[:2])
    for n in range(len(row_filter)):
        df.loc[df['Classification'] == str(row_filter[n])[:2], ['Classification']] = n 

    df.to_excel(new_filename, index=False)

