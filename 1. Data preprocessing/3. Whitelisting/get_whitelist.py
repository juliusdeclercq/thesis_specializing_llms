# -*- coding: utf-8 -*-
"""
Created on Fri Mar 28 13:37:30 2025

@author: Julius de Clercq
"""

import pandas as pd

df = pd.read_excel("form type metrics.xlsx")
# df = df[['Form type', 'Whitelisted']]

whitelist = df[['Form type', 'file_length', 'raw_file_length', 'total_filings']].loc[df['Whitelisted'] == 1] 
summary = whitelist[[column for column in whitelist.columns if column != 'Form type']].sum(axis = 0)   

token_estimate = {"LB": summary['file_length'] / 4 , "UB": summary['file_length'] / 3}
print(f"The cleaned subsetted dataset comprises an estimated {round(token_estimate['LB']/1e12, 2)} to {round(token_estimate['UB']/1e12, 2)} trillion tokens.")

whitelisted_form_types = whitelist['Form type'] 
whitelisted_form_types.to_csv()

pd.to_pickle(whitelisted_form_types, 'whitelist.pkl') 
