# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 00:19:49 2025

@author: Julius de Clercq
"""


import os
import pickle 
import pandas as pd 
from pathlib import Path as pl

def get_data():
    log_path = os.path.join(pl.cwd().parent, "output", "processing_log_aggregated.pkl")
    with open(log_path, 'rb') as file:
        df = pickle.load(file)
    return df
    
df = get_data()
sums = df.sum() 

dir(sums) 

df.to_excel("processing_logs_yearly.xlsx")
sums.to_excel("processing_log_sums.xlsx")
