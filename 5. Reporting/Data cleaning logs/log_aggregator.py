# -*- coding: utf-8 -*-
"""
Created on Sun Jan 19 19:31:54 2025

@author: Julius de Clercq
"""

import os
import pickle 
import pandas as pd 

#%%

def load_logs(year):
    
    log_dir = os.path.join("/projects/prjs1109/intermediate", str(year), "logs", "logs")
    logs = []
    days = []
    for log_path in os.listdir(log_dir):
        days.append(log_path[5:13])
        with open(os.path.join(log_dir, log_path), 'rb') as file:
            logs.append(pickle.load(file))

    return (logs, days)


#%%


def aggregate_day_logs(year):
    logs, days = load_logs(year)
    # Initialize processing_stats dictionary
    processing_stats = {}
    for key in logs[0]["processing_log"].keys():
        if isinstance(logs[0]["processing_log"][key], dict):
            processing_stats.update({key: {sub_key: [] for sub_key in logs[0]["processing_log"][key].keys()}})
        else: 
            processing_stats.update({key: []})
    
    # Aggregate daily logs into a dictionary of lists "processing_stats"
    for log in logs:
        for key, value in log["processing_log"].items():
            if not isinstance(value, dict):  # Aggregating simple values (e.g., 'files_processed', 'documents_processed')
                processing_stats[key].append(value) 
            else:   # Aggregating values in nested dictionaries (e.g., 'HTML', 'UU_encodings')
                for sub_key, sub_value in value.items():
                    processing_stats[key][sub_key].append(sub_value)
    
    # Now make a pandas dataframe out of it, denesting any nested values
    df_processing_stats = pd.DataFrame(index = days) 
    for key, value in processing_stats.items():
        if not isinstance(value, dict):  # Aggregating simple values (e.g., 'files_processed', 'documents_processed')
            df_processing_stats[key] = value
        else:
            for sub_key, sub_value in value.items():
                df_processing_stats[f"{key}_{sub_key}"] = sub_value
    
    year_log = {}
    for column in df_processing_stats.columns :
        year_log.update({column: df_processing_stats[column].sum()})
    
    return year_log


#%%
def main():
    years = [i for i in range(2012, 2025)]
    df_year_stats = pd.DataFrame()
    df_year_stats.empty 
    for year in years:
        year_log = aggregate_day_logs(year)
        df_year_stats = pd.concat([df_year_stats, pd.DataFrame(year_log, index = [year])])
        
    # Save output    
    with open("processing_log_aggregated.pkl", 'wb') as file:
        pickle.dump(df_year_stats, file)


#%%
if __name__ == "__main__": 
    main()


