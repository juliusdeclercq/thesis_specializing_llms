# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 00:30:15 2025

@author: Julius de Clercq
"""

import os
import json
import pickle
import shutil
import pandas as pd 
import pathlib as pl
from concurrent.futures import ProcessPoolExecutor


#%%
def load_filing_info(year):
    # target_dir = os.path.join(pl.Path(__file__).resolve().parent.parent, "testy", "intermediate", str(year), "logs", "filing_info")     #  Change to Snellius directory
    target_dir = os.path.join("/projects/prjs1109/intermediate", str(year), "logs", "filing_info")    
    info_paths = os.listdir(target_dir)

    type_data = {}
    type_info = {}
    for info_path in info_paths:
        with open(os.path.join(target_dir, info_path), "r", encoding="utf-8") as f:
            
            for line in f:
                if not line.strip():
                    continue  

                # Load JSON and clean data
                data = json.loads(line.strip())
                form_type = data.pop("form_type")
                for key in ["documents", "date", "removed", "file_name"]:
                    data.pop(key, None)

                # Store in list instead of concatenating DataFrames
                if form_type not in type_data:
                    type_data[form_type] = []
                type_data[form_type].append(data)
                
                # Store sums of the integer variables.
                if form_type not in type_info:
                    type_info[form_type] = {key: 0 for key in data.keys() if key not in ["new_file_name", "firm", "CIK", "SIC"]}
                    type_info[form_type]["num_filings"] = 0
                # for key in type_info[form_type].keys():
                #     type_info[form_type][key] += data.get(key, 0)
                # type_info[form_type]["num_filings"] += 1
                
                # Write directly to disk in append mode with csv's (avoiding memory explosion)
                # This cannot be done for pickle format, which is the faster format.
                # for form_type, records in type_data.items():
                #     df = pd.DataFrame(records)
                #     output_path = os.path.join(os.getcwd(), "agg_filing_info", f"agg_filing_info_{year}_{form_type}.csv")
                #     df.to_csv(output_path, index=False, mode='a', header=not os.path.exists(output_path))
    
    # Converting lists to datframes. 
    for form_type, records in type_data.items():
        type_data[form_type] = pd.DataFrame(records)
        
    return (type_data, type_info)

#%%
def parallel_worker(year):
    type_data, type_info = load_filing_info(year)
    with open(os.path.join(os.getcwd(), "agg_filing_info", f"agg_filing_info_{year}.pkl"), 'wb') as file:
        pickle.dump(type_data, file)
    # with open(os.path.join(os.getcwd(), "agg_filing_info", f"agg_filing_info_sums_{year}.pkl"), 'wb') as file:
    #     pickle.dump(type_info, file)
    
    
        
def parallel_aggregation(years):
    with ProcessPoolExecutor(max_workers = 13)  as executor: 
        executor.map(parallel_worker, years)
    
#%%
def main():

    years = [i for i in range(2012, 2025)]
    
    # Make remove the output directory if it exists to avoid duplicate data; then recreate.
    output_dir = os.path.join(os.getcwd(), "agg_filing_info")
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)  # Delete if it exists
    os.makedirs(output_dir)  # Now create a fresh empty directory
    
    parallel_aggregation(years)

#%%
if __name__ == "__main__": 
    main()



