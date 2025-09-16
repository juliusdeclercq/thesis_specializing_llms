# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 13:43:46 2025

@author: Julius de Clercq
"""



#%%         Imports
import os
import re
# import sys
# import logging
import requests
# from concurrent.futures import ProcessPoolExecutor
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
# from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support import expected_conditions as EC
from multiprocessing import Lock

from time import (time as t, sleep)
start = t()

os.chdir(os.path.dirname(os.path.abspath(__file__)))            # Setting working directory to the location of this script.

#%%             Driver initialization
###########################################################
### driver, wait = initialize_driver(wait_time = 5)
def initialize_driver(wait_time = 5, headless = True):
    """
    Purpose:
        Initialize the chrome driver.

    Inputs:
        chrome_driver_path                      path to chrome driver

    Return value:
        driver                                  instance of chrome driver
    """

    download_dir = os.path.join(os.getcwd(), "EDGAR_bulk")

    chrome_options = Options()
    prefs = {"download.default_directory": download_dir, "download.prompt_for_download": False}
    chrome_options.add_experimental_option("prefs", prefs)
    # chrome_options.add_argument("user-agent = Bulk_downloader (Vrije Universiteit Amsterdam); j.l.h.de.clercq@businessdatascience.com)")
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument("--disable-search-engine-choice-screen")

    if headless:
        chrome_options.add_argument('--headless')
    chrome_service = Service("C:/chromedriver/chromedriver.exe")
    driver = webdriver.Chrome(service=chrome_service, options=chrome_options)
    wait = WebDriverWait(driver, wait_time)
    # actions = ActionChains(driver)

    return driver, wait


#%%             Request with limit
###########################################################
### response, last_request_time = rate_limited_request(url, last_request_time, headers)

def rate_limited_request(url, headers):
    """
    Here we make requests to the webpage while making sure we do not make more requests
    than 10 per second. Otherwise the webpage will block the request.


    """
    global last_request_time
    with Lock():
        # Ensure 10 requests per second, so sleep if last request was within 0.1 seconds.
        # Taking a little bit margin here just to be safe, so 0.11 instead of 0.1.
        current_time = t()
        elapsed_time = current_time - last_request_time
        if elapsed_time < 0.11:
            sleep(0.11 - elapsed_time)
        last_request_time = t()
        response = requests.get(url, headers=headers)
    return response

#%%
def find_file_index_after_break(file_name, year, quarter, driver):
    """
    If the download was stopped before completion and needs to be resumed, this function
    returns a list with the index of the remaining files to be downloaded from this page.

    Input:
    file_name: str              File name (with suffix) of the last file which was successfully downloaded.
    year: int                   Year at which the download stopped.
    quarter: int                Quarter at which the download stopped.

    Output:
    file_list: list             List with the index of the remaining files on this page.

    """
    # year = 2022
    # quarter = 2
    # file_name = "20220615.nc.tar.gz"

    # driver, wait = initialize_driver(wait_time = 10, headless = False)
    # Go to the page of the respective year and quarter, and wait for it to load.
    # driver.get(f"https://www.sec.gov/Archives/edgar/Feed/{year}/QTR{quarter}/")
    # wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="main-content"]/table/tbody/tr[2]/td[1]/a')))
    file_list = []
    i = 2
    search_file_idx = None
    while True:
        try:
            current_file_name = driver.find_element(By.XPATH, f'//*[@id="main-content"]/table/tbody/tr[{i}]/td[1]/a').text
            if current_file_name == file_name:
                search_file_idx = i
            file_list.append(i)
            i += 1
        except Exception:
            # If we get an error, it means that we have found the end of the list of files to download.
            break
    # driver.quit()

    # Only include the index of the files that have not yet been downloaded.
    file_list = file_list[file_list.index(search_file_idx) + 1:]
    # file_list.index(search_file_idx)
    return file_list

#%%

def EDGAR_finished_download_lister(year):
    """
    We check what files are already downloaded, such that these are not downloaded
    again. This makes sure we can both double check the downloads for a year and
    resume downloading from the right position if the download was interrupted.

    """
    year_dir = os.path.join(os.getcwd(), "EDGAR_bulk", str(year))
    digit_list = []

    # Iterate over all files in the directory
    for filename in os.listdir(year_dir):
        # Use a regular expression (regex) to find the first eight digits in the filename
        match = re.match(r'^\d{8}', filename)
        if match:
            # Append the match to the list as a string
            digit_list.append(match.group(0))

    return digit_list

#%%
def EDGAR_bulk_downloader(years):

    """
    This function downloads the .tar.gz bulk files
    """

    driver, wait = initialize_driver(wait_time = 10, headless = False)
    headers = {'User-Agent': 'Bulk_downloader (Vrije Universiteit Amsterdam); j.l.h.de.clercq@businessdatascience.com)'}
    quarters = [i for i in range (1, 5)]
    download_results = {}

    for year in reversed(years):
        # We create a directory per year of filings, unless the directory already exists.
        year_dir = os.path.join(os.getcwd(), "EDGAR_bulk", str(year))
        try:
            os.mkdir(year_dir)
        except:
            FileExistsError

        # Find the list of files already downloaded.
        digit_list = EDGAR_finished_download_lister(year)

        for quarter in reversed(quarters):
            # Skip the missing quarters in 1995 and 1996, and 2024Q4 (which is the future), and 2024Q3 (which is finished)
            # if year == 1995 and (quarter == 1 or quarter == 2):
            #     continue
            # if year == 1996 and (quarter == 2 or quarter == 3):
            #     continue
            # if year == 2013 and (quarter == 2 or quarter == 3 or quarter == 4):

            # Go to the page of the respective year and quarter, and wait for it to load.
            driver.get(f"https://www.sec.gov/Archives/edgar/Feed/{year}/QTR{quarter}/")
            wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="main-content"]/table/tbody/tr[2]/td[1]/a')))

            # Find which files are to be downloaded from this page.
            download_names = {}
            i = 2
            while True:
                try:
                    download_element = f'//*[@id="main-content"]/table/tbody/tr[{i}]/td[1]/a'
                    download_name = driver.find_element(By.XPATH, download_element).text
                    if not download_name[:8] in digit_list:
                        download_names.update({i : download_name[:8]})
                    i += 1
                except Exception:
                    # If we get an error, it means that we have found the end of the list of files to download.
                    break

            # Download the files to be downloaded and log the results.
            download_results[year] = {"success" : {},
                                      "fail": {}}
            for file_idx, file_name in download_names.items():
                # If anything goes wrong with the download, we log this by returning the file_idx
                try:
                    # Find the element to download by its position in the table (i.e. file_idx) and download.
                    download_element = f'//*[@id="main-content"]/table/tbody/tr[{file_idx}]/td[1]/a'
                    download_name = driver.find_element(By.XPATH, download_element).text
                    download_url = driver.find_element(By.XPATH, download_element).get_attribute("href")
                    download_path = os.path.join(year_dir, download_name)

                    response = rate_limited_request(download_url, headers)

                    if 'html' not in response.headers.get('Content-Type', ''):
                        with open(download_path, 'wb') as file:
                            for chunk in response.iter_content(chunk_size=8192):
                                if chunk:
                                    file.write(chunk)
                    download_results[year]["success"].update({file_idx : file_name})
                    print(f"Success: {file_name}")
                except Exception:
                    download_results[year]["fail"].update({file_idx : file_name})
                    print(f"Fail: {file_name}")

        driver.quit()
        return download_results


#%%
def main():
    last_request_time = 0
    # years = [i for i in range(1998, 2025)]
    years = [2016]
    
    
    download_results = EDGAR_bulk_downloader(years)
    
    
    print(download_results)

if __name__ == "main":
    main()








