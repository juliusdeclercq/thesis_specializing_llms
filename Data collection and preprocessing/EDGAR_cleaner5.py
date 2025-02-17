# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 12:40:45 2025

@author: Julius de Clercq
"""

import os
import sys
import uu
import re
import json
import pickle 
import random
import tarfile
import argparse 
import pdfplumber
from io import BytesIO
from time import time as t 
from bs4 import BeautifulSoup
from multiprocessing import Lock
from concurrent.futures import ProcessPoolExecutor

lock = Lock()

#%%             Argument parser
def parse_arguments():
    """
    This is to handle the input_dir and output_dir arguments that should be passed
    to the script when executing it from the command line or a bash file. This
    is necessary for execution from Snellius.
        
    """
    # Create argument parser
    parser = argparse.ArgumentParser(description="Process input file and save results to output directory.")

    # Add arguments
    parser.add_argument('input_year',  nargs='?', type=str, help="Input directory year.")
    parser.add_argument('scratch_dir', nargs='?', type=str, help="Path to the output directory")
    parser.add_argument('scratch_output_dir', nargs='?', type=str, help="Path to the output directory")

    # Parse arguments
    try:
        args = parser.parse_args()
        input_year = args.input_year
        input_year = int(input_year) if not input_year == "large" else input_year   # passing 'large' as input_year for the directory with day files larger than 3GB.
        scratch_dir = args.scratch_dir
        scratch_output_dir = args.scratch_output_dir
    except Exception:
        print("\n\nYear and output directory arguments not passed to script.\n\n")
        input_year = scratch_dir = scratch_output_dir = ""
        # print("\nThe script is flexible enough that this is no issue.\n")

    return input_year, scratch_dir, scratch_output_dir

#%%
class EDGAR_cleaner:
    """
    The purpose of this class is to clean the raw EDGAR filings. An instance of this
    class is created once per year of data. The processing of this data is performed
    per day of data, and follows the following sequence:
        1. Decompress the .tar.gz file that contains all filings of a particular day.
            The unpacked data is saved to an intermediate directory: "YEAR_unpacked".

        The following steps are performed in parallel, using the ProcessPoolExecutor
        from concurrent.futures, with a separate process per filing:

            file_killer()
            2. Delete uninformative files. These files may only contain the UU-encoded
                PDF-file of a filing or be a placeholder for a paper filing. Such
                files are spotted by blacklisted lines in the file.

            encoded_block_remover()
            3. Remove any UU-encoded imgages or PDF files contained in a submission.


            HTML_parser()
            4. Remove HTML formatting by parsing the HTML input using BeautifulSoup.
                This should be done with minimal loss of information.

    """
    def __init__(self, year, scratch_dir, scratch_output_dir, verbose = True, test_mode = True):
        self.verbose = verbose
        self.cwd = os.getcwd()
        self.year = year
        
        if scratch_dir == "local":   # If this is the case, I am testing on my local machine.
            self.output_dir = os.path.join(os.getcwd(), "test", str(year))
            self.log_dir = os.path.join(self.output_dir, "logs") 
            self.input_dir = os.path.join(self.cwd, "EDGAR_bulk", str(year))
        else:   # Otherwise, we are working on Snellius, which has a different file organization system.
            self.output_dir = os.path.join(scratch_output_dir,  str(year)) 
            self.log_dir = os.path.join(self.output_dir, "logs") 
            self.input_dir = os.path.join(scratch_dir,  "data",   str(year))
        self.filings_dir = os.path.join(self.output_dir, "filings")
        os.makedirs(self.filings_dir, exist_ok = True)
        os.makedirs(self.output_dir,  exist_ok = True)
        os.makedirs(self.log_dir, exist_ok = True)
        os.makedirs(os.path.join(self.log_dir, "logs"), exist_ok = True)
        os.makedirs(os.path.join(self.log_dir, "filing_info"), exist_ok = True)
        
        # Initialize processing log        
        self.processing_log = {}
        self.processing_log["files_processed"] = 0
        self.processing_log["documents_processed"] = 0
        self.processing_log["char_encoding_formats"] = {}
        self.processing_log["HTML"] = {"characters_removed": 0,
                                       "blocks_parsed": 0}
        self.processing_log["UU_encodings"] = {"characters_removed": 0,
                                               "documents_removed": 0,
                                               "pdfs_extracted": 0,
                                               "pdfs_failed": 0,
                                               "characters_extracted": 0}
        self.processing_log["Kills"] = {"characters_removed": 0,
                                        "files_killed": 0}
        
        # Compile regex objects.
        self.regex_patterns = {"document": re.compile(r"""<DOCUMENT>.*?</DOCUMENT>""", flags=re.DOTALL),
                               "text_extract": re.compile(r'<TYPE>TEXT-EXTRACT'),
                               "type": re.compile(r"<TYPE>(.*)\s*"),
                               "description": re.compile(r"<DESCRIPTION>(.*)\s*"),
                               "HTML": [re.compile(r"""<html>?(.*?)</html>""", flags=re.DOTALL), 
                                        re.compile(r"""<HTML>?(.*?)</HTML>""", flags=re.DOTALL)],
                               "XML": [re.compile(r"""<xml>?(.*?)</xml>""", flags=re.DOTALL), 
                                       re.compile(r"""<XML>?(.*?)</XML>""", flags=re.DOTALL)],
                               "UU_encoding": re.compile(r"begin\s\d{3}.*end", flags=re.DOTALL),
                               "encoded_PDF": re.compile(r"<PDF>\s*begin\s\d{3}.*end\s*</PDF>", flags=re.DOTALL),
                               "info_patterns": {"firm": re.compile(r"<CONFORMED-NAME>(.*)\s*"),
                                                "date": re.compile(r"<TIMESTAMP>(.*)\s*"),
                                                "CIK": re.compile(r"<CIK>(.*)\s*"),
                                                "form_type": re.compile(r"<FORM-TYPE>(.*)\s*"),
                                                "SIC": re.compile(r"<ASSIGNED-SIC>(.*)\s*")
                                                },
                               "other_date_patterns" : [re.compile(r"<FILING-DATE>(.*)\s*"),
                                                        re.compile(r"<DATE-OF-FILING-DATE-CHANGE>(.*)\s*")],
                               "accession_id_end": re.compile(r"<ACCESSION-NUMBER>(.*)\s*"),
                               "faulty_escape_chars": re.compile(r'(?<!\\)\\(?![ntUu])')
                               }
    
    def __getstate__(self):
        """
        Explicit definitions for state retrieval and updating are necessary for 
        class pickleability, which is necessary for multiprocessing.
        """
        return self.__dict__
    def __setstate__(self, state):
        self.__dict__.update(state)
        
    ###########################################
    def _extract_file_info(self, file_content, file_name):
        """
        Here some file information is gathered. This info should be:
            file name
            filing company
            form type
            
        """
        file_info = {"new_file_name" : None,    # Placeholder s.t. this is the first item.
                     "file_name" : file_name,
                     "file_length": None,       # Placeholder s.t. this is the third item.
                     "raw_file_length": len(file_content)
                     }
        for key, pattern in self.regex_patterns["info_patterns"].items():
            match = pattern.search(file_content)
            file_info[key] = match.group(1) if match else None
        
        # Making sure we get the date right. There is some variation in how dates are reported in the metadata.
        # By default we check by <TIMESTAMP>, otherwise by <DATE-OF-FILING-DATE-CHANGE>, and if that does 
        # not work we check <FILING-DATE>.
        if not file_info["date"] == None:
            file_info["date"] = file_info["date"][:8] # Taking only the date (YYYYMMDD) out of the timestamp.
        else:
            for date_pattern in self.regex_patterns["other_date_patterns"]:
                match = date_pattern.search(file_content)
                if match:
                    file_info["date"] = match.group(1)[:8]
                    break
        
        # Apparently some files exist that have no accession ID. Giving such files some random suffix for uniqueness.
        accession_id = self.regex_patterns["accession_id_end"].search(file_content)
        accession_id_end = accession_id.group(1)[-6:] if accession_id else "MISSING_ACCESSION_ID_" + ''.join(str(random.randint(0,9)) for _ in range(6))
            
        new_file_name = f"{file_info['date']}_{file_info['form_type']}_{file_info['CIK']}_{accession_id_end}.txt".replace("/", "-")
        file_info["new_file_name"] = new_file_name
        return new_file_name, file_info
    
    ###########################################
    def _file_killer(self, file_content, file_name):
        """
        Deleting uninformative files, found by spotting blacklisted lines.
            "This document was generated as part of a paper submission." : no digital content
            
        Now only removing the paper submission placeholder files. The XML-tables 
        found in form types 3, 4 and 5 are considered too informative to remove.

        """

        blacklisted_lines = [
                             # "<FORM-TYPE>UPLOAD",
                             # "<FORM-TYPE>3",
                             # "<FORM-TYPE>4",
                             # "<FORM-TYPE>5",
                             "This document was generated as part of a paper submission."
                             ]

        if any(nono in file_content for nono in blacklisted_lines):
            # if self.verbose:
            #     print(f"\nDeleted uninformative file: {file_name}\n")
            return True
        else:
            return False
    
    ###########################################
    def _extract_pdf_text(self, uuencoded_pdf):
        """
        Decode UU-encoded content and extract text from the PDF. Touch up the text 
        extract to match the formatting of the given text extracts. 
        """
        # Step 1: Decode UU-encoded content
        uuencoded_bytes = BytesIO(uuencoded_pdf.encode('utf-8'))
        decoded_bytes = BytesIO()
        with open(os.devnull, 'w') as fnull: # Suppressing "trailing garbage" warnings from some crappy dependency of the uu module.
            stderr_original = sys.stderr
            try:
                sys.stderr = fnull
                uu.decode(uuencoded_bytes, decoded_bytes)
            finally:
                sys.stderr = stderr_original
                
        # Step 2: Extract text from PDF using pdfplumber.
        text = ""
        decoded_bytes.seek(0) # Stream reset.
        with pdfplumber.open(decoded_bytes) as pdf:
            for page in pdf.pages:
                text += page.extract_text() + "\n\n"  # Collect text from each page
        
        
        # Step 3: Mimic document structure.
        extract = "<DOCUMENT>\n<TYPE>TEXT-EXTRACT\n<SEQUENCE>1\n<FILENAME>text_extract.txt\n<TEXT>\n" + text.strip() + "\n</TEXT>\n</DOCUMENT>"
        char_counts = {"content": len(text.strip()),
                       "doc_metadata": len(extract) - len(text.strip())}
        return extract, char_counts
    
    ###########################################
    def _document_processor(self, file_content, processing_log):
        """
        Processes the documents contained in a filing by looping over the documents.
        XML sections are not parsed due to informative tags which would otherwise be
        removed. UU-encodings are removed and HTML sections are parsed. Information 
        per document is saved and later appended to the file_info dictionary per filing.
        
        The documents are processed in a manner that minimizes (working) memory usage
        by avoiding intermediate objects when possible, using pre-compiled regex objects
        instead of compiling these in each iteration, and -importantly- compiling the 
        processed content in chunks to keep the output size to its minimum.
        
        Pseudocode:
        1. text-extract present?
        2. loop over documents 
            2.1: get document info
                doc type
                description (if present)
                raw size (in chars)
                UU-encoding present?
                HTML present?
                XML present? 
            2.2 if UU encoding present:
                    if doc type is not pdf or PDF:
                        delete document
    		    else:
                    if text-extract present:
                        delete document
        			else:
        				extract text from UU-encoded PDF
        				substitute text-extract for document
            2.3 if HTML present:
                    extract clean text from HTML
                    substitute clean text for HTML block

        """
        # 0. Anterior definitions
        error_log = []
        kill_file = False
        docs = []   # list to save the info on the documents of the filing.
        filing_counts = {"chars_filing_metadata": 0,
                         "chars_doc_metadata": 0,
                         "chars_content": 0}
        # 1. Find if text-extract is present
        text_extract_present = bool(self.regex_patterns["text_extract"].search(file_content, re.DOTALL))
        
        # Initialize output with the filing's metadata, which is all text until the first document tag.
        output = [file_content[:file_content.find("\n<DOCUMENT>")]]
        filing_counts["chars_filing_metadata"] = len(output[0]) + 13    # 13 = len("</SUBMISSION>"), with which the filing is closed.
        
        # 2. Loop over documents
        for document in self.regex_patterns["document"].finditer(file_content):
            document = document.group(0) # Get the string from the match object.
            doc_output = [document[:document.find("\n<TEXT>")]]
            
            # 2.1. Get document info
            doc_info = {"removed": False,
                        "type": self.regex_patterns["type"].search(document).group(1).strip(),
                        "description": (desc.group(1).strip() if (desc := self.regex_patterns["description"].search(document)) else None),
                        "size_raw": len(document),
                        }
            doc_info["UU_encoding"] = doc_info["encoded_PDF"] = doc_info["HTML"] = doc_info["XML"] = False
            UU_encoding = self.regex_patterns["UU_encoding"].search(document)
            if UU_encoding:
                doc_info["UU_encoding"] = True
                encoded_PDF = self.regex_patterns["encoded_PDF"].search(document)
                if encoded_PDF:
                    doc_info["encoded_PDF"] = True
            for HTML_pattern in self.regex_patterns["HTML"]:
                HTML = HTML_pattern.search(document)
                if HTML:
                    doc_info["HTML"] = True
                    break
            for XML_pattern in self.regex_patterns["XML"]:
                XML = XML_pattern.search(document)
                if XML:
                    doc_info["XML"] = True
                    break
            
            # 2.2. Process UU-encodings
            if doc_info["UU_encoding"]:
                if not doc_info["encoded_PDF"]: # UU-encodings that are not PDF files are removed.
                    doc_info["removed"] = True
                    processing_log["UU_encodings"]["characters_removed"] += doc_info["size_raw"]
                    processing_log["UU_encodings"]["documents_removed"] += 1
                else:   # If the encoding does encode a PDF document...
                    if text_extract_present:    # ...and there already is a text-extract, just remove the encoding.
                        doc_info["removed"] = True
                        processing_log["UU_encodings"]["characters_removed"] += doc_info["size_raw"]
                        processing_log["UU_encodings"]["documents_removed"] += 1
                    else:       # ...but there is no text-extract, extract the text from the UU-encoded PDF to the filing.
                        try: # If this goes wrong, the PDF encoding may be corrupted, so we must discard the file. 
                            text_extract, char_counts = self._extract_pdf_text(UU_encoding.group(0))
                            doc_output.append(text_extract)
                            filing_counts["chars_doc_metadata"] += len(doc_output[0]) + char_counts["doc_metadata"]
                            filing_counts["chars_content"] += char_counts["content"]
                            processing_log["UU_encodings"]["pdfs_extracted"] += 1
                            processing_log["UU_encodings"]["characters_extracted"] += char_counts["content"]
                        except Exception as e:  # This triggers if reading the PDF fails, in which case the filing is removed.
                            error_log.append({"error_type": "PDF", "doc_info": doc_info, "error": e})
                            doc_info["removed"] = kill_file = True
                            processing_log["UU_encodings"]["pdfs_failed"] += 1
            
            # 2.3. Parse HTML blocks
            elif doc_info["HTML"]:    # Here the HTML content is parsed and the plain text extracted.
                soup = BeautifulSoup(HTML.group(0), 'html.parser')
                cleaned_text = soup.get_text().replace("\\", "/")
                doc_output.append("\n".join(["<TEXT>", cleaned_text, "</TEXT>\n</DOCUMENT>"]))
                processing_log["HTML"]["blocks_parsed"] += 1
                processing_log["HTML"]["characters_removed"] += len(HTML.group(0)) - len(cleaned_text)
                filing_counts["chars_content"] += len(cleaned_text)
                filing_counts["chars_doc_metadata"] += len(doc_output[0]) + 25   # This is the number of characters of "<TEXT>" + "</TEXT>\n</DOCUMENT>"
                
            else: # If we have neither UU-encodings nor HTML, we simply keep the document as is.
                output.append(document)
                doc_info["size_cleaned"] = doc_info["size_raw"]
                filing_counts["chars_doc_metadata"] += len(doc_output[0]) + 25   # 25 = len("<TEXT>" + "</TEXT>\n</DOCUMENT>")
                filing_counts["chars_content"] += len(document) - (len(doc_output[0]) + 25)
            # 2.4. Save the document info
            # This only triggers if we did any processing. If not, this step was performed in the lines directly above. 
            if doc_info["HTML"] or doc_info["UU_encoding"]: 
                if doc_info["removed"]:
                    doc_info["size_cleaned"] = 0
                else:
                    doc_info["size_cleaned"] = sum(len(chunk) for chunk in doc_output)
                    output.append("\n".join(doc_output))
                
            docs.append(doc_info)
            processing_log["documents_processed"] += 1
            
        # If the output only contains the documents metadata, it means that no
        # document remains after processing, so we have removed all content. 
        # Therefore we should kill the file. 
        if len(output) == 1:
            kill_file = True 
        
        if kill_file: # Returning an empty string if the file should be killed, as this may save memory.
            return "", docs, filing_counts, kill_file, processing_log, error_log
        
        output.append("</SUBMISSION>")
        processed_content = "\n".join(output)
        return processed_content, docs, filing_counts, kill_file, processing_log, error_log
    
    ###########################################
    def _process_day(self, day_file, test_filing_limit = None):
        """
        Processing the files in a folder through partial unpacking. The files
        are appended to an uncompressed .tar folder per day. After parallel
        processing, these .tar folders are merged into one.
        
        Also need to specify separate logs for the day, because the logs for the 
        full year cannot be accessed concurrently by the parallel workers. Except
        if I implement a lock to avoid race conditions, but I think that is slower 
        than just merging the daily logs after parallel processing. Hence, the latter
        is implemented.
        """
        i = 0
        
        error_log = []
        # Initialize daily processing log (cannot reference instance variables such as self.processing_log in parallel)
        processing_log = {}
        processing_log["files_processed"] = 0
        processing_log["documents_processed"] = 0
        processing_log["char_encoding_formats"] = {}
        processing_log["HTML"] = {"characters_removed": 0,
                                       "blocks_parsed": 0}
        processing_log["UU_encodings"] = {"characters_removed": 0,
                                               "documents_removed": 0,
                                               "pdfs_extracted": 0,
                                               "pdfs_failed": 0,
                                               "characters_extracted": 0}
        processing_log["Kills"] = {"characters_removed": 0,
                                        "files_killed": 0}
        
        day_tar_path = os.path.join(self.input_dir, day_file)
        output_day_tar = os.path.join(self.filings_dir, f"{day_file[:8]}.tar")
        file_info_path = os.path.join(self.log_dir, "filing_info", f"filing_info_{day_file[:8]}.jsonl")
        with open(file_info_path, 'w') as jsonl_file:
            pass  # Initialize the file.
        # Create and initialize the (intermediate) output tar file. Keep this open to reduce I/O overhead.
        with tarfile.open(output_day_tar, 'w') as tar, tarfile.open(day_tar_path, 'r:gz') as archive, open(file_info_path, 'a') as jsonl_file:
            ############################################################################################################
            # Step 1: Sequential partial decompression of the files in the day's .tar.gz (archive) folder.
                # Iterate over each file in the archive
            for member in archive.getmembers():
                if not member.isfile(): # Check if the member is a regular file (not a directory)
                    continue # If it's not a file, we skip it. 
                
                # Open the file inside the archive
                with archive.extractfile(member) as file:
                    # Process the file (e.g., read contents)
                    raw_data = file.read()
                
                file_content = raw_data.decode('utf-8', 'ignore')
                file_name = member.name.lstrip("./") 
                ############################################################################################################
                # Step 2: Process the decompressed files
                # 2.1: Skip the file if it should be killed (thereby effectively killing it).
                if self._file_killer(file_content, file_name):
                    processing_log["Kills"]["characters_removed"] += len(file_content)
                    processing_log["Kills"]["files_killed"]       += 1
                    continue 
                
                # 2.2: Extract file information.
                new_file_name, file_info = self._extract_file_info(file_content, file_name)
                # 2.3: Process documents
                processed_content, docs, filing_counts, kill_file, processing_log, filing_error_log = self._document_processor(file_content, processing_log)
                file_info["removed"] = kill_file
                file_info.update(filing_counts) 
                if filing_error_log:
                    error_log.append({"filing": new_file_name, "old_name": file_info["file_name"], "removed": kill_file, "log": filing_error_log})
                if kill_file:   # This is for if there is some malfunction in the document processor, as it likely indicates a corrupted file. 
                    processing_log["Kills"]["characters_removed"] += len(file_content)
                    processing_log["Kills"]["files_killed"]       += 1
                    continue
                
                file_info["documents"] = docs
                file_info["file_length"] = len(processed_content)
                processing_log["files_processed"] += 1
                
                ############################################################################################################
                # Step 3: Append file to uncompressed .tar file, and append file_info to .jsonl file.
                # File must be written to intermediate folder.
                
                json.dump(file_info, jsonl_file)  # Serialize the file_info dictionary
                jsonl_file.write('\n')  # Add a newline to separate JSON objects     
                
                content_bytes = processed_content.encode('utf-8', 'ignore')
                # Create a TarInfo object for the new file, specifying the file name and size.
                tar_info = tarfile.TarInfo(name = new_file_name)
                tar_info.size = len(content_bytes)
                # Add the file to the tar archive.
                tar.addfile(tar_info, BytesIO(content_bytes))
                
                # Stop the loop at the filing limit, if there is one. 
                if test_filing_limit:
                    i += 1 
                    if i == test_filing_limit:
                        break 
                    
        ############################################################################################################
        # Step 4: Save info lists and logs to output immediately. Aggregating on yearly level may likely lead to OOM errors. 
        
        logs = {"processing_log": processing_log,
                "error_log": error_log
                    }
        with lock:
            try:
                with open(os.path.join(self.log_dir, "logs", f"logs_{day_file[:8]}.pkl"), 'wb') as file:
                    pickle.dump(logs, file)
            except Exception as e:
                print(f"\n\nSaving log logs_{day_file[:8]}.pkl ran into the following error: \n{e}\n\n")
        
        return logs
    
    ###########################################
    def multiprocess_year_processor(self, active_cores, test_limits):
        """
        Processes files in a year's data directory in parallel. Decided to parallelize on
        the yearly level, not on the daily level, because decompression is time consuming and
        needs to be parallelized. 
        
        Parallelized using multiprocessing with ProcessPoolExecutor. Multithreading does not
        really work due to Python's Global Interpreter Lock (GIL). However, multiprocessing
        is more difficult as it requires the full class to be pickleable. This has been solved.
        
        One of the tests, with 15 days to process and a limit of 10 filings for each, gave the 
        following results for runtimes:
            sequential:         58 seconds (non-parallel benchmark)
            multithreading:     50 seconds
            multiprocessing:    13 seconds
        
        So multiprocessing is clearly necessary for effective parallelization. 
        """
        strt = t()
        
        if test_limits["day_limit"]:
            file_list = os.listdir(self.input_dir)
            file_list.sort()
            file_list =  file_list[:test_limits["day_limit"]]
        else:
            file_list = os.listdir(self.input_dir)
            file_list.sort()
        
        log_list = []
        #  Parallel processing of the days in the year.
        with ProcessPoolExecutor(max_workers = active_cores)  as executor: 
            futures = [executor.submit(self._process_day, day_file, test_filing_limit = test_limits["filing_limit"]) for day_file in file_list] 
        
        for future in futures:
            log_list.append(future.result())
        
        with open(os.path.join(self.log_dir, f"logs_{self.year}.pkl"), 'wb') as file:
            pickle.dump(log_list, file)
            
        runtime = t() - strt
        if self.verbose:
            n_files = "all" if not test_limits['filing_limit'] else test_limits['filing_limit']
            print(f"Processing {n_files} files for each of {len(futures)} days of {self.year} with multiprocessing took: {round(runtime)} seconds!")
        
################################################################################################################
#%%
def main():
    year, scratch_dir, scratch_output_dir = parse_arguments()
    # If I'm not passing the year and output_dir arguments to the script, I am testing on my local machine.
    if year == "":
        year = 2015
        scratch_dir = "local"
    
    # This can be used to hard code the number of cores to use on Snellius, in case OOM errors persist.
    # But by setting this to os.cpu_count() I simply use all cores at my disposal. 
    # Locally, I want to keep one core vacant for other processes.
    active_cores = 135 if not scratch_dir == "local" else os.cpu_count() - 1 
    
    # If local, use test limits. If not, no limits as we are on Snellius and not testing. 
    test_limits = {"day_limit":     15 if scratch_dir == "local" else None,       # 30, 30
                   "filing_limit":  20 if scratch_dir == "local" else None}
    
    cleaner = EDGAR_cleaner(year, scratch_dir, scratch_output_dir, verbose = True)
    cleaner.multiprocess_year_processor(active_cores, test_limits)
    
    return cleaner
        
#%%
if __name__ == "__main__": 
    cleaner = main()

