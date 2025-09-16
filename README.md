# Specializing LLMs in Finance

Code base for my thesis on Specializing LLMs in Finance.





Code is organized by methodology section as much as possible. Directories with miscellaneous code were used for code that could not be efficiently organized through this scheme.



Execution order of directories is given by the numbered list (i.e., 1.1 Data collection, 1.2 ..., 2, 3, ..., 5). Within the directories, the scripts were executed in the following order:



1. 1\. EDGAR\_bulk\_scraper3.py, jb\_inquisitor.sh
   2. jb\_clean.sh, jb\_clean\_large.sh
   3. get\_whitelist.py, jb\_whitelist.sh
   4. jb\_shuf.sh
   5. jb\_tokenize.sh, jb\_merge.sh (Ideally only jb\_tokenize.sh is needed. jb\_merge.sh was used to salvage tokenized data that failed to merge due to OOM kill event.)
2. create\_venv.sh, jb\_train.sh, jb\_split.sh
3. convert\_tatqa.py, merge\_alpaca\_tatqa.py, jb\_instruct.sh
4. create\_venv\_PIXIU.sh, jb\_eval.sh
5. 
