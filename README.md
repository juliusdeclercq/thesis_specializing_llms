# Specializing LLMs in Finance

Code base for my thesis on Specializing LLMs in Finance.





The code is organized by methodology section as much as possible. Directories with miscellaneous code were used for code that could not be efficiently organized through this scheme. Execution order of directories is given by the numbered list (i.e., 1.1 Data collection, 1.2 ..., 2, 3, ..., 5). Within the directories, the scripts were executed in the following order:



1. 1\. EDGAR\_bulk\_scraper3.py, jb\_inquisitor.sh
   2. jb\_clean.sh, jb\_clean\_large.sh
   3. get\_whitelist.py, jb\_whitelist.sh., jb\_subsetmerge.sh

   4. jb\_shuf.sh
   5. jb\_tokenize.sh, jb\_merge.sh (note that jb\_merge.sh was used to salvage tokenized data that failed to merge due to an OOM-kill event. Ideally only jb\_tokenize.sh is needed.)

2. create\_venv.sh, jb\_train.sh, jb\_split.sh
3. convert\_tatqa.py, merge\_alpaca\_tatqa.py, jb\_instruct.sh
4. create\_venv\_PIXIU.sh, jb\_eval.sh
5. Reporting scripts may be executed in any order.



Note that all job scripts were executed using the `sbatch` command, and that jb\_whitelist.sh and jb\_subsetmerge.sh are executed with a positional argument denoting the year for which the data is to be subsetted (e.g., `sbatch jb\\\_whitelist.sh 2015`). The output of the domain adaptation, instruction tuning, and evaluation rounds, are contained in SLURM logs in the `Output` subdirectories of their corresponding methodological step. 





