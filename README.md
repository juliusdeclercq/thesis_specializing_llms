# Specializing LLMs in Finance

Code base for my thesis on Specializing LLMs in Finance.


The code is organized by methodology section as much as possible. Directories with miscellaneous code were used for code that could not be efficiently organized through this scheme. Execution order of directories is given by the numbered list (i.e., 1.1 Data collection, 1.2 ..., 2, 3, ..., 5). Within the directories, the scripts were executed in the following order:
1. 1. `EDGAR_bulk_scraper3.py`, `jb_inquisitor.sh`
   2. `crane3.sh`, `jb_clean.sh`, `jb_clean_large.sh`, `crane4.sh`
   3. `get_whitelist.py`, `jb_whitelist.sh`, `jb_subsetmerge.sh`
   4. `jb_tokenize.sh`\*, `jb_merge.sh`\* (NB: `jb_merge.sh` was used to salvage tokenized data that failed to merge due to an OOM-kill event. Ideally only `jb_tokenize.sh` is needed.)
   5. `jb_shuf.sh`\*

2. `create_venv.sh`, `jb_train.sh`\*, `jb_split.sh`\*
3. `convert_tatqa.py`, `merge_alpaca_tatqa.py`, `jb_instruct.sh`**
4. `create_venv_PIXIU.sh`, `jb_eval.sh`***
5. Reporting scripts may be executed in any order.


- \* = these scripts were executed for both long- and short-sequence training (sequence lengths of 8192 and 2048 tokens, respectively).
- \** = these scripts were executed for both the base, and long- and short-sequence domain-adapted models
- \*** = this script was executed for the base, the two domain-adapted, and the three instruction-tuned models.

Note that all job scripts were executed using the `sbatch` command, and that `jb_whitelist.sh` and `jb_subsetmerge.sh` are executed with a positional argument denoting the year for which the data is to be subsetted (e.g., `sbatch jb_whitelist.sh 2015`). The output of the domain adaptation, instruction tuning, and evaluation rounds, are contained in SLURM logs in the `/Output` subdirectories of their corresponding methodological step. 

The models are accessible through [a collection on Hugging Face](https://huggingface.co/collections/juliusdeclercq/thesis-specializing-llms-in-finance-68d05d92d5e04dad38df14d4).



