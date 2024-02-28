# mebocost_paper_scripts
the python, R, and bash scripts for mebocost manuscript

##### brown adipsoe tissue 
BAT_summary.py: scripts to summarize results for mCCC in brown adipose tissue detected by MEBOCOST 
bat_gene_interaction.py: analyze pathway activity and mCCC enzyme-sensor expression in brown adipose tissue based on bulk RNA-seq
mouseBAT_scRNA_py.py: analyze scRNA-seq data using scanpy
##### benchmarking analysis 
benchmark_crispr_screen.py: benchmark MEBOCOST using CRISPR screen dataset and TCGA patient survival data for colorectal cancer
benchmark_spatial.py: benchmark MEBOCOST using spatial transcriptomics, including addressing two questions from reviewers
human_white_adipose.py: scripts to analyze mCCC dynamics in obesity by MEBOCOST
##### compare with other tools
cellphonedb_run.py: predict communications by CellPhoneDB
neurochat_run.R: predict communications by neuronChat
scConnect_run.py: predict communications by scConnect
##### other
ssgsea_ucell_r.R: scripts for UCell to compute pathway enrichment in single cells
cmd.sh: command lines to run COMPASS for flux analysis and STRIDE for deconvolution of spatial transcriptomics
Response_Reviewer.py: scripts to perform analysis to address reviewer questions

