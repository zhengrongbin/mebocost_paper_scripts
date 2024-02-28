import os,sys
from cellphonedb.src.core.methods import cpdb_statistical_analysis_method

meta_file = sys.argv[1]
count_file = sys.argv[2]
out_path = sys.argv[3]

## cellphonedb, downloaded at Mar 9, 2023
deconvoluted, means, pvalues, significant_means = cpdb_statistical_analysis_method.call(
        cpdb_file_path = 'cellphonedb.zip',
        meta_file_path = meta_file,
        counts_file_path = count_file,
        counts_data = 'hgnc_symbol',
        output_path = out_path)