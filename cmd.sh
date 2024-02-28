## command line for COMPASS for metabolic flux analysis by scRNA-seq
exptsv_path=$1
species=$2
output_path=$3
temp_path=$4
core=$5
## for groupby cell types
$compass --data $exptsv_path --num-thread $core --species $species --output-dir $output_path --temp-dir $temp_path --calc-metabolites --lambda 0
## for single cell
$compass --data $exptsv_path --num-thread $core --species $species --output-dir $output_path --temp-dir $temp_path --calc-metabolites --lambda 0.25


### command line for STRIDE for integrating scRNA-seq to compute cell type proportion in spots in spatial transcriptomics
## example for heart samples
STRIDE deconvolve --sc-count ../../heart_sc_count.tsv --sc-celltype ../../heart_sc_meta.tsv --st-count ../st_counts.tsv --normalize
