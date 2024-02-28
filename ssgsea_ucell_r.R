# ###### ssGSEA analysis performed in R using GSVA package

library(GSVA)
library(data.table)
tpm = read.csv('brown_adipose_expression_tpm.tsv', row.names = 1, sep = '\t')
geneSet = readRDS('/Users/rongbinzheng//Documents//CommonData/KEGG_Pathway/mouse_KEGG_terms_symbol.rds')
met_sensor = read.csv('/Users/rongbinzheng/Documents/github/MEBOCOST/data/mebocost_db/mouse/mouse_met_sen_October-25-2022_14-52-47.tsv',
                     sep = '\t')
met_enzyme = read.csv('/Users/rongbinzheng/Documents/github/MEBOCOST/data/mebocost_db/mouse/metabolite_associated_gene_reaction_HMDB_summary_mouse.tsv',
                     sep = '\t')

met_genes = NULL

for (g in subset(met_enzyme, HMDB_ID %in% as.vector(met_sensor$HMDB_ID))$gene){
    met_genes = c(met_genes, sapply(strsplit(g, '\\;')[[1]], function(x) {strsplit(x, '\\[')[[1]][1]}))
}
met_genes = unique(c(met_genes, met_sensor$Gene_name))
geneSet_new = lapply(geneSet, function(x){
    setdiff(x, met_genes)
})
res_new = GSVA::gsva(as.matrix(log2(tpm+1)), geneSet_new, method = 'ssgsea', mx.diff=FALSE)
write.csv(res_new, file = 'ssGSEA_bulkBAT_withoutMetGenes.csv')

### UCell software to comptue pathway enrichment in single-cell level
library(UCell)
exp.mat <- fread('./scanpy_exp/cold2_sc_exp_updated_for_scFEA.csv')
genenames = as.vector(exp.mat$V1)
exp.mat = exp.mat[,2:ncol(exp.mat)]
exp.mat = as.data.frame(exp.mat)
rownames(exp.mat) = genenames
pathway = readRDS('/lab-share/Cardio-Chen-e2/Public/rongbinzheng/CommonData/mouse_KEGG_terms_symbol.rds')
## rank genes
ranks <- StoreRankings_UCell(exp.mat)
## compute score for pathway
u.scores.2 <- ScoreSignatures_UCell(features = pathway, precalc.ranks = ranks)
melted <- reshape2::melt(u.scores.2)
colnames(melted) <- c("Cell", "Signature", "UCell_score")
melted = merge(melted, meta, by.x = 'Cell', by.y = 0)
melted$Signature = gsub('_UCell', '', melted$Signature)
write.table(melted, file = 'cold2_sc_exp_UCell_KEGG_pathway_sub_table.tsv', sep = '\t', quote = F)


