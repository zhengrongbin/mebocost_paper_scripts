import os,sys
import pandas as pd
import scanpy as sc
import scConnect as cn
### scConnect downloaded at Feb 2, 2024

h5ad_file = sys.argv[1]
label=sys.argv[2]

adata = sc.read_h5ad(h5ad_file)

adata_tissue = cn.genecall.meanExpression(adata, groupby="celltype",
                                          normalization=False, use_raw=False, transformation="")
adata_tissue = cn.connect.ligands(adata_tissue, organism='hsapiens')
adata_tissue = cn.connect.receptors(adata_tissue, organism='hsapiens')
adata_tissue = cn.connect.specificity(adata_tissue, n=1000, groupby="celltype", organism='hsapiens')
cn.connect.save_specificity(adata_tissue, "scConnect_res/%s_celltype_specificity.xlsx"%label)

#cn.connect.load_specificity(adata_tissue, "scConnect_res/%s_celltype_specificity.xlsx"%label)

edges = cn.connect.interactions(emitter=adata_tissue, target=adata_tissue, self_reference=True, organism='hsapiens')
nodes = cn.connect.nodes(adata_tissue)
G_tissue = cn.graph.build_graph(edges, nodes)
Gs_tissue = cn.graph.split_graph(G_tissue)
edge_new = []
for x in edges:
    tmp = x[-1]
    tmp['sender'] = x[0]
    tmp['receiver'] = x[1]
    edge_new.append(tmp)
edge_new_df = pd.DataFrame(edge_new)
edge_new_df.to_csv('scConnect_res/%s_scConnect_res.tsv'%label, sep = '\t')