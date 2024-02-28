#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import pickle as pk
from matplotlib import pyplot as plt
import pickle as pkl
import os, collections
import scanpy as sc
from mebocost import mebocost
from scipy.stats import spearmanr, pearsonr, ranksums, wilcoxon, ttest_ind, chisquare, mannwhitneyu
from matplotlib.backends.backend_pdf import PdfPages
from scipy.cluster.hierarchy import dendrogram, linkage
# from matplotlib import rcParams
# rcParams['font.family'] = 'Arial'
from statsmodels import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.preprocessing import StandardScaler, MinMaxScaler
import statsmodels.api as sm
from sklearn import metrics

import magic

plt.rcParams.update(plt.rcParamsDefault)
rc={"axes.labelsize": 16, "xtick.labelsize": 12, "ytick.labelsize": 12,
    "figure.titleweight":"bold", #"font.size":14,
    "figure.figsize":(5.5,4.2), "font.weight":"regular", "legend.fontsize":10,
    'axes.labelpad':8, 'figure.dpi':300}
plt.rcParams.update(**rc)


colmap = {'Adipocytes': '#1f77b4',
         'Basophils': '#aec7e8',
         'Bcells': '#ff7f0e',
         'CD4T':'#ffbb78',
         'CD8T': '#2ca02c',
         'Cytotoxic_T': '#98df8a',
         'EC': '#d62728',
         'Erythroid-like': '#ff9896',
         'ILC2s': '#9467bd',
         'Lymph_EC':'#c5b0d5',
         'MSC':'#8c564b',
         'Macrophages':'#c49c94',
         'NK':'#e377c2',
         'NMSC':'#f7b6d2',
         'Neutrophils':'#7f7f7f',
         'Pdgfra_APC':'#c7c7c7',
         'Pericytes':'#bcbd22',
         'Platelets':'#dbdb8d',
         'Treg':'#17becf',
         'VSM':'#9edae5'}


# In[2]:


## CDS annotation
cds_ann = pd.read_csv('/Users/rongbinzheng/Documents/CommonData/mm10/gencode.vM23.annotation.gene_cds_length.csv',
                     index_col = 0)


# In[3]:


## count data downloaded from ARCHS4, normalized to TPM values
bat_count = pd.read_csv('./brown_adipose_expression_matrix.tsv', sep = '\t', index_col = 0)
common_genes = np.intersect1d(bat_count.index, cds_ann.index)
RPK = bat_count.loc[common_genes,].apply(lambda col: col * 1000 / cds_ann.loc[common_genes,'CDS_bp'])
index = RPK.sum() / 1e06
tpm = RPK.apply(lambda row: row / index, axis = 1)


# In[4]:


# tpm.to_csv('brown_adipose_expression_tpm.tsv', sep = '\t')
tpm = pd.read_csv('../../../../update_2023/evaluation/bulkBAT/brown_adipose_expression_tpm.tsv', index_col = 0, sep = '\t')
tpm_log = np.log2(tpm+1)


# ###### ssGSEA analysis performed in R using GSVA package

# In[6]:


## back to python
ssgsea_new = pd.read_csv('../../../../update_2023/evaluation/bulkBAT/ssGSEA_bulkBAT_withoutMetGenes.csv', index_col = 0)
ssgsea_new.head(2)


# In[7]:


## metabolite enzymes aggeration for bulk data
bulk_met = mebocost.create_obj(exp_mat=np.log2(tpm+1), 
                               cell_ann = pd.DataFrame(tpm.columns.tolist(), index = tpm.columns,
                                                                              columns = ['cell_type']),
                              group_col = ['cell_type'],
                               species = 'mouse',
                              config_path = '/Users/rongbinzheng/Documents/test/MEBOCOST/mebocost.conf')
bulk_met._load_config_()
bulk_met.estimator()
bulk_met_mat = pd.DataFrame(bulk_met.met_mat.toarray(),
                            index = bulk_met.met_mat_indexer,
                            columns = bulk_met.met_mat_columns)


# ## test metabolite-sensor pairs in BAT bulk RNA-seq
# 

# In[8]:


cold2_mebo = mebocost.load_obj('../../../../mebocost_BAT_v1.0.2/cold2/scBAT_mebocost_cold2.pk')


# In[9]:


met_sen_db = cold2_mebo.met_sensor.copy()
met_ann_db = cold2_mebo.met_ann.copy()
met_enzyme_db = cold2_mebo.met_enzyme.copy()
met_name = {m:n for m,n in met_ann_db[['HMDB_ID', 'metabolite']].values.tolist()}


# In[10]:


cold2_commu_res = pd.read_excel('../../../../update_2023/BAT_update/detected_communication_conditions.xlsx', sheet_name='Cold2')
cold2_commu_res['met_sen'] = cold2_commu_res['Metabolite']+'~'+cold2_commu_res['Sensor']
RT_commu_res = pd.read_excel('../../../../update_2023/BAT_update/detected_communication_conditions.xlsx', sheet_name='RT')
RT_commu_res['met_sen'] = RT_commu_res['Metabolite']+'~'+cold2_commu_res['Sensor']
TN_commu_res = pd.read_excel('../../../../update_2023/BAT_update/detected_communication_conditions.xlsx', sheet_name='TN')
TN_commu_res['met_sen'] = TN_commu_res['Metabolite']+'~'+cold2_commu_res['Sensor']
cold7_commu_res = pd.read_excel('../../../../update_2023/BAT_update/detected_communication_conditions.xlsx', sheet_name='Cold7')
cold7_commu_res['met_sen'] = cold7_commu_res['Metabolite']+'~'+cold2_commu_res['Sensor']

## 
most_var_comm = pd.read_csv('../../../../update_2023/BAT_update/most_var_comm.tsv', sep = '\t', index_col = 0)
most_var_comm.head()


# In[11]:


all_met_sen = met_sen_db['HMDB_ID']+'~'+met_sen_db['Gene_name']
all_met_sen.head()


# #### correlation between metabolite enzyme-sensor expression and pathway enrichment score
# 

# In[12]:


corr_stat_allpathway = []
for i in all_met_sen.tolist():
    m, s = i.split('~')
    mn = met_name[m]
    mv, sv = bulk_met_mat.loc[m], tpm_log.loc[s]
    copre = mv * sv
    for p in ssgsea_new.index.tolist():
        scores_tmp = ssgsea_new.loc[p]
        r1, p1 =spearmanr(scores_tmp, copre[scores_tmp.index])
        r2, p2 =pearsonr(scores_tmp, copre[scores_tmp.index])
        corr_stat_allpathway.append([i, m, mn, s, p, r1, p1, r2, p2])
    
corr_stat_allpathway = pd.DataFrame(corr_stat_allpathway, 
                                columns = ['ms', 'm', 'mn', 's', 'pathway', 'sp_r_co', 'sp_p_co', 'pr_r_co', 'pr_p_co'])
corr_stat_allpathway['pair1'] = corr_stat_allpathway['m']+'~'+corr_stat_allpathway['s']
corr_stat_allpathway['pair2'] = corr_stat_allpathway['mn']+'~'+corr_stat_allpathway['s']

corr_stat_allpathway['TN'] = corr_stat_allpathway.pair1.isin(TN_commu_res['met_sen'].unique())
corr_stat_allpathway['RT'] = corr_stat_allpathway.pair1.isin(RT_commu_res['met_sen'].unique())
corr_stat_allpathway['Cold2'] = corr_stat_allpathway.pair1.isin(cold2_commu_res['met_sen'].unique())
corr_stat_allpathway['Cold7'] = corr_stat_allpathway.pair1.isin(cold7_commu_res['met_sen'].unique())



# In[53]:


## pathway genes
pathway = pd.read_csv('/Users/rongbinzheng/Documents/CommonData/KEGG_Pathway/mouse_KEGG_terms_symbol.txt',
                     sep = '\t', index_col = 0, header = None)

gene_len = pathway[1].apply(lambda x: len(x.split(';'))).sort_values()
### exclude small pathway and large pathway 
good_pathway = gene_len[(gene_len>50) & (gene_len < 100)].index.tolist()

plot_df = corr_stat_allpathway.pivot_table(index = 'pair2', columns = 'pathway', values = 'sp_r_co')
cold2_pair = corr_stat_allpathway.query('Cold2 == True')['pair2'].unique()

# fig, ax = plt.subplots()
g = sns.clustermap(plot_df.loc[cold2_pair, good_pathway],
                   center = 0,
                   xticklabels = True,
               cmap = 'bwr', figsize = (18, 14),
                cbar_pos=(0.8, 1, .15, .02),
                   dendrogram_ratio = 0.01,
                cbar_kws=dict(orientation='horizontal')
                  )
g.savefig('plots/Lipolysis_all_mCCC_heatmap_cold2.pdf')
plt.show()

xtick_order = []
for i in g.ax_heatmap.get_xticklabels():
    xtick_order.append(i.get_text())

sum_values = plot_df.loc[cold2_pair, good_pathway].sum()[xtick_order]
import matplotlib
from adjustText import adjust_text


norm = matplotlib.colors.Normalize(vmin = -5, vmax = 5)
my_cmap = plt.cm.get_cmap('RdBu_r')
color = my_cmap(norm(sum_values))

fig, ax = plt.subplots(figsize = (18, 10))
sm = ax.bar(sum_values.index, sum_values)
ax.hlines([-3,0, 3], *ax.get_xlim(), linestyle = 'dashed', color = 'grey')
for i in range(len(sum_values)):
    ax.vlines(i, -4.5, 4.5, color = 'lightgrey', zorder = -100)
ax.set_xlim(-1, 75)
ax.set(xlabel = '', ylabel='Sum')
# ax.set_xticks([])
ax.tick_params(axis = 'x', rotation = 90)
sns.despine()
plt.tight_layout()
fig.savefig('plots/Lipolysis_all_mCCC_sum_barplot_cold2.pdf')
plt.show()



# In[152]:


#### wordcloud analysis for pathway names for the two clusters
from wordcloud import WordCloud
import random

def grey_color_func(word, font_size, position, orientation, random_state=None,
                    **kwargs):
    return "black" 

# Generate a word cloud image
## remove a pathway that is ambiguous for two clusters
text = xtick_order[:-xtick_order.index('mmu04978 Mineral absorption')]  
text = '\n'.join([' '.join(x.split(' ')[1:]) for x in text]).replace('%2F', '/') ## remove KEGG ID
wc = WordCloud(background_color="white", max_words=50,random_state=1000,
               width=700, height=360,
               max_font_size=50,min_font_size=1).generate(text)

plt.imshow(wc.recolor(color_func=grey_color_func), interpolation='bilinear', cmap = 'binary')
plt.axis("off")
plt.tight_layout()
plt.savefig('plots/woldclound_metabolism.pdf', dpi = 300)
plt.show()
plt.close()

text = xtick_order[xtick_order.index('mmu04978 Mineral absorption'):]
text = '\n'.join([' '.join(x.split(' ')[1:]) for x in text]).replace('%2F', '/') ## remove KEGG ID
wc = WordCloud(background_color="white", max_words=50,random_state=1000, 
               width=700, height=360,
               max_font_size=50,min_font_size=1).generate(text)

plt.imshow(wc.recolor(color_func=grey_color_func), interpolation='bilinear', cmap = 'binary')
plt.axis("off")
plt.tight_layout()
plt.savefig('plots/woldclound_signaling.pdf', dpi = 300)
plt.show()
plt.close()

