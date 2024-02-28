#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
from matplotlib import pyplot as plt
import pickle as pkl
import os, collections
import scanpy as sc
from mebocost import mebocost
from scipy.stats import spearmanr, pearsonr, ranksums, wilcoxon, ttest_ind, chisquare
from matplotlib.backends.backend_pdf import PdfPages
from scipy.cluster.hierarchy import dendrogram, linkage
# from matplotlib import rcParams
# rcParams['font.family'] = 'Arial'
import gseapy as gp

import magic

import sys
sys.path.append('./scripts/')
import importlib

plt.rcParams.update(plt.rcParamsDefault)
rc={"axes.labelsize": 16, "xtick.labelsize": 12, "ytick.labelsize": 12,
    "figure.titleweight":"bold", #"font.size":14,
    "figure.figsize":(5.5,4.2), "font.weight":"regular", "legend.fontsize":10,
    'axes.labelpad':8, 'figure.dpi':300}
plt.rcParams.update(**rc)



# In[16]:


df = pd.DataFrame([[1,2], [3,4], ['adipose', 'aEC'], ['Kidney', 'vEC']]).T
#                   , index = [('adipose', 'aEC'), ('Kidney', 'vEC')],
#                  columns = ['PC1', 'PC2'])
df = df.groupby([2, 3]).mean()
df.index

sns.scatterplot(data = df, x= , y = , hue = 'tissue', style = '')


# ## effects from circulating system

# In[203]:


## human
blood_hmdb = pd.read_csv('../../../../data_collection/Human_Metabolome_Database/HMDB_met_blood_concentration.csv',
                        index_col = 0)
blood_hmdb_met = blood_hmdb[blood_hmdb['biospecimen'].isin(['Blood'])].groupby('HMDBID')['concentration'].median().sort_values()#.reset_index()

## mouse
met_ann = pd.read_csv('/Users/rongbinzheng/Documents/github/MEBOCOST/data/mebocost_db/common/metabolite_annotation_HMDB_summary.tsv',
                     sep = '\t')
alias = {str(i).upper():[str(x).upper() for x in str(j).split('; ')] for i,j in met_ann[['metabolite', 'synonyms_name']].values.tolist()}

mblood_met = pd.read_excel('../../../../mouse_plasma/To Rongbin in serum metabolomics data in cold vs TN Luiz 2017.xlsx')
mblood_met.index = mblood_met.iloc[:,1]+'_s'+mblood_met.iloc[:,0].astype('str')
mblood_met = mblood_met.drop(['Omics #', 'Group'], axis = 1)
mblood_met = mblood_met.T
mblood_met_avg = pd.DataFrame([mblood_met.loc[:,mblood_met.columns.str.startswith('Cold')].T.mean(),
                               mblood_met.loc[:,mblood_met.columns.str.startswith('Neutral')].T.mean()],
                             index = ['Cold', 'TN']).T
mblood_met_avg_matched = {x: mblood_met_avg.loc[mblood_met_avg.index.isin([str(i).upper()]) | mblood_met_avg.index.isin(alias.get(str(i).upper(), []))] for x, i in met_ann[['HMDB_ID', 'metabolite']].values.tolist()}
mblood_met_avg_matched_dict = {x:mblood_met_avg_matched[x].iloc[0,:].mean() for x in mblood_met_avg_matched if mblood_met_avg_matched[x].shape[0] != 0}
mblood_met_avg_matched = pd.Series(mblood_met_avg_matched_dict)



# In[204]:


from sklearn.linear_model import LinearRegression

def _blood_correct_test_(commu_res, blood_cont, commu_score_col = 'Commu_Score', title = '', pdf = False):

#     commu_res['label'] = [' : '.join(sorted(x)) for x in commu_res[['Sender', 'Receiver']].values.tolist()]

    commu_res['blood_level'] = [np.nan if x not in blood_cont.index.tolist() else blood_cont[x] for x in commu_res['Metabolite'].tolist()]
    commu_res['blood_level'] = np.log(commu_res['blood_level'])
    commu_res['blood_level'] = (commu_res['blood_level'] - commu_res['blood_level'].min()) / (commu_res['blood_level'].max() - commu_res['blood_level'].min())
    commu_res = commu_res[~pd.isna(commu_res['blood_level'])]#[['blood_level', 'Commu_Score']]
    plotm = commu_res.drop_duplicates(['Sender', 'Metabolite'])
    rm, pm = pearsonr(plotm['blood_level'], plotm['met'])
    plote = commu_res.drop_duplicates(['Receiver', 'Sensor'])
    re, pe = pearsonr(plote['blood_level'], plote['exp'])
    rme, pme = pearsonr(commu_res['met'], commu_res['exp'])
    r1, p1 = pearsonr(commu_res['blood_level'], commu_res['Commu_Score'])
    rmc, pmc = pearsonr(commu_res['met'], commu_res['Commu_Score'])
    rec, pec = pearsonr(commu_res['exp'], commu_res['Commu_Score'])

    model = LinearRegression(fit_intercept = True)
#     commu_res = commu_res[['Sender', 'Metabolite', 'Metabolite_Name', 'Receiver', 'Sensor', 'Commu_Score', 'blood_level', 'permutation_test_fdr']].dropna()
    model.fit(commu_res[['blood_level']], commu_res['Commu_Score'])
    commu_res['pred'] = model.predict(commu_res[['blood_level']])
    commu_res['corrected_commu'] = commu_res['Commu_Score'] - commu_res['pred']
    r2, p2 = pearsonr(commu_res['blood_level'], commu_res['corrected_commu'])
    fig, ax = plt.subplots(figsize = (22, 4), nrows = 1, ncols = 5)
    sns.regplot(data = plotm,
                x = 'blood_level', y = 'met', ci = False, ax = ax[0],
               scatter_kws={'alpha':.5})
    sns.regplot(data = plote,
                x = 'blood_level', y = 'exp', ci = False, ax = ax[1],
               scatter_kws={'alpha':.5})
    sns.regplot(data = commu_res,
                x = 'met', y = 'exp', ci = False, ax = ax[2],
               scatter_kws={'alpha':.5})
    sns.regplot(data = commu_res,
                x = 'blood_level', y = 'Commu_Score', ci = False, ax = ax[3],
               scatter_kws={'alpha':.5})
    sns.regplot(data = commu_res,
                x = 'blood_level', y = 'corrected_commu', ci = False, ax = ax[4],
               scatter_kws={'alpha':.5})
#     plt.xlabel('Metabolite level in blood (log)')
#     ax.set_ylabel('Communication score')
    fig.suptitle(title)
    sns.despine()
    plt.tight_layout()
    pdf.savefig(fig) if pdf else plt.show()
#     plt.show()
    plt.close()
    return(re, pe, rm, pm, rme, pme, rmc, pmc, rec, pec, r1, p1, r2, p2)


# In[205]:


mebo_samples = {'human_adipose': '../../../../update_2023/disco/mebocost_res/adipose.h5ad.Normal.mebocost.pk',
         'human_bone_marrow': '../../../../update_2023/disco/mebocost_res/bone_marrow.h5ad.Normal.mebocost.pk',
         'human_brain': '../../../../update_2023/disco/mebocost_res/brain.h5ad.Normal.mebocost.pk',
         'human_breast_milk': '../../../../update_2023/disco/mebocost_res/breast_milk.h5ad.Normal.mebocost.pk',
         'human_kidney': '../../../../update_2023/disco/mebocost_res/kidney.h5ad.Normal.mebocost.pk',
         'human_liver': '../../../../update_2023/disco/mebocost_res/liver.h5ad.Normal.mebocost.pk',
         'human_lung': '../../../../update_2023/disco/mebocost_res/lung.h5ad.Normal.mebocost.pk',
         'human_pancreas': '../../../../update_2023/disco/mebocost_res/pancreas.h5ad.Normal.mebocost.pk',
         'human_skin': '../../../../update_2023/disco/mebocost_res/skin.h5ad.Normal.mebocost.pk',
          'human_BAT': '/Users/rongbinzheng/Documents/BCH/ChenLab/Metabolism/human_adipose/E-MTAB-8564/human_scBAT_mebocost.pk',
          'human_heart':'../../../../update_2023/evaluation/spatial/CCI_datasets/human_heart/sample_8_st/mebocost_res.pk',
          'human_intestinal_A3':'../../../../update_2023/evaluation/spatial/CCI_datasets/human_intestinal/ST_A3_GSM4797918/mebocost_res.pk',
          'human_intestinal_A4':'../../../../update_2023/evaluation/spatial/CCI_datasets/human_intestinal/ST_A4_GSM4797919/mebocost_res.pk'}


# In[10]:


comm_res_dict = {}
prop_cutoff = 0.2
for sample in mebo_samples:
    mebo_obj = mebocost.load_obj(path = mebo_samples[sample])
    avg_exp = pd.DataFrame(mebo_obj.avg_exp.toarray(), index = mebo_obj.avg_exp_indexer, columns = mebo_obj.avg_exp_columns)
    avg_met = pd.DataFrame(mebo_obj.avg_met.toarray(), index = mebo_obj.avg_met_indexer, columns = mebo_obj.avg_met_columns)
    comm_res = mebo_obj._filter_lowly_aboundant_(pvalue_res = mebo_obj.original_result.copy(),
                                                    cutoff_prop = prop_cutoff,
                                                    met_prop=mebo_obj.met_prop,
                                                    exp_prop=mebo_obj.exp_prop,
                                                    min_cell_number = 10
                                                 )
#     commu_res_good = commu_res[(commu_res['permutation_test_fdr'] < 0.05)]
    comm_res['met'] = [avg_met.loc[j,i] for i,j in comm_res[['Sender', 'Metabolite']].values]
    comm_res['exp'] = [avg_exp.loc[j,i] for i,j in comm_res[['Receiver', 'Sensor']].values]
    comm_res['label'] = [' : '.join(sorted(x)) for x in comm_res[['Sender', 'Receiver']].values.tolist()]
    comm_res = comm_res.sort_values(['Sender', 'Receiver', 'Metabolite', 'Sensor'])
    comm_res['Annotation'] = mebo_obj.original_result.sort_values(['Sender', 'Receiver', 'Metabolite', 'Sensor'])['Annotation'].tolist()
    comm_res_dict[sample] = {'all':comm_res, 'sig':comm_res[(comm_res['permutation_test_fdr'] < 0.05)]}

    


# In[206]:


# import pickle as pk
# out = open('../../../../update_2023/disco/disco_mCCC_commu_res_dict.pk', 'wb')
# pk.dump(comm_res_dict, out)
# out.close()

import pickle as pk
comm_res_dict = pk.load(open('../../../../update_2023/disco/disco_mCCC_commu_res_dict.pk', 'rb'))


# In[208]:


corr_res = {}
pdf=PdfPages('plots/blood_effect_measures.pdf')
for sample in mebo_samples:
    corr_res[sample] = _blood_correct_test_(commu_res=comm_res_dict[sample]['sig'],
                                                blood_cont=blood_hmdb_met,
                                                commu_score_col = 'Commu_Score', title = sample, pdf = pdf)
pdf.close()


# In[219]:


## plot
corr_res_df = pd.DataFrame(corr_res).T
corr_res_df.columns = ['Sensor_Corr', 'Sensor_p', 'Met_Corr', 'Met_p', 
                       'Met_Sensor_Corr', 'Met_Sensor_p', 'Met_Comm_Corr', 'Met_Comm_p',
                        'Sensor_Comm_Corr', 'Sensor_Comm_p',
                       'Comm_Corr', 'Comm_p', 'Corrected_Comm_Corr', 'Corrected_Comm_p']


plot_df = corr_res_df.loc[:,corr_res_df.columns.str.endswith('_Corr')].drop(['Met_Sensor_Corr'], axis = 1)

yorder = ['Met_Comm_Corr', 'Sensor_Comm_Corr', 'Comm_Corr', 'Sensor_Corr', 'Met_Corr', 'Corrected_Comm_Corr']
fig, ax = plt.subplots(figsize = (10, 5))
sns.heatmap(data = plot_df.sort_values('Comm_Corr').T.loc[yorder,], cmap = 'coolwarm', linewidth = .5,
            center = 0, vmax = 1, vmin = -1, square = True, 
            annot = plot_df.sort_values('Comm_Corr').T.loc[yorder,], fmt = '.0e',
            cbar_kws={'shrink':0.4, 'label':'Corr Coefficient'})
plt.tight_layout()
fig.savefig('plots/blood_met_corr_heatmap2.pdf')
plt.show()



# In[212]:


fig, ax = plt.subplots(figsize = (3, 2))
sns.boxplot(data = plot_df.T.loc[yorder,].T.melt(), y = 'variable', x = 'value', color = 'darkorange')
# ax.set_xlim(-1, 1)
sns.despine()
ax.set_ylabel('')
ax.set_xlabel('')
ax.set_yticklabels([])
plt.tight_layout()
fig.savefig('plots/blood_met_corr_boxplot.pdf')
plt.show()


# In[215]:


ttest_ind(plot_df['Met_Comm_Corr'], plot_df['Comm_Corr'], alternative = 'greater')


# In[216]:


ttest_ind(plot_df['Sensor_Comm_Corr'], plot_df['Comm_Corr'], alternative = 'greater')


# ## enzyme genes for steriod hormones and fatty acid

# In[126]:


met_sensor = pd.read_csv('/Users/rongbinzheng/Documents/github/MEBOCOST/data/mebocost_db/human/met_sen_October-25-2022_14-52-47.tsv',
                        sep = '\t')
# met_sensor_update = pd.read_csv('/Users/rongbinzheng/Documents/BCH/ChenLab/Metabolism/MEBOCOST_Verisons/v2.0.0/met_sen_May-18-2023.tsv',
#                         sep = '\t')
met_enzyme = pd.read_csv('/Users/rongbinzheng/Documents/github/MEBOCOST/data/mebocost_db/human/metabolite_associated_gene_reaction_HMDB_summary.tsv',
                        sep = '\t')
met_ann = pd.read_csv('/Users/rongbinzheng/Documents/github/MEBOCOST/data/mebocost_db/common/metabolite_annotation_HMDB_summary.tsv',
                     sep = '\t')

met_gene_new = []
for i, line in met_enzyme.iterrows():
    genes = line['gene'].split('; ')
    for g in genes:
        tmp = line.copy()
        tmp['gene'] = g
        met_gene_new.append(tmp)
met_gene = pd.DataFrame(met_gene_new) ## each row is the related gene annotation for metabolite
met_gene['gene_name'] = met_gene['gene'].apply(lambda x: x.split('[')[0])
met_gene = pd.merge(met_gene, met_ann[['HMDB_ID', 'class', 'super_class']], left_on = 'HMDB_ID', right_on = 'HMDB_ID')


# In[84]:


### look for steroids
good_met = met_gene.query('direction == "product"')['metabolite'].tolist()
good_met = np.intersect1d(good_met, met_sensor['standard_metName'])
steriod_gene = met_gene.dropna()[met_gene.dropna()['class'].str.contains('Steroid')]
steriod_gene = steriod_gene[steriod_gene['metabolite'].isin(good_met)]
steriod_gene = steriod_gene.drop_duplicates(['metabolite', 'gene_name'])
steriod_gene = steriod_gene.query('metabolite != "Vitamin D3"')
steriod_gene['value'] = [1 if x == 'product' else -1 for x in steriod_gene['direction'].tolist()]
dat_steriod = steriod_gene.pivot_table(columns = ['gene_name'], index = ['metabolite'], values = ['value']).fillna(0).astype('int')
dat_steriod.columns = dat_steriod.columns.get_level_values(1).tolist()

### look for lipids
lipd_gene = met_gene.dropna()[met_gene.dropna()['super_class'].str.contains('Lipid')]
lipd_gene = lipd_gene[lipd_gene['metabolite'].isin(good_met)]
lipd_gene = lipd_gene.drop_duplicates(['metabolite', 'gene_name'])
lipd_gene = lipd_gene[~lipd_gene['metabolite'].isin(steriod_gene['metabolite'].tolist()+['Vitamin D3',
                                                   'Dihomo-gamma-linolenic acid', 'Eicosapentaenoic acid', 'Docosahexaenoic acid'])]
lipd_gene['value'] = [1 if x == 'product' else -1 for x in lipd_gene['direction'].tolist()]
dat_lipd = lipd_gene.pivot_table(columns = ['gene_name'], index = ['metabolite'], values = ['value']).fillna(0).astype('int')
dat_lipd.columns = dat_lipd.columns.get_level_values(1).tolist()


## plot 
dat = pd.concat([dat_steriod, dat_lipd]).fillna(0)
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(data = dat, cmap = 'coolwarm', xticklabels = False, linewidth = .1)
fig.savefig('plots/steroid_lipid_enzyme_genes_heatmap.pdf')
plt.show()


# ## macromolecules and cell-specific sensor signaling
# #### macromolecules include Adenosine diphosphate ribose, Adenosine triphosphate, ADP

# In[5]:


cold2_comm = pd.read_excel('../../../../update_2023/BAT_update/detected_communication_conditions.xlsx', sheet_name='Cold2')
cold2_comm['label'] = cold2_comm['Sender']+'~'+cold2_comm['Metabolite_Name']+'~'+cold2_comm['Sensor']+'~'+cold2_comm['Receiver']

cold2_mebo = mebocost.load_obj('../../../../mebocost_BAT_v1.0.2/cold2/scBAT_mebocost_cold2.pk')
cold2_mebo.commu_res['label'] = cold2_mebo.commu_res['Sender']+'~'+cold2_mebo.commu_res['Metabolite_Name']+'~'+cold2_mebo.commu_res['Sensor']+'~'+cold2_mebo.commu_res['Receiver']
cold2_mebo.commu_res = cold2_mebo.commu_res[cold2_mebo.commu_res['label'].isin(cold2_comm['label'])]


cold2_mebo.FlowPlot(pval_method='permutation_test_fdr',
                pval_cutoff=0.05,
                sender_focus=[],
                metabolite_focus=['Adenosine triphosphate'],
                sensor_focus=[],
                receiver_focus=[],
                remove_unrelevant = True,
                and_or='and',
                node_label_size=12,
                node_alpha=0.6,
                figsize=(6.5, 5),
                node_cmap='Set1',
                line_cmap='bwr',
                line_vmin = None,
                line_vmax = 15.5,
                node_size_norm=(20, 150),
                linewidth_norm=(0.5, 5),
                save='./plots/ATP_flowplot_cold2.pdf',
                show_plot=True,
                comm_score_col='Commu_Score',
                comm_score_cutoff=None,
                text_outline=False,
                return_fig = False)


# ### UCell for downstream pathway activity in single cell level

# In[ ]:


exp_mat = pd.DataFrame(cold2_mebo.exp_mat.toarray(),
                      index = cold2_mebo.exp_mat_indexer, columns = cold2_mebo.exp_mat_columns)


# In[47]:


colmap = {'Adipocytes': '#d62728',
         'Basophils': '#aec7e8',
         'Bcells': '#ff7f0e',
         'CD4T': '#ffbb78',
         'CD8T': '#2ca02c',
         'Cytotoxic_T': '#98df8a',
         'EC': '#1f77b4',
         'Erythroid-like': '#ff9896',
         'ILC2s': '#9467bd',
         'Lymph_EC': '#c5b0d5',
         'MSC': '#8c564b',
         'Macrophages': '#c49c94',
         'NK': '#e377c2',
         'NMSC': '#f7b6d2',
         'Neutrophils': '#7f7f7f',
         'Pdgfra_APC': '#c7c7c7',
         'Pericytes': '#bcbd22',
         'Platelets': '#dbdb8d',
         'Treg': '#17becf',
         'VSM': '#9edae5',
         'Non.Receiver':'lightgrey'}


# In[124]:


receivers = ['VSM', 'Platelets', 'NMSC', 'Macrophages', 'MSC', 'Basophils']

ucell_res = pd.read_csv('./data/cold2_sc_exp_UCell_KEGG_pathway_sub_table.tsv', sep = '\t')


index = (exp_mat.loc['P2rx1']>0) | (exp_mat.loc['P2rx4']>0) | (exp_mat.loc['P2rx7']>0)

exp_cells = exp_mat.columns[index].tolist()
pways = ['mmu04020 Calcium signaling pathway']
df = ucell_res[(ucell_res['cell_type'].isin(receivers)) & (ucell_res['Signature'].isin(pways))]
df = df[df['Cell'].isin(exp_cells)]
df1 = ucell_res[(~ucell_res['cell_type'].isin(receivers)) & (ucell_res['Signature'].isin(pways))]
df1['cell_type'] = 'Non.Receiver'
df = pd.concat([df, df1])
#     print(df.groupby(['Signature', 'cell_type'])['UCell_score'].median())

pdf=PdfPages('plots/cold2_P2rx_UCell_Calcium_pathway_scores.pdf')
fig,ax = plt.subplots(figsize=(5, 5))
sns.boxplot(data = df,
            x = 'cell_type', y = 'UCell_score', #hue = 'cell_type',
            showfliers = False, width = .7, #palette=colmap, 
            color = 'lightblue')
ax.set(ylabel=pways[0], xlabel = '')
ax.tick_params(axis = 'x', rotation = 90)
ax.hlines(np.median(df[df['cell_type'] == 'Non.Receiver']['UCell_score']),
         *ax.get_xlim(), linestyle = 'dashed', color = 'black', zorder = 100, linewidth = .5)
sns.despine()
plt.tight_layout()
plt.show()

p_res = df.groupby('Signature').apply(lambda x: {j:ranksums(x[x['cell_type'] == j]['UCell_score'], 
                                                x[x['cell_type'] == 'Non.Receiver']['UCell_score'],
                                                           alternative = 'greater')[1] for j in x['cell_type'].unique().tolist() if j != 'Non.Receiver'})
print(p_res.to_dict())
pdf.savefig(fig)
pdf.close()


# In[ ]:




