#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import pickle as pk
import os, collections
import scanpy as sc
from mebocost import mebocost
from scipy.stats import spearmanr, pearsonr, ranksums, wilcoxon, ttest_ind, chisquare, kstest, mannwhitneyu
from statsmodels.stats.multitest import fdrcorrection
from statsmodels.stats.weightstats import ztest
from matplotlib.backends.backend_pdf import PdfPages
from scipy.cluster.hierarchy import dendrogram, linkage
# from matplotlib import rcParams
# rcParams['font.family'] = 'Arial'
import h5py
# import sys
# sys.path.append('./scripts/')
# import importlib
import cobra
from cobra.io import read_sbml_model, write_sbml_model

plt.rcParams.update(plt.rcParamsDefault)
rc={"axes.labelsize": 16, "xtick.labelsize": 12, "ytick.labelsize": 12,
    "figure.titleweight":"bold", #"font.size":14,
    "figure.figsize":(5.5,4.2), "font.weight":"regular", "legend.fontsize":10,
    'axes.labelpad':8, 'figure.dpi':300}
plt.rcParams.update(**rc)




# In[2]:


### expression matrix and meta data for CRC smart-seq2 scFRNA-seq
crc_exp = pd.read_csv('../../../../update_2023/evaluation/crispr_screen_coculture/coloncancer/CRC_GSE146771_Smartseq2_expression.csv', index_col = 0)
crc_meta = pd.read_csv('../../../../update_2023/evaluation/crispr_screen_coculture/coloncancer/CRC_GSE146771_Smartseq2_CellMetainfo_table.tsv', 
                       sep = '\t', index_col = 0)


# In[4]:


## reorganize the meta table
crc_meta_new = crc_meta.copy()
celltypes = crc_meta_new['Celltype (major-lineage)'].unique()
asg_id = pd.Series(range(0, celltypes.shape[0])).astype('str')+': '+celltypes
asg_id.index = celltypes
crc_meta_new['celltype'] = [asg_id[x] for x in crc_meta_new['Celltype (major-lineage)'].tolist()]
label_df = crc_meta_new.groupby(['celltype'])[['UMAP_1', 'UMAP_2']].mean()
label_df['label'] = [x.split(': ')[0] for x in label_df.index.tolist()]



# In[6]:


## plot UMAP
fig, ax = plt.subplots(figsize = (5, 3.5))
sns.scatterplot(data = crc_meta_new, x = 'UMAP_1', y = 'UMAP_2',
               hue = 'celltype', edgecolor = 'none',
               palette='tab20', s = 2, alpha = .6, zorder = 100)
for i, line in label_df.iterrows():
    ax.text(line['UMAP_1'], line['UMAP_2'], line['label'], zorder = 100)
sns.despine()
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.grid(zorder = -5)
plt.tight_layout()
fig.savefig('plots/CRC_umap.pdf')
plt.show()


# ### MEBOCOST for CRC
# 

# In[5]:


## compass flux
compass_met_ann = pd.read_csv('/Users/rongbinzheng/Documents/test/MEBOCOST/data/Compass/met_md.csv')

compass_rxn_ann = pd.read_csv('/Users/rongbinzheng/Documents/test/MEBOCOST/data/Compass/rxn_md.csv')
met_ann = pd.read_csv('/Users/rongbinzheng/Documents/github/MEBOCOST/data/mebocost_db/common/metabolite_annotation_HMDB_summary.tsv',
                     sep = '\t')

def _get_compass_flux_(sample_path):  
    uptake_path = os.path.join(sample_path, 'uptake.tsv')
    secret_path = os.path.join(sample_path, 'secretions.tsv')
    reaction_path = os.path.join(sample_path, 'reactions.tsv')

    uptake = pd.read_csv(uptake_path, index_col = 0, sep = '\t')
    secretion = pd.read_csv(secret_path, index_col = 0, sep = '\t')
    reaction = pd.read_csv(reaction_path, index_col = 0, sep = '\t')
    
    efflux_mat = pd.merge(secretion, compass_met_ann[['met', 'hmdbID']],
                            left_index = True, right_on = 'met').dropna()
    efflux_mat = pd.merge(efflux_mat, met_ann[['Secondary_HMDB_ID', 'metabolite']],
                            left_on = 'hmdbID', right_on = 'Secondary_HMDB_ID')
    efflux_mat = efflux_mat.drop(['met','hmdbID','Secondary_HMDB_ID'], axis = 1).groupby('metabolite').max()
    influx_mat = pd.merge(uptake, compass_met_ann[['met', 'hmdbID']],
                            left_index = True, right_on = 'met').dropna()
    influx_mat = pd.merge(influx_mat, met_ann[['Secondary_HMDB_ID', 'metabolite']],
                            left_on = 'hmdbID', right_on = 'Secondary_HMDB_ID')
    influx_mat = influx_mat.drop(['met','hmdbID','Secondary_HMDB_ID'], axis = 1).groupby('metabolite').max()
    return(efflux_mat, influx_mat)
efflux_mat, influx_mat = _get_compass_flux_(sample_path='/Users/rongbinzheng/Documents/BCH/ChenLab/Metabolism/update_2023/flux_2023/compass/avg_exp_compass/human_CRC_GSE146771/')


# In[ ]:


## run mebocost using base model
mebo_obj = mebocost.create_obj(
                        adata = None,
                        group_col = ['Celltype (minor-lineage)'],
                        met_est = 'mebocost',
                        config_path = '/Users/rongbinzheng/Documents/test/MEBOCOST/mebocost.conf',
                        exp_mat=crc_exp.T,
                        cell_ann=crc_meta,
                        species='human',
                        met_pred=None,
                        met_enzyme=None,
                        met_sensor=None,
                        met_ann=None,
                        scFEA_ann=None,
                        compass_met_ann=None,
                        compass_rxn_ann=None,
                        gene_network=None,
                        gmt_path=None,
                        cutoff_exp='auto', ## automated cutoff to exclude lowly ranked 25% sensors across all cells
                        cutoff_met='auto', ## automated cutoff to exclude lowly ranked 25% metabolites across all cells
                        cutoff_prop=0.15, ## at lease 25% of cells should be expressed the sensor or present the metabolite in the cell group (specified by group_col)
                        sensor_type=['Receptor', 'Transporter', 'Nuclear Receptor'],
                        thread=4
                        )
## metabolic communication inference
## Note: by default, this function include estimator for metabolite abundance
commu_res = mebo_obj.infer_commu(
                                n_shuffle=1000,
                                seed=12345, 
                                Return=True, 
                                thread=None,
                                save_permuation=False,
                                min_cell_number = 1
                            )
mebo_obj.save('coloncancer/CRC_GSE146771_Smartseq2.mebocost.pk')


# In[6]:


## update communication result
crc_mebo = mebocost.load_obj('../../../../update_2023/evaluation/crispr_screen_coculture/coloncancer/CRC_GSE146771_Smartseq2.mebocost.pk')
crc_commu_res = crc_mebo.commu_res.copy()

commu_res_new = crc_mebo._filter_lowly_aboundant_(pvalue_res = crc_mebo.original_result.copy(),
                                                    cutoff_prop = 0.2,
                                                    met_prop=crc_mebo.met_prop,
                                                    exp_prop=crc_mebo.exp_prop,
                                                    min_cell_number = 1
                                                 )
## update your commu_res in mebocost object, 
## so that the object can used to generate figure based on the updated data
crc_mebo.commu_res = commu_res_new.copy()
crc_commu_res = crc_mebo.commu_res.copy()
crc_commu_res['Annotation'] = crc_mebo.original_result.sort_values(['Sender', 'Receiver', 'Metabolite', 'Sensor'])['Annotation'].tolist()



# In[7]:


## commu constrained by flux
crc_commu_res['sender_transport_flux'] = [efflux_mat.loc[m,c.replace(' ', '~').replace('/', ':')] if m in efflux_mat.index.tolist() else np.nan for c, m in crc_commu_res[['Sender', 'Metabolite_Name']].values.tolist()]
crc_commu_res['receiver_transport_flux'] = [influx_mat.loc[m,c.replace(' ', '~').replace('/', ':')] if m in influx_mat.index.tolist() else np.nan for c, m in crc_commu_res[['Receiver', 'Metabolite_Name']].values.tolist()]
crc_commu_res['label'] = [' : '.join(sorted(x)) for x in crc_commu_res[['Sender', 'Receiver']].values.tolist()]
## norm flux
flux_norm = lambda x: (x/np.abs(x)) * np.sqrt(np.abs(x)) if x != 0 else 0
x1 = 'sender_transport_flux'
x2 = 'receiver_transport_flux'
crc_commu_res[x1] = [flux_norm(x) for x in crc_commu_res[x1].tolist()]
crc_commu_res[x2] = [flux_norm(x) for x in crc_commu_res[x2].tolist()]
## efflux and influx
tmp1 = crc_commu_res.query('Annotation != "Receptor"').copy()
tmp2 = crc_commu_res.query('Annotation == "Receptor"').copy()
efflux_cut, influx_cut = 0, 0
tmp1 = tmp1[(tmp1[x1]>efflux_cut) & (tmp1[x2]>influx_cut)]
tmp2 = tmp2[(tmp2[x1]>efflux_cut)]
crc_commu_res = pd.concat([tmp1, tmp2])


# In[8]:


## sig mCCC from cancer to NK
sig_comm = crc_commu_res.query('permutation_test_fdr < 0.05 and (Sender == "Malignant" and Receiver == "NK")')
## sig mCCC from NK to cancer
sig_comm2 = crc_commu_res.query('permutation_test_fdr < 0.05 and (Receiver == "Malignant" and Sender == "NK")')
plot_comm = pd.concat([sig_comm, sig_comm2])


# In[11]:


### plot flow for communications
fig = mebocost.CP._FlowPlot_(comm_res=plot_comm, 
                pval_method='permutation_test_fdr',
                pval_cutoff=0.05,
                sender_focus = [],
                metabolite_focus = [],
                sensor_focus = [],
                receiver_focus = [],
                remove_unrelevant = False,
                and_or = 'and',
                node_label_size = 8,
                node_alpha = .8,
                figsize = (10, 5),
                node_cmap = 'Set1',
                line_cmap = 'bwr',
                line_vmin = None,
                line_vmax = 15.5,
                node_size_norm=(20, 150),
                linewidth_norm=(0.5, 5),
                pdf=None, 
                save_plot = False, 
                show_plot = True,
                comm_score_col = 'Commu_Score',
                comm_score_cutoff = None,
                cutoff_prop = None,
                text_outline = False,
                return_fig = True)

fig.savefig('plots/CRC_cancer_nk_flow.pdf')


# ### CRISPR screen data
# 

# In[9]:


## crispr screen data processed by MAGecK downloaded from database
crispr_crc_1657 = pd.read_csv('../../../../update_2023/evaluation/crispr_screen_coculture/BIOGRID-ORCS-SCREEN_1657-1.1.13.screen.tab.txt', sep = '\t')
def quant_norm(df):
    ranks = (df.rank(method="first")
              .stack())
    rank_mean = (df.stack()
                   .groupby(ranks)
                   .mean())
    # Add interpolated values in between ranks
    finer_ranks = ((rank_mean.index+0.5).to_list() +
                    rank_mean.index.to_list())
    rank_mean = rank_mean.reindex(finer_ranks).sort_index().interpolate()
    return (df.rank(method='average')
              .stack()
              .map(rank_mean)
              .unstack())
crispr_crc_1657_qnorm = crispr_crc_1657[['SCORE.1', 'SCORE.3']].copy()
crispr_crc_1657_qnorm = quant_norm(crispr_crc_1657_qnorm)
crispr_crc_1657_qnorm['OFFICIAL_SYMBOL'] = crispr_crc_1657['OFFICIAL_SYMBOL']


# In[10]:


## check sgRNA of enzymes in cancer for cancer to NK mCCC, and sensor in cancer for NK to cancer mCCC
sensors = sig_comm2['Sensor'].unique()
nonsig_sensors = crc_commu_res.query('Sender == "NK" and Receiver == "Malignant"')['Sensor'].unique()
nonsig_sensors = list(np.setdiff1d(nonsig_sensors, sensors))

mets = sig_comm['Metabolite'].unique()
enzymes = []
prod_enzyme = crc_mebo.met_enzyme.copy()
for i,j in prod_enzyme[prod_enzyme['HMDB_ID'].isin(mets)][['HMDB_ID', 'gene']].values.tolist():
    for x in j.split(';'):
        enzymes.append([i, x.split('[')[0]])
enzymes = pd.DataFrame(enzymes, columns = ['met', 'gene'])
 
nonsigmets = crc_commu_res.query('Sender == "Malignant" and Receiver == "NK"')['Metabolite'].unique()
nonsigmets = list(np.setdiff1d(nonsigmets, mets))
nonsig_enzymes = []
for i,j in prod_enzyme[prod_enzyme['HMDB_ID'].isin(nonsigmets)][['HMDB_ID', 'gene']].values.tolist():
    for x in j.split(';'):
        nonsig_enzymes.append([i, x.split('[')[0]])
nonsig_enzymes = pd.DataFrame(nonsig_enzymes, columns = ['met', 'gene'])

plot_df = crispr_crc_1657_qnorm.copy()
label1 = []
for x in plot_df['OFFICIAL_SYMBOL'].tolist():
    if x in pd.Series(sensors).str.replace(' ', '').tolist():
        label1.append('Sig. mCCC')
    elif x in enzymes['gene'].str.replace(' ', '').tolist():
        label1.append('Sig. mCCC')
    else:
        label1.append('other')
plot_df['label1'] = label1
pplot_df = plot_df[['SCORE.1', 'SCORE.3', 'label1', 'OFFICIAL_SYMBOL']].melt(id_vars = ['label1', 'OFFICIAL_SYMBOL'])
pplot_df['variable'] = ['sgRNA enriched' if x == 'SCORE.1' else 'sgRNA depleted' for x in pplot_df['variable'].tolist()]


print(pplot_df.groupby('label1').apply(lambda df: wilcoxon(df.query('variable == "sgRNA enriched"')['value'],
                                                    df.query('variable == "sgRNA depleted"')['value'], alternative = "greater"))
)
print(pplot_df.groupby('variable').apply(lambda df: mannwhitneyu(df.query('label1 != "other"')['value'],
                                                    df.query('label1 == "other"')['value'],
                                                      alternative = 'less' if df['variable'].tolist()[0] == 'sgRNA depleted' else 'greater')))

pplot_df['value'] = -np.log(pplot_df['value'])


# In[12]:


pplot_df.groupby(['label1', 'variable'])['OFFICIAL_SYMBOL'].count()


# In[14]:


## plot boxplot
fig, ax = plt.subplots(figsize = (6, 5))
sns.boxplot(data = pplot_df,
               hue = 'variable', y = 'value', x = 'label1',
            ax = ax, width = .7,
            showfliers = False)
ax.set_ylabel('-Log MAGeCK Score')
ax.set_xlabel('')
ax.tick_params(axis = 'x', rotation = 45)
ax.legend(title = '', loc='center left', bbox_to_anchor=(1, 0.5))
sns.despine()
plt.tight_layout()
fig.savefig('plots/CRC_cancer_nk_crispr.pdf')
plt.show()


# ### survival analysis using TCGA patient data
# 

# In[13]:


from lifelines.statistics import logrank_test
from lifelines import KaplanMeierFitter, CoxPHFitter
def _surv_(m, s, mat, clinic = False): 
    cph = CoxPHFitter()
    if not clinic:
        dat = mat.query('cancer_type == "Tumor"')[[m, s, 'OS.time', 'OS']]#, 'age_at_initial_pathologic_diagnosis', 'I','II','III']]
        dat.columns = ['m', 's', 'OS.time', 'OS']#, 'age_at_initial_pathologic_diagnosis', 'I','II','III']
    else:
        dat = mat.query('cancer_type == "Tumor"')[[m, s, 'OS.time', 'OS', 'age_at_initial_pathologic_diagnosis', 'I','II','III']]
        dat.columns = ['m', 's', 'OS.time', 'OS', 'age_at_initial_pathologic_diagnosis', 'II','III','IV']

    dat['m'] = dat['m'] - np.median(dat['m'])
    dat['s'] = dat['s'] - np.median(dat['s'])
    dat['m*s'] = dat['m']* dat['s']
    try:
        cph.fit(dat, 'OS.time', event_col='OS')
        return({'m':m, 's':s, 'p':cph.summary['p'], 'z':cph.summary['z'], 'hr':cph.hazard_ratios_})
    except:
        return({'m':m, 's':s, 'p':None, 'z':None, 'hr':None})

    


# In[14]:


### data from https://xenabrowser.net/datapages/?cohort=TCGA%20Colon%20and%20Rectal%20Cancer%20(COADREAD)&removeHub=https%3A%2F%2Fxena.treehouse.gi.ucsc.edu%3A443

## test survival
coadread_exp = pd.read_csv('../../../../update_2023/evaluation/crispr_screen_coculture/TCGA/TCGA.COADREAD.sampleMap_HiSeqV2.gz', compression = 'gzip', sep = '\t',
                          index_col = 0)

## aggerate enzyme expression of metabolites from bulk data
coad_bulk_met = mebocost.create_obj(exp_mat=coadread_exp, 
                               cell_ann = pd.DataFrame(coadread_exp.columns, index = coadread_exp.columns,
                                                                              columns = ['cell_type']),
                              group_col = ['cell_type'],
                               species = 'human',
                              config_path = '/Users/rongbinzheng/Documents/test/MEBOCOST/mebocost.conf')
coad_bulk_met._load_config_()
coad_bulk_met.estimator()
coad_bulk_met = pd.DataFrame(coad_bulk_met.met_mat.toarray(),
                            index = coad_bulk_met.met_mat_indexer,
                            columns = coad_bulk_met.met_mat_columns)



# In[15]:


## cat with clinic features
coadread_mat = pd.concat([coadread_exp.T, coad_bulk_met.T], axis = 1)
coadread_mat['cancer_type'] = ['Normal' if int(x.split('-')[-1]) >= 10 else 'Tumor' for x in coadread_mat.index.tolist()]

survival = pd.read_csv('../../../../update_2023/evaluation/crispr_screen_coculture/TCGA/survival_COADREAD_survival.txt', sep = '\t')

coadread_mat = pd.merge(coadread_mat, survival.drop(['Redaction'], axis = 1), left_index = True, right_on = 'sample')
coadread_mat.index = coadread_mat['sample'].tolist()

coad_read_clinic = pd.read_csv('../../../../update_2023/evaluation/crispr_screen_coculture/TCGA/TCGA.COADREAD.sampleMap_COADREAD_clinicalMatrix', sep = '\t')
coadread_mat_clinic = pd.merge(coadread_mat, coad_read_clinic[['age_at_initial_pathologic_diagnosis', 'pathologic_stage', 'sampleID']].dropna(), left_index = True, right_on = 'sampleID')
coadread_mat_clinic.index = coadread_mat_clinic['sampleID'].tolist()
coadread_mat_clinic = coadread_mat_clinic[coadread_mat_clinic['pathologic_stage'] != '[Discrepancy]']
coadread_mat_clinic['pathologic_stage'] = [x.replace('Stage ', '').replace('A', '').replace('B', '').replace('C', '') for x in coadread_mat_clinic['pathologic_stage'].tolist()]
stages = pd.get_dummies(coadread_mat_clinic['pathologic_stage'])
stages = stages.loc[:,~stages.apply(lambda col: np.all(col == 0))]
coadread_mat_clinic = pd.merge(coadread_mat_clinic, stages, left_index = True, right_index = True)


# In[16]:


## survival for all pairs
siv_clinic_res = {}
for mn, m, s in crc_commu_res.drop_duplicates(['Metabolite', 'Sensor'])[['Metabolite_Name', 'Metabolite', 'Sensor']].values.tolist():
    print(1, m, s)
    if m not in coadread_mat_clinic.columns.tolist() or s not in coadread_mat_clinic.columns.tolist():
        continue
    if np.all(coadread_mat_clinic[m] == 0) or np.all(coadread_mat_clinic[s] == 0):
        continue
    siv_clinic_res[mn+'~'+s]=_surv_(m = m, s = s, mat = coadread_mat_clinic, clinic = True)


# In[17]:


sig_ms1 = sig_comm['Metabolite_Name']+'~'+sig_comm['Sensor']
sig_ms2 = sig_comm2['Metabolite_Name']+'~'+sig_comm2['Sensor']

plot_df = pd.DataFrame([[siv_clinic_res[x]['z']['m*s'],
                         siv_clinic_res[x]['p']['m*s'],
                         siv_clinic_res[x]['hr']['m*s']] for x in siv_clinic_res],
                       index = siv_clinic_res, columns = ['z', 'p', 'hr']).sort_values('hr', ascending = False)
labels = []
for x in plot_df.index.tolist():
    if x in sig_ms1.tolist() and x in sig_ms2.tolist():
        labels.append('Sig. mCCC') 
    elif x in sig_ms1.tolist():
        labels.append('Sig. mCCC') 
    elif x in sig_ms2.tolist():
        labels.append('Sig. mCCC') 
    else:
        labels.append('Other')
plot_df['label'] = labels

plot_df = plot_df.reset_index()
plot_df['rank'] = range(0, plot_df.shape[0])
mebocost_surv_res = plot_df.copy()

fig, ax = plt.subplots(figsize = (4,4))
sns.boxplot(x = plot_df['label'], y = np.log2(plot_df['hr']),
            showfliers = False, width = .4, order = ['Sig. mCCC', 'Other'],
           palette={'Sig. mCCC':'red', 'Other':'lightblue'})
ax.set(xlabel = '', ylabel = 'log2(Harzard Ratio)\nSur(Metabolite enzymes × Sensor)')
s1, p1 = ttest_ind(np.log2(plot_df.query('label == "Sig. mCCC"')['hr']),
               np.log2(plot_df.query('label == "Other"')['hr']), alternative = 'greater')
ax.hlines(0, *ax.get_xlim(), linestyle = 'dashed', color = 'grey')
ax.tick_params(axis = 'x', rotation = 90)
ax.set_ylim(-0.85, 1.3)
ax.set_title('p=%.2e'%p1)
sns.despine()
plt.tight_layout()
fig.savefig('plots/CRC_cancer_nk_harzard_ratio_box.pdf')
plt.show()
plt.close()


# In[19]:


plot_df.groupby('label')['index'].count()


# ## CellPhoneDB

# In[10]:


## cellphonedb significant ones
cpdb_res = pd.read_csv('../../../../update_2023/evaluation/crispr_screen_coculture/coloncancer/cellphonedb/statistical_analysis_significant_means_08_17_2023_13_33_25.txt', sep = '\t')
cpdb_res_met = cpdb_res[pd.isna(cpdb_res['gene_a'])]
met_list = [x.split('_')[0] for x in cpdb_res_met['interacting_pair']]

# cp_res_met_genes = []
# for x in cp_res_met['partner_a'].tolist():
cpdb_res_list = []
for i, line in cpdb_res.iterrows():
    ligand = line['interacting_pair'].split('_')[0]
    receptor = line['partner_b'].split(':')[-1] if pd.isna(line['gene_b']) else line['gene_b']
    tmp = line[12:].dropna()
    for cp in tmp.index.tolist():
        sender, receiver = cp.split('|')
        cpdb_res_list.append([sender, receiver, ligand, receptor, line['interacting_pair'], tmp[cp], line['partner_a'], line['partner_b']])
cpdb_res_list = pd.DataFrame(cpdb_res_list, columns = ['Sender', 'Receiver', 'Ligand_Name', 'Sensor', 'interacting_pair', 'Commu_Score', 'Partner_a', 'Partner_b'])
cpdb_res_list['Met'] = cpdb_res_list['Ligand_Name'].isin(met_list)




# In[67]:


## focus on NK and cancer cells
need = ['NK', 'Malignant']
df = cpdb_res_list[(cpdb_res_list['Sender'].isin(need)) & (cpdb_res_list['Receiver'].isin(need))]#.query('Met == True')
genes = []
for i, line in df.iterrows():
    if line['Met'] == True:
        tmp1 = line['Partner_a'].split('_by')[1].split('_and_') if '_by' in line['Partner_a'] else []
        tmp2 = line['Sensor'].split('_')
        genes.extend(tmp1+tmp2)
    else:
        tmp1 = line['Ligand_Name'].split('_')
        tmp2 = line['Sensor'].split('_')
        genes.extend(tmp1+tmp2)
cpdb_genes = np.unique(genes)


# In[68]:


## check genes scores for mCCC genes
sensors = sig_comm2['Sensor'].unique()
nonsig_sensors = crc_commu_res.query('Sender == "NK" and Receiver == "Malignant"')['Sensor'].unique()
nonsig_sensors = list(np.setdiff1d(nonsig_sensors, sensors))

mets = sig_comm['Metabolite'].unique()
enzymes = []
prod_enzyme = crc_mebo.met_enzyme.copy()
for i,j in prod_enzyme[prod_enzyme['HMDB_ID'].isin(mets)][['HMDB_ID', 'gene']].values.tolist():
    for x in j.split(';'):
        enzymes.append([i, x.split('[')[0]])
enzymes = pd.DataFrame(enzymes, columns = ['met', 'gene'])
 
nonsigmets = crc_commu_res.query('Sender == "Malignant" and Receiver == "NK"')['Metabolite'].unique()
nonsigmets = list(np.setdiff1d(nonsigmets, mets))
nonsig_enzymes = []
for i,j in prod_enzyme[prod_enzyme['HMDB_ID'].isin(nonsigmets)][['HMDB_ID', 'gene']].values.tolist():
    for x in j.split(';'):
        nonsig_enzymes.append([i, x.split('[')[0]])
nonsig_enzymes = pd.DataFrame(nonsig_enzymes, columns = ['met', 'gene'])


plot_df = crispr_crc_1657_qnorm.copy()
label1 = []
label2 = []
mebo_genes = pd.Series(sensors).str.replace(' ', '').tolist()+enzymes['gene'].str.replace(' ', '').tolist()
for x in plot_df['OFFICIAL_SYMBOL'].tolist():
    if x in mebo_genes:
        label1.append('MEBOCOST')
    else:
        label1.append('other')
    if x in cpdb_genes:
        label2.append('CellPhoneDB')
    else:
        label2.append('other')
plot_df['label1'] = label1
plot_df['label2'] = label2
pplot_df = plot_df[['SCORE.1', 'SCORE.3', 'label1', 'label2', 'OFFICIAL_SYMBOL']].melt(id_vars = ['label1', 'label2', 'OFFICIAL_SYMBOL'])




# In[60]:


p1 = pplot_df.query('label1 == "MEBOCOST"')
p1['label'] = 'MEBOCOST'
p2 = pplot_df.query('label2 == "CellPhoneDB"')
p2['label'] = 'CellPhoneDB'
p3 = pplot_df.query('label1 == "other" and label2 == "other"')
p3['label'] = 'other'

p_df = pd.concat([p1, p2, p3])
p_df['variable'] = ['sgRNA enriched' if x == 'SCORE.1' else 'sgRNA depleted' for x in p_df['variable'].tolist()]
p_res = p_df.groupby('label').apply(lambda df: ranksums(df.query('variable == "sgRNA enriched"')['value'],
                                                    df.query('variable == "sgRNA depleted"')['value'], alternative = "greater"))
print(p_res)

p_df['value'] = -np.log(p_df['value'])
fig, ax = plt.subplots(figsize = (7, 5))
sns.boxplot(data = p_df,
               hue = 'variable', y = 'value', x = 'label', ax = ax,
            showfliers = False)
ax.set_ylabel('-Log MAGeCK Score')
ax.set_xlabel('')
ax.set_title('MEBOCOST=%.2e, CellPhoneDB=%.2e, other=%.2e'%(p_res['MEBOCOST'][-1], p_res['CellPhoneDB'][-1], p_res['other'][-1]))
ax.tick_params(axis = 'x', rotation = 45)
ax.legend(title = '',
          loc='center left', bbox_to_anchor=(1, 0.5))
sns.despine()
plt.tight_layout()
fig.savefig('plots/CRC_mebocost_cellphoneDB_all.pdf')
plt.show()




# In[43]:


## loading all cellphonedb results to do comparsiosn
cpdb_all_res = pd.read_csv('../../../../update_2023/evaluation/crispr_screen_coculture/coloncancer/cellphonedb/statistical_analysis_means_08_17_2023_13_33_25.txt',
                           sep = '\t')

cpdb_genes = {}
for i, line in cpdb_all_res.iterrows():
    if pd.isna(line['gene_a']):
        tmp1 = line['partner_a'].split('_by')[1].split('_and_') if '_by' in line['partner_a'] else []
        tmp2 = line['partner_b'].split(':')[-1].split('_') if pd.isna(line['gene_b']) else line['gene_b']
    else:
        tmp1 = line['gene_a'].split('_')
        tmp2 = line['partner_b'].split(':')[-1].split('_') if pd.isna(line['gene_b']) else line['gene_b']
    cpdb_genes[line['interacting_pair']] = {'ligand':tmp1, 'receptor':tmp2}


# In[44]:


## survival 
siv_clinic_cpdb_res = {}
for p in cpdb_genes:
    m, s = cpdb_genes[p].values()
    m = np.intersect1d(m, coadread_mat_clinic.columns)
    s = np.intersect1d(s, coadread_mat_clinic.columns)
    if len(m)==0 or len(s)==0:
        continue
    dat = coadread_mat_clinic[['OS.time', 'OS', 'age_at_initial_pathologic_diagnosis', 'I','II','III', 'cancer_type']]
    dat['m'] = coadread_mat_clinic.loc[:,m].T.mean()
    dat['s'] = coadread_mat_clinic.loc[:,s].T.mean()
    if np.all(dat['m'] == 0) or np.all(dat['s'] == 0):
        continue

    siv_clinic_cpdb_res[p]=_surv_(m = 'm', s = 's', mat = dat, clinic = True)


    


# In[89]:


## combine
cpdb_sig_comm = cpdb_res_list.query('(Sender == "Malignant" and Receiver == "NK")')
cpdb_sig_comm2 = cpdb_res_list.query('(Receiver == "Malignant" and Sender == "NK")')

cpdb_sig_ms1 = cpdb_sig_comm['interacting_pair']
cpdb_sig_ms2 = cpdb_sig_comm2['interacting_pair']

plot_df = pd.DataFrame([[siv_clinic_cpdb_res[x]['z']['m*s'],
                         siv_clinic_cpdb_res[x]['p']['m*s'],
                         siv_clinic_cpdb_res[x]['hr']['m*s']] if type(siv_clinic_cpdb_res[x]['z']) != type(None) else [[None, None, None]] for x in siv_clinic_cpdb_res],
                       index = siv_clinic_cpdb_res, columns = ['z', 'p', 'hr']).sort_values('hr', ascending = False).dropna()
labels = []
for x in plot_df.index.tolist():
    if x in cpdb_sig_ms1.tolist():
        labels.append('Sig. CCC') 
    elif x in cpdb_sig_ms2.tolist():
        labels.append('Sig. CCC') 
    else:
        labels.append('Other')
plot_df['label'] = labels
plot_df['met'] = [x in cpdb_res_met['interacting_pair'].tolist() for x in plot_df.index.tolist()]
plot_df = plot_df.reset_index()
plot_df['rank'] = range(0, plot_df.shape[0])

## label for cellphoneDB and MEBOCOST for comparison plot
plot_df['Label'] = ['CellPhoneDB' if x == 'Sig. CCC' else 'Other' for x in plot_df['label'].tolist()]
mebocost_surv_res['Label'] = ['MEBOCOST' if x == 'Sig. mCCC' else 'Other' for x in mebocost_surv_res['label'].tolist()]

plot_df2 = pd.concat([mebocost_surv_res, plot_df])
# plot_df2 = pd.concat([mebocost_surv_res, plot_df])
fig, ax = plt.subplots(figsize = (4,4))
sns.boxplot(x = plot_df2['Label'], y = np.log2(plot_df2['hr']),
            showfliers = False, width = .4, order = ['MEBOCOST', 'CellPhoneDB', 'Other'],
           palette={'MEBOCOST':plt.cm.get_cmap('Set2')(0), 'CellPhoneDB':plt.cm.get_cmap('Set2')(1),
                   'Other':'lightgrey'}, whis = 2)
ax.set(xlabel = '', ylabel = 'log2(Harzard Ratio)\nSur(Metabolite enzymes × Sensor)')
ax.hlines(0, *ax.get_xlim(), linestyle = 'dashed', color = 'grey')
ax.tick_params(axis = 'x', rotation = 90)
ax.set_ylim(-0.85, 1.3)
sns.despine()
plt.tight_layout()
# fig.savefig('plots/CRC_mebocost_cellphoneDB_survival_all.pdf')
plt.show()
plt.close()



# In[90]:


print(ttest_ind(np.log2(plot_df2.query('Label == "MEBOCOST"')['hr']),
               np.log2(plot_df2.query('Label == "Other"')['hr'].dropna()), alternative = 'greater'))


# In[91]:


print(ttest_ind(np.log2(plot_df2.query('Label == "CellPhoneDB"')['hr']),
               np.log2(plot_df2.query('Label == "Other"')['hr'].dropna()), alternative = 'greater'))


# In[92]:


print(ttest_ind(np.log2(plot_df2.query('Label == "MEBOCOST"')['hr']),
               np.log2(plot_df2.query('Label == "CellPhoneDB"')['hr']), alternative = 'greater'))

