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
from scipy.stats import spearmanr, pearsonr, ranksums, wilcoxon, ttest_ind, chisquare, mannwhitneyu
from statsmodels.stats.weightstats import ztest
from matplotlib.backends.backend_pdf import PdfPages
from scipy.cluster.hierarchy import dendrogram, linkage
# from matplotlib import rcParams
# rcParams['font.family'] = 'Arial'

import magic

import sys
sys.path.append('./scripts/')
import importlib
from sklearn.metrics import auc, roc_curve, average_precision_score, precision_recall_curve

import cobra
from cobra.io import read_sbml_model, write_sbml_model

from FBA_CCLE_Scripts import fba_process2

plt.rcParams.update(plt.rcParamsDefault)
rc={"axes.labelsize": 16, "xtick.labelsize": 12, "ytick.labelsize": 12,
    "figure.titleweight":"bold", #"font.size":14,
    "figure.figsize":(5.5,4.2), "font.weight":"regular", "legend.fontsize":10,
    'axes.labelpad':8, 'figure.dpi':300}
plt.rcParams.update(**rc)



# ### COMPASS for flux analysis

# In[2]:


compass_res = {'human_heart_s1':'/Users/rongbinzheng/Documents/BCH/ChenLab/Metabolism/update_2023/flux_2023/compass/avg_exp_compass/human_heart/',
          'human_heart_s2':'/Users/rongbinzheng/Documents/BCH/ChenLab/Metabolism/update_2023/flux_2023/compass/avg_exp_compass/human_heart/',
          'human_intestinal':'/Users/rongbinzheng/Documents/BCH/ChenLab/Metabolism/update_2023/flux_2023/compass/avg_exp_compass/human_intestinal/',
          'human_PDAC':'/Users/rongbinzheng/Documents/BCH/ChenLab/Metabolism/update_2023/flux_2023/compass/avg_exp_compass/human_PDAC/',
          'human_SCC_s1':'//Users/rongbinzheng/Documents/BCH/ChenLab/Metabolism/update_2023/flux_2023/compass/avg_exp_compass/sample_s1/',
          'human_SCC_s2':'/Users/rongbinzheng/Documents/BCH/ChenLab/Metabolism/update_2023/flux_2023/compass/avg_exp_compass/sample_s2/'
        }
compass_met_ann = pd.read_csv('/Users/rongbinzheng/Documents/test/MEBOCOST/data/Compass/met_md.csv')

compass_rxn_ann = pd.read_csv('/Users/rongbinzheng/Documents/test/MEBOCOST/data/Compass/rxn_md.csv')

met_ann = pd.read_csv('/Users/rongbinzheng/Documents/github/MEBOCOST/data/mebocost_db/common/metabolite_annotation_HMDB_summary.tsv',
                     sep = '\t')
alias = {str(i).upper():[str(x).upper() for x in str(j).split('; ')] for i,j in met_ann[['metabolite', 'synonyms_name']].values.tolist()}

human_gem = read_sbml_model('FBA_CCLE_Scripts/data/Human-GEM.xml') 



# #### get co-locolization score

# In[3]:


samples = {'human_heart_s1':'/Users/rongbinzheng/Documents/BCH/ChenLab/Metabolism/update_2023/evaluation/spatial/CCI_datasets/human_heart/sample_s1/',
          'human_heart_s2':'/Users/rongbinzheng/Documents/BCH/ChenLab/Metabolism/update_2023/evaluation/spatial/CCI_datasets/human_heart/sample_s2/',
          'human_intestinal':'/Users/rongbinzheng/Documents/BCH/ChenLab/Metabolism/update_2023/evaluation/spatial/CCI_datasets/human_intestinal/',
          'human_PDAC':'/Users/rongbinzheng/Documents/BCH/ChenLab/Metabolism/update_2023/evaluation/spatial/CCI_datasets/human_paad/PDAC/',
          'human_SCC_s1':'/Users/rongbinzheng/Documents/BCH/ChenLab/Metabolism/update_2023/evaluation/spatial/CCI_datasets/human_scc/sample_s1/',
          'human_SCC_s1':'/Users/rongbinzheng/Documents/BCH/ChenLab/Metabolism/update_2023/evaluation/spatial/CCI_datasets/human_scc/sample_s2/'
          }
evaluation_data = {}
for s in samples:
    coord_path = '%s/st_coord.tsv'%samples[s]
    dev_path = '%s/stride/STRIDE_spot_celltype_frac.txt'%samples[s]
    dev_df=pd.read_csv(dev_path, sep='\t', index_col = 0)
    plot_df = pd.DataFrame(spearmanr(dev_df)[0], index = dev_df.columns, columns = dev_df.columns)

    plot_df = plot_df.where(np.triu(np.ones(plot_df.shape)).astype(np.bool)).unstack().reset_index().dropna()
    plot_df.columns = ['ct1', 'ct2', 'corr']
    plot_df = plot_df[~(plot_df['ct1'] == plot_df['ct2'])]
    plot_df['label'] = [' : '.join(sorted(x)) for x in plot_df[['ct1', 'ct2']].values.tolist()]
    evaluation_data[s] = plot_df

    


# In[4]:


## random cell fraction data
evaluation_data_random = {}
n_rand = 100
np.random.seed(123)
for s in samples:
    coord_path = '%s/st_coord.tsv'%samples[s]
    dev_path = '%s/stride/STRIDE_spot_celltype_frac.txt'%samples[s]
    dev_df=pd.read_csv(dev_path, sep='\t', index_col = 0)
    evaluation_data_random[s] = collections.defaultdict()
    for i in range(n_rand):
        tmp = dev_df.copy()
        for l in tmp.columns.tolist():
            tmp[l] = np.random.random(tmp.shape[0])
    
        plot_df = pd.DataFrame(spearmanr(tmp)[0], index = tmp.columns, columns = tmp.columns) #dev_df.corr()

        plot_df = plot_df.where(np.triu(np.ones(plot_df.shape)).astype(np.bool)).unstack().reset_index().dropna()
        plot_df.columns = ['ct1', 'ct2', 'corr']
        plot_df = plot_df[~(plot_df['ct1'] == plot_df['ct2'])]
        plot_df['label'] = [' : '.join(sorted(x)) for x in plot_df[['ct1', 'ct2']].values.tolist()]
        evaluation_data_random[s][i] = plot_df
        


# #### collect commu res by different cutoff/parameters
# 

# In[5]:


cobra_samples = {'human_heart_s1':'/Users/rongbinzheng/Documents/BCH/ChenLab/Metabolism/update_2023/flux_2023/human_heart_tmp/',
          'human_heart_s2':'/Users/rongbinzheng/Documents/BCH/ChenLab/Metabolism/update_2023/flux_2023/human_heart_tmp/',
          'human_intestinal_A3':'/Users/rongbinzheng/Documents/BCH/ChenLab/Metabolism/update_2023/flux_2023/human_intestinal_tmp/',
          'human_PDAC_B':'/Users/rongbinzheng/Documents/BCH/ChenLab/Metabolism/update_2023/flux_2023/human_PDAC_tmp/',
          'human_SCC_rep2':'/Users/rongbinzheng/Documents/BCH/ChenLab/Metabolism/update_2023/flux_2023/sample_s1_tmp/',
          'human_SCC_rep3':'/Users/rongbinzheng/Documents/BCH/ChenLab/Metabolism/update_2023/flux_2023/sample_s2_tmp/',
          }

sample_foler = {'human_heart_s1':'/Users/rongbinzheng/Documents/BCH/ChenLab/Metabolism/update_2023/evaluation/spatial/CCI_datasets/human_heart/sample_s1/',
          'human_heart_s2':'/Users/rongbinzheng/Documents/BCH/ChenLab/Metabolism/update_2023/evaluation/spatial/CCI_datasets/human_heart/sample_s2/',
          'human_intestinal':'/Users/rongbinzheng/Documents/BCH/ChenLab/Metabolism/update_2023/evaluation/spatial/CCI_datasets/human_intestinal/',
          'human_PDAC':'/Users/rongbinzheng/Documents/BCH/ChenLab/Metabolism/update_2023/evaluation/spatial/CCI_datasets/human_paad/PDAC/',
          'human_SCC_s1':'/Users/rongbinzheng/Documents/BCH/ChenLab/Metabolism/update_2023/evaluation/spatial/CCI_datasets/human_scc/sample_s1/',
          'human_SCC_s1':'/Users/rongbinzheng/Documents/BCH/ChenLab/Metabolism/update_2023/evaluation/spatial/CCI_datasets/human_scc/sample_s2/'
          }

def _cobra_eflux_influx_(comm_res, gem_model, flux_res, alias,
                        subsystem=['Exchange/demand reactions', 'Transport reactions']):
    """
    focus on transport reaction to calculate efflux and influx
    """
#     info('Calculate efflux for sender cells and influx for receiver cells')
    met_reaction_df = []
    for reaction in gem_model.reactions:
        for met in reaction.metabolites:
            met_reaction_df.append([met.id[:-1], met.id, met.name, met.compartment, reaction.id,
                                    reaction.metabolites[met], reaction.compartments, reaction.subsystem])
    met_reaction_df = pd.DataFrame(met_reaction_df,
                                  columns = ['met_id','met_cid', 'met_name',
                                             'met_comp', 'reaction_id', 'direction',
                                             'reaction_comp', 'subsystem'])
    met_reaction_df['reaction_comp'] = ['; '.join(list(x)) for x in met_reaction_df['reaction_comp'].tolist()]
    ## cell surface transport and exchange reaction
    transport_r = met_reaction_df.query('(subsystem == "Transport reactions" and (reaction_comp == "e; c" or reaction_comp == "c; e")) or (subsystem == "Exchange/demand reactions")')
    ## exchange reaction only labels met in extracellular[e], while transport labels two entries for extracellular[e] and cytoplasma[c] 
    exchange_r = transport_r.query('subsystem == "Exchange/demand reactions"')
    exchange_r_tmp = exchange_r.copy()
    exchange_r_tmp['met_comp'] = "c"
    exchange_r_tmp['direction'] = -exchange_r_tmp['direction']
    ## update transport_r dataframe
    transport_r = pd.concat([transport_r, exchange_r_tmp])
    transport_r_flux_d = flux_res.loc[transport_r['reaction_id'],].apply(lambda col: col * transport_r['direction'].tolist())
    transport_r_flux_d.index = transport_r.index.tolist()
    transport_r = pd.concat([transport_r, transport_r_flux_d], axis = 1)
    ## select focused reaction
    transport_r = transport_r[transport_r['subsystem'].isin(subsystem)]
    ## calculate efflux and influx
    efflux = {}
    for m in comm_res['Metabolite_Name'].unique().tolist():
        tr = transport_r.loc[(transport_r['met_name'].str.upper() == m.upper()) | 
                transport_r['met_name'].str.upper().isin(alias[m.upper()]),].query('met_comp == "e"')
        if tr.shape[0] != 0:
            v = tr.drop_duplicates(subset=['reaction_id']).drop(tr.columns[:8].tolist(), axis = 1).max()
            efflux[m] = v

    influx = {}
    for m in comm_res['Metabolite_Name'].unique().tolist():
        tr = transport_r.loc[(transport_r['met_name'].str.upper() == m.upper()) | 
                transport_r['met_name'].str.upper().isin(alias[m.upper()]),].query('met_comp == "c"')
        if tr.shape[0] != 0:
            v = tr.drop_duplicates(subset=['reaction_id']).drop(tr.columns[:8].tolist(), axis = 1).max()
            influx[m] = v
    return(efflux, influx)

def _get_compass_reaction_(sample, subsystem=['Transport, extracellular', 'Exchange/demand reaction']):
    reaction_path = os.path.join(compass_res[sample], 'reactions.tsv')
    reaction = pd.read_csv(reaction_path, index_col = 0, sep = '\t')

    transport_r_compass = []

    for i, line in compass_rxn_ann[compass_rxn_ann['subsystem'].isin(['Transport, extracellular', 'Exchange/demand reaction'])].iterrows():#.query('subsystem == "Transport, extracellular"').iterrows():
        #['subsystem'][compass_rxn_ann['subsystem'].str.contains('Transport')].unique()
        x = line['rxn_formula']
        rxn_id = line['rxn_code_nodirection']
        s = line['subsystem']
        try:
            x = x.replace('\nNo genes', '')
            sub, prod = x.split(' --> ')
            for i in sub.split(' + '):
                i = i.split(' * ')[1].rstrip(']').split(' [')
                c = i[-1]
                m = ' ['.join(i[:-1])
                transport_r_compass.append([m, rxn_id, c, -1, s])
            for i in prod.split(' + '):
                i = i.split(' * ')[1].rstrip(']').split(' [')
                c = i[-1]
                m = ' ['.join(i[:-1])
                transport_r_compass.append([m, rxn_id, c, 1, s])
        except:
            continue
    transport_r_compass = pd.DataFrame(transport_r_compass, columns = ['met_name', 'rxn_id', 'compartment', 'direction', 'subsystem'])
    exchange_r_compass = transport_r_compass.query('subsystem == "Exchange/demand reaction"')
    exchange_r_compass['compartment'] == "c"
    exchange_r_compass['direction'] = -exchange_r_compass['direction']
    transport_r_compass = pd.concat([transport_r_compass, exchange_r_compass])
    transport_r_compass = transport_r_compass[transport_r_compass['subsystem'].isin(subsystem)]
    transport_r_compass = pd.merge(transport_r_compass, compass_met_ann[['metName', 'hmdbID']].drop_duplicates(), 
                                  left_on = 'met_name', right_on = 'metName', how = 'left')
    transport_r_compass = pd.merge(transport_r_compass, met_ann[['HMDB_ID','Secondary_HMDB_ID', 'metabolite']], 
                                   left_on = 'hmdbID', right_on = 'Secondary_HMDB_ID', how = 'left')
    transport_r_compass = transport_r_compass.drop(['metName', 'hmdbID', 'Secondary_HMDB_ID'], axis = 1)
    
    transport_r_compass_values = {}
    for r in transport_r_compass['rxn_id'].unique().tolist():
        if r+'_neg' in reaction.index.tolist() and r+'_pos' in reaction.index.tolist():
            v = reaction.loc[r+'_pos'] - reaction.loc[r+'_neg']
        elif r+'_pos' in reaction.index.tolist():
            v = reaction.loc[r+'_pos']
        elif r+'_neg' in reaction.index.tolist():
            v = reaction.loc[r+'_neg'] 
            ## this is usually for extraceullar metabolite in exchange reaction, 
            ## so extracellular met as left side in reaction, do not take minus to keep extracellualar met in the right
        else:
            continue
        transport_r_compass_values[r] = v #pd.Series(v.tolist(), index = v.index)
    transport_r_compass_values = pd.DataFrame(transport_r_compass_values) 

    transport_r_compass = pd.merge(transport_r_compass, transport_r_compass_values.T,
            left_on = 'rxn_id', right_index = True, how = 'left').dropna()
    for x in transport_r_compass_values.index.tolist():
        transport_r_compass[x] = transport_r_compass['direction'] * transport_r_compass[x]
    
    ## add flux to communication table
    sender_transport_flux = {}
    for m in transport_r_compass['metabolite'].unique().tolist():
        tr = transport_r_compass.loc[(transport_r_compass['metabolite'] == m),].query('compartment == "e"')
        if tr.shape[0] != 0:
            v = tr.drop_duplicates(subset=['rxn_id']).drop(tr.columns[:6].tolist(), axis = 1).max()
    #         v.index = [c.replace(' ', '~').replace('/', ':') for c in v.index.tolist()]
            sender_transport_flux[m] = v

    receiver_transport_flux = {}
    for m in transport_r_compass['metabolite'].unique().tolist():
        tr = transport_r_compass.loc[(transport_r_compass['metabolite'] == m),].query('compartment == "c"')
        if tr.shape[0] != 0:
            v = tr.drop_duplicates(subset=['rxn_id']).drop(tr.columns[:6].tolist(), axis = 1).max()
    #         v.index = [c.replace(' ', '~').replace('/', ':') for c in v.index.tolist()]
            receiver_transport_flux[m] = v

    return(sender_transport_flux, receiver_transport_flux)

def _get_commu_res_from_react_(cutoff_prop=0, level_cutoff=0, 
#                                subsystem = ['Transport, extracellular', 'Exchange/demand reaction']
                               subsystem = ['Exchange/demand reaction']
                              ):
    res_pack_tmp = {}
    x1 = 'sender_transport_flux'
    x2 = 'receiver_transport_flux'
    for sample in sample_foler:
        mebo_path = '%s/mebocost_res.pk'%sample_foler[sample]
        mebo_obj = mebocost.load_obj(mebo_path)
        if str(level_cutoff) == 'auto':
            exp_prop, met_prop =  mebo_obj.exp_prop, mebo_obj.met_prop
        else:
            exp_prop, met_prop = mebo_obj._check_aboundance_(cutoff_exp = level_cutoff,
                                                       cutoff_met = level_cutoff)

        comm_res = mebo_obj._filter_lowly_aboundant_(pvalue_res = mebo_obj.original_result.copy(),
                                                        cutoff_prop = cutoff_prop,
                                                        met_prop=met_prop,
                                                        exp_prop=exp_prop,
                                                        min_cell_number = 1
                                                     )
        comm_res = comm_res.sort_values(['Sender', 'Receiver', 'Metabolite', 'Sensor'])
        comm_res['Annotation'] = mebo_obj.original_result.sort_values(['Sender', 'Receiver', 'Metabolite', 'Sensor'])['Annotation'].tolist()
        comm_res['sig'] = (comm_res['permutation_test_fdr'] < 0.05) #& (comm_res['metabolite_prop_in_sender'] > 0.2) & (comm_res['sensor_prop_in_receiver'] > 0.2)
        avg_met_celltype = pd.DataFrame(mebo_obj.avg_met.toarray(),index = mebo_obj.avg_met_indexer,
                               columns = mebo_obj.avg_met_columns)
        avg_sensor_celltype = pd.DataFrame(mebo_obj.avg_exp.toarray(),index = mebo_obj.avg_exp_indexer,
                                       columns = mebo_obj.avg_exp_columns)
        comm_res['met_in_sender'] = [avg_met_celltype.loc[m, c] for c, m in comm_res[['Sender', 'Metabolite']].values.tolist()]
        comm_res['sensor_in_receiver'] = [avg_sensor_celltype.loc[m, c] for c, m in comm_res[['Receiver', 'Sensor']].values.tolist()]

        sender_transport_flux, receiver_transport_flux = _get_compass_reaction_(sample, subsystem = subsystem)
        

        comm_res['sender_transport_flux'] = [sender_transport_flux[m][c.replace(' ', '~').replace('/', ':')] if m in sender_transport_flux else np.nan for c, m in comm_res[['Sender', 'Metabolite_Name']].values.tolist()]
        comm_res['receiver_transport_flux'] = [receiver_transport_flux[m][c.replace(' ', '~').replace('/', ':')] if m in receiver_transport_flux else np.nan for c, m in comm_res[['Receiver', 'Metabolite_Name']].values.tolist()]
        comm_res['label'] = [' : '.join(sorted(x)) for x in comm_res[['Sender', 'Receiver']].values.tolist()]
        coloc_corr = {}
        for i, j in evaluation_data[sample].iterrows():
            coloc_corr[j['label']] = j['corr']
        comm_res[sample] = [coloc_corr.get(x, np.nan) for x in comm_res['label'].tolist()]
        ## norm flux
        x1 = 'sender_transport_flux'
        x2 = 'receiver_transport_flux'
        flux_norm = lambda x: (x/np.abs(x)) * np.sqrt(np.abs(x)) if x != 0 else 0
        # comm_res[x1] = ((comm_res[x1]/comm_res[x1].abs()) * np.sqrt(np.abs(comm_res[x1])))#.fillna(0)
        # comm_res[x2] = ((comm_res[x2]/comm_res[x2].abs()) * np.sqrt(np.abs(comm_res[x2])))#.fillna(0)
        comm_res[x1] = [flux_norm(x) for x in comm_res[x1].tolist()]
        comm_res[x2] = [flux_norm(x) for x in comm_res[x2].tolist()]
        res_pack_tmp[sample] = comm_res
    return(res_pack_tmp)


def _get_compass_flux_(sample):  
    uptake_path = os.path.join(compass_res[sample], 'uptake.tsv')
    secret_path = os.path.join(compass_res[sample], 'secretions.tsv')
    reaction_path = os.path.join(compass_res[sample], 'reactions.tsv')

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

def _get_commu_res_(cutoff_prop=0, level_cutoff=0, suffix = 'mebocost_res.pk', folders = sample_foler):
    res_pack_tmp = {}
    x1 = 'sender_transport_flux'
    x2 = 'receiver_transport_flux'
    for sample in folders:
        mebo_path = '%s/%s'%(sample_foler[sample], suffix)
        print(mebo_path)
        mebo_obj = mebocost.load_obj(mebo_path)
        if str(level_cutoff) == 'auto':
            exp_prop, met_prop =  mebo_obj.exp_prop, mebo_obj.met_prop
        else:
            exp_prop, met_prop = mebo_obj._check_aboundance_(cutoff_exp = level_cutoff,
                                                       cutoff_met = level_cutoff)

        comm_res = mebo_obj._filter_lowly_aboundant_(pvalue_res = mebo_obj.original_result.copy(),
                                                        cutoff_prop = cutoff_prop,
                                                        met_prop=met_prop,
                                                        exp_prop=exp_prop,
                                                        min_cell_number = 1
                                                     )
        comm_res = comm_res.sort_values(['Sender', 'Receiver', 'Metabolite', 'Sensor'])
        comm_res['Annotation'] = mebo_obj.original_result.sort_values(['Sender', 'Receiver', 'Metabolite', 'Sensor'])['Annotation'].tolist()
        comm_res['sig'] = (comm_res['permutation_test_fdr'] < 0.05) #& (comm_res['metabolite_prop_in_sender'] > 0.2) & (comm_res['sensor_prop_in_receiver'] > 0.2)

        avg_met_celltype = pd.DataFrame(mebo_obj.avg_met.toarray(),index = mebo_obj.avg_met_indexer,
                               columns = mebo_obj.avg_met_columns)
        avg_sensor_celltype = pd.DataFrame(mebo_obj.avg_exp.toarray(),index = mebo_obj.avg_exp_indexer,
                                       columns = mebo_obj.avg_exp_columns)
        comm_res['met_in_sender'] = [avg_met_celltype.loc[m, c] for c, m in comm_res[['Sender', 'Metabolite']].values.tolist()]
        comm_res['sensor_in_receiver'] = [avg_sensor_celltype.loc[m, c] for c, m in comm_res[['Receiver', 'Sensor']].values.tolist()]
        ## compass flux
        efflux_mat, influx_mat = _get_compass_flux_(sample = sample)
        comm_res['sender_transport_flux'] = [efflux_mat.loc[m,c.replace(' ', '~').replace('/', ':')] if m in efflux_mat.index.tolist() else np.nan for c, m in comm_res[['Sender', 'Metabolite_Name']].values.tolist()]
        comm_res['receiver_transport_flux'] = [influx_mat.loc[m,c.replace(' ', '~').replace('/', ':')] if m in influx_mat.index.tolist() else np.nan for c, m in comm_res[['Receiver', 'Metabolite_Name']].values.tolist()]
        ## cobra flux
        avg_flux_res = pd.DataFrame()
        tmp = cobra_samples[sample]
        for s in os.listdir(tmp):
            if s.startswith('.'):
                continue
            print(tmp+'/'+s)
            ss = pd.read_csv(tmp+'/'+s, index_col = 0)
            ss.columns = [s.replace('.csv', '')]
            avg_flux_res = pd.concat([avg_flux_res, ss], axis = 1)
        
        cobra_efflux_mat, cobra_influx_mat = _cobra_eflux_influx_(comm_res=comm_res, 
                                                                  gem_model=human_gem, 
                                                                  flux_res=avg_flux_res, alias=alias,
                                                                 subsystem=['Exchange/demand reactions', 'Transport reactions'])
        
        comm_res['cobra_sender_transport_flux'] = [cobra_efflux_mat[m][c.replace(' ', '~').replace('/', ':')] if m in list(cobra_efflux_mat.keys()) else np.nan for c, m in comm_res[['Sender', 'Metabolite_Name']].values.tolist()]
        comm_res['cobra_receiver_transport_flux'] = [cobra_influx_mat[m][c.replace(' ', '~').replace('/', ':')] if m in list(cobra_influx_mat.keys()) else np.nan for c, m in comm_res[['Receiver', 'Metabolite_Name']].values.tolist()]

        comm_res['label'] = [' : '.join(sorted(x)) for x in comm_res[['Sender', 'Receiver']].values.tolist()]
        coloc_corr = {}
        for i, j in evaluation_data[sample].iterrows():
            coloc_corr[j['label']] = j['corr']
        comm_res[sample] = [coloc_corr.get(x, np.nan) for x in comm_res['label'].tolist()]
        ## norm flux
        flux_norm = lambda x: (x/np.abs(x)) * np.sqrt(np.abs(x)) if x != 0 else 0
        comm_res[x1] = [flux_norm(x) for x in comm_res[x1].tolist()]
        comm_res[x2] = [flux_norm(x) for x in comm_res[x2].tolist()] 
        res_pack_tmp[sample] = comm_res
    return(res_pack_tmp)



# In[6]:


# res_pack = collections.defaultdict()
# res_pack['cutoff_prop=%s, level_cutoff=%s'%(0, 0)] = _get_commu_res_(cutoff_prop=0, level_cutoff=0)
# out = open('data/res_pack_20240205_with_mousecortex.pk', 'wb')
# pk.dump(res_pack, out)
# out.close()

res_pack = pk.load(open('data/res_pack_20240205_with_mousecortex.pk', 'rb'))
res_pack.keys()


# In[7]:


### determine the thresthold
efflux_rates = []
for x in res_pack['cutoff_prop=0, level_cutoff=0']:
    if x.startswith('human') and 'human_sample_s8' in res_pack:
        continue
    l = res_pack['cutoff_prop=0, level_cutoff=0'][x].drop_duplicates(['Sender', 'Metabolite_Name'])['sender_transport_flux']
    efflux_rates.extend(l)


influx_rates = []
for x in res_pack['cutoff_prop=0, level_cutoff=0']:
    if x.startswith('human') and 'human_sample_s8' in res_pack:
        continue
    l = res_pack['cutoff_prop=0, level_cutoff=0'][x].drop_duplicates(['Receiver', 'Sensor'])['receiver_transport_flux']
    influx_rates.extend(l)

## determine efflux and influx threshold 
print('efflux cutoff: ', np.percentile(efflux_rates, 25))
print('efflux cutoff: ', np.percentile(influx_rates, 25))


# In[27]:


print('near vs bg: ',  ranksums(near_list, corr_rand_list, alternative = 'greater'))
print('far vs bg: ', ranksums(far_list, corr_rand_list, alternative = 'less'))


# ## correlation analysis
# 

# In[7]:


## for each metabolie 
x1 = 'sender_transport_flux'
x2 = 'receiver_transport_flux'
cutoff1, cutoff2 = 0.1, -0.4 ## define near and far cell types
efflux_cut, influx_cut = 10, 2
mbcorr_res = {}
mcorr_res={}
mcorr_res2={}
mcorr_res3={}

for x in res_pack:
    res_pack_tmp = res_pack[x].copy()
    mbcorr_res[x] = {'num_sensor':{}}
    mcorr_res[x] = {'num_sensor':{}}
    mcorr_res2[x] = {'num_sensor':{}}
    mcorr_res3[x] = {'num_sensor':{}}
    for sample in res_pack_tmp:
#         if sample == 'mouse_cortex':
#             continue
        comm_res = res_pack_tmp[sample].copy()
        comm_res = comm_res[(comm_res['Sender'] != comm_res['Receiver'])].query('sig == True')
        ## base
        mbcorr_res[x][sample] = comm_res.groupby('Metabolite_Name').apply(lambda df: pd.Series(spearmanr(df[['Commu_Score', sample, 'label']].dropna().groupby('label').median().dropna()),
                                                                                     index = ['R', 'p']))
        mbcorr_res[x][sample]['cellpair_num']=comm_res.groupby('Metabolite_Name').apply(lambda df: df.drop_duplicates('label').shape[0]).tolist()
        mbcorr_res[x][sample]['comm_num']=comm_res.groupby('Metabolite_Name').apply(lambda df: df.shape[0]).tolist()
        mbcorr_res[x]['num_sensor'][sample] = np.unique(comm_res['Sensor'].dropna()).shape[0]

        ## base_efflux_influx_cut
        tmp1 = comm_res.query('Annotation != "Receptor"').copy()
        tmp2 = comm_res.query('Annotation == "Receptor"').copy()
        tmp1 = tmp1[(tmp1[x1]>efflux_cut) & (tmp1[x2]>influx_cut)]
        tmp2 = tmp2[(tmp2[x1]>efflux_cut)]
        tmp = pd.concat([tmp1, tmp2])
        mcorr_res[x][sample] = tmp.groupby('Metabolite_Name').apply(lambda df: pd.Series(spearmanr(df[['Commu_Score', sample, 'label']].dropna().groupby('label').median().dropna()),
                                                                                     index = ['R', 'p']))
        mcorr_res[x][sample]['cellpair_num']=tmp.groupby('Metabolite_Name').apply(lambda df: df.drop_duplicates('label').shape[0]).tolist()
        mcorr_res[x][sample]['comm_num']=tmp.groupby('Metabolite_Name').apply(lambda df: df.shape[0]).tolist()
        mcorr_res[x]['num_sensor'][sample] = np.unique(tmp['Sensor'].dropna()).shape[0]


# In[40]:


res_pack['cutoff_prop=0, level_cutoff=0'].keys()


# In[9]:


## random
## for each metabolie -- base_efflux_influx_CommScore
x1 = 'sender_transport_flux'
x2 = 'receiver_transport_flux'
efflux_cut, influx_cut = 10, 2
cutoff1, cutoff2 = 0.1, -0.4
n_rand = 100
mcorr_res_random = {}
res_pack_tmp = res_pack['cutoff_prop=%s, level_cutoff=%s'%(0, 0)]
for sample in res_pack_tmp:
    comm_res = res_pack_tmp[sample].copy()
    ## remove autocrine since spatial info is not meaningful for same cell type
    comm_res = comm_res[(comm_res['Sender'] != comm_res['Receiver'])]
    mcorr_res_random[sample] = collections.defaultdict()
    for n in range(n_rand):
        coloc_corr = {}
        for i, j in evaluation_data_random[sample][n].iterrows():
            coloc_corr[j['label']] = j['corr']
        comm_res[sample] = [coloc_corr.get(x, np.nan) for x in comm_res['label'].tolist()]

        ## base_efflux_influx
        tmp1 = comm_res.query('sig == True and Annotation != "Receptor"').copy()
        tmp2 = comm_res.query('sig == True and Annotation == "Receptor"').copy()
        tmp1 = tmp1[(tmp1[x1]>efflux_cut) & (tmp1[x2]>influx_cut)]
        tmp2 = tmp2[(tmp2[x1]>efflux_cut)]
        tmp = pd.concat([tmp1, tmp2])
        mcorr_res_random[sample][n] = tmp.groupby('Metabolite_Name').apply(lambda df: pd.Series(spearmanr(df[['Commu_Score', sample, 'label']].dropna().groupby('label').median().dropna()),
                                                                                     index = ['R', 'p']))
        mcorr_res_random[sample][n]['cellpair_num']=tmp.groupby('Metabolite_Name').apply(lambda df: df.shape[0]).tolist()

sample_order = ['human_heart_s8', 'human_heart_s10',
       'human_intestinal_A3', 'human_PDAC_B','human_SCC_rep2',
       'human_SCC_rep3']
## plot real and random
x = 'cutoff_prop=0, level_cutoff=0'
mcorr_res_df = pd.DataFrame()
for s in mcorr_res[x]:
    if s == 'num_sensor':
        continue
    ttmp = mcorr_res[x][s].copy()
    ttmp['sample'] = s
    mcorr_res_df = pd.concat([mcorr_res_df, ttmp])
    
plot_df = mcorr_res_df.copy()
plot_df.columns= ['Real data','p','cellpair_num','comm_num','sample']
plot_df['Random data'] = [np.mean([x.loc[i]['R'] for x in list(mcorr_res_random[l['sample']].values())]) for i, l in mcorr_res_df.iterrows()]
fig, ax = plt.subplots(figsize = (6,5.5))
## require at least 10 cell pairs to get a relatively reliable correlation result
sns.boxplot(data = plot_df.query('cellpair_num>10')[['Real data', 'Random data', 'sample']].melt(id_vars=['sample']),
               x = 'sample', y = 'value', hue = 'variable', order = sample_order,
               palette={'Real data':'darkorange', 'Random data':'lightgrey'}, showfliers = False)
ax.hlines(0, *ax.get_xlim(), linestyle = 'dashed', color = 'grey')
ax.tick_params(axis = 'x', rotation = 90)
ax.set(xlabel='', ylabel='Spearman R')
ax.legend(title = '', loc='upper left', bbox_to_anchor=(1, 1))
sns.despine()
plt.tight_layout()
# fig.savefig('plots/real_random_corr_compare.pdf')
plt.show()


# In[11]:


plot_df.query('cellpair_num>10').groupby('sample')['sample'].count()


# In[14]:


plot_df.query('cellpair_num>10').groupby('sample').apply(lambda df: ranksums(df['Real data'],
                                                                              df['Random data'], alternative = 'greater'))


# In[15]:


### only flux to define sender and receiver
## for each metabolie -- efflux_influx
x1 = 'sender_transport_flux'
x2 = 'receiver_transport_flux'
flux_cut = 0
cutoff1, cutoff2 = 0.1, -0.4
mfcorr_res = {}
res_pack_tmp = res_pack['cutoff_prop=%s, level_cutoff=%s'%(0, 0)]
for sample in res_pack_tmp:
    comm_res = res_pack_tmp[sample].copy()
    ## remove autocrine since spatial info is not meaningful for same cell type
    comm_res = comm_res[(comm_res['Sender'] != comm_res['Receiver'])]
    tmp1 = comm_res.query('Annotation != "Receptor"').copy()
    tmp2 = comm_res.query('Annotation == "Receptor"').copy()
    combine_score = lambda x, xx: ((x*xx)/np.abs(x*xx)) * np.sqrt(np.abs(x*xx))
    tmp1['value'] = combine_score(tmp1[x1], tmp1[x2]) * tmp1['sensor_in_receiver']
    tmp2['value'] = tmp2[x1] * tmp2['sensor_in_receiver']
    tmp = pd.concat([tmp1, tmp2])
    mfcorr_res[sample] = {}
    ## try different cutoffs
    for p in [60, 65, 70, 75, 80, 85, 90, 95]:
        tmp_new = tmp[(tmp['value'] > np.percentile(tmp['value'].dropna(), p))]
        mfcorr_res[sample][p] = tmp_new.groupby(['Metabolite_Name']).apply(lambda df: pd.Series(spearmanr(df[['Commu_Score', sample]].dropna()),
                                                                                     index = ['R', 'p']))
        mfcorr_res[sample][p]['cellpair_num']=tmp_new.groupby(['Metabolite_Name']).apply(lambda df: df.drop_duplicates('label').shape[0]).tolist()
        mfcorr_res[sample][p]['comm_num']=tmp_new.groupby(['Metabolite_Name']).apply(lambda df: df.shape[0]).tolist()



# In[18]:


## show the performance for flux only, base mebocost, base+flux
x = 'cutoff_prop=0, level_cutoff=0'
mcorr_res_df = pd.DataFrame()
for s in mcorr_res[x]:
    if s == 'num_sensor':
        continue
    ttmp = mcorr_res[x][s].copy()
    ttmp['sample'] = s
    mcorr_res_df = pd.concat([mcorr_res_df, ttmp])
mbcorr_res_df = pd.DataFrame()
for s in mbcorr_res[x]:
    if s == 'num_sensor':
        continue
    ttmp = mbcorr_res[x][s].copy()
    ttmp['sample'] = s
    mbcorr_res_df = pd.concat([mbcorr_res_df, ttmp])
    
mbcorr_res_df['method'] = 'base'
mcorr_res_df['method'] = 'base+flux'

mfcorr_res_df = pd.DataFrame()
for s in mfcorr_res:
    if s == 'num_sensor':
        continue
    ttmp = mfcorr_res[s][80].copy()
    ttmp['sample'] = s
    mfcorr_res_df = pd.concat([mfcorr_res_df, ttmp])

mbcorr_res_df['method'] = 'base'
mcorr_res_df['method'] = 'base+flux'
mfcorr_res_df['method'] = 'flux'

plot_df = pd.concat([mfcorr_res_df, mbcorr_res_df, mcorr_res_df])
fig, ax = plt.subplots(figsize = (7, 5.5))
sns.violinplot(data = plot_df.query('cellpair_num > 10'),
               x = 'sample', y = 'R', hue = 'method',
               order = sample_order, palette = 'tab20', scale='width',
#                color = 'darkorange',
               cut = 0)
# sns.boxplot(data = plot_df,
#                x = 'sample', y = 'R', hue = 'method',
#                order = sample_need_new, palette = 'tab20')
ax.hlines(0, *ax.get_xlim(), linestyle = 'dashed', color = 'grey')
ax.tick_params(axis = 'x', rotation = 90)
ax.legend(loc='upper left', bbox_to_anchor=(1, 1), title = 'method')
ax.set(xlabel='', ylabel='Spearmzan R')
sns.despine()
plt.tight_layout()
fig.savefig('plots/flux_base_baseflux_compare.pdf')

plt.show()




# ### cellphoneDB

# In[8]:


## cellphone db
cpdb_path =  {'human_heart_s1':'/Users/rongbinzheng/Documents/BCH/ChenLab/Metabolism/update_2023/evaluation/spatial/cellphonedb/cellphonedb_res/cellphonedb_output_human_heart/statistical_analysis_significant_means_04_14_2023_16:00:12.txt',
          'human_heart_s2':'/Users/rongbinzheng/Documents/BCH/ChenLab/Metabolism/update_2023/evaluation/spatial/cellphonedb/cellphonedb_res/cellphonedb_output_human_heart/statistical_analysis_significant_means_04_14_2023_16:00:12.txt',
          'human_intestinal':'/Users/rongbinzheng/Documents/BCH/ChenLab/Metabolism/update_2023/evaluation/spatial/cellphonedb/cellphonedb_res/cellphonedb_output_human_intestinal/statistical_analysis_significant_means_04_14_2023_17:08:57.txt',
          'human_PDAC':'/Users/rongbinzheng/Documents/BCH/ChenLab/Metabolism/update_2023/evaluation/spatial/cellphonedb/cellphonedb_res/cellphonedb_output_human_PDAC/statistical_analysis_significant_means_05_10_2023_17:07:52.txt',
          'human_SCC_s1':'//Users/rongbinzheng/Documents/BCH/ChenLab/Metabolism/update_2023/evaluation/spatial/cellphonedb/cellphonedb_res/cellphonedb_output_human_scc_s1/statistical_analysis_significant_means_05_20_2023_14:35:43.txt',
          'human_SCC_s2':'/Users/rongbinzheng/Documents/BCH/ChenLab/Metabolism/update_2023/evaluation/spatial/cellphonedb/cellphonedb_res/cellphonedb_output_human_scc_s2/statistical_analysis_significant_means_05_10_2023_17:12:29.txt'
             }

## each ligand
mcorrres_cp = {}
numres_cp = {}
numsensor_cp = {}
cutoff1, cutoff2 = 0.1, -0.4
for sample in cpdb_path:
    cpdb_res = pd.read_csv(cpdb_path[sample], sep = '\t')
    cpdb_res_met = cpdb_res[pd.isna(cpdb_res['gene_a'])]
    numres_cp[sample] = pd.Series([cpdb_res.shape[0], cpdb_res_met.shape[0]], index = ['All', 'Met'])
    met_list = [x.split('_')[0] for x in cpdb_res_met['interacting_pair']]
    cpdb_res_list = []
    for i, line in cpdb_res.iterrows():
        ligand = line['interacting_pair'].split('_')[0]
        receptor = line['gene_b']
        tmp = line[12:].dropna()
        for cp in tmp.index.tolist():
            sender, receiver = cp.split('|')
            cpdb_res_list.append([sender, receiver, ligand, receptor, tmp[cp]])
    cpdb_res_list = pd.DataFrame(cpdb_res_list, columns = ['Sender', 'Receiver', 'Ligand_Name', 'Sensor', 'Commu_Score'])
    cpdb_res_list['metabolite'] = [x in met_list for x in cpdb_res_list['Ligand_Name'].tolist()]
    numsensor_cp[sample]=cpdb_res_list.groupby('metabolite')['Sensor'].apply(lambda x: np.unique(x.dropna()).shape[0])
    cpdb_res_list['label'] = [' : '.join(sorted(x)) for x in cpdb_res_list[['Sender', 'Receiver']].values.tolist()]
    coloc_corr = {}
    for i, j in evaluation_data[sample].iterrows():
        coloc_corr[j['label']] = j['corr']
    cpdb_res_list[sample] = [coloc_corr.get(x, np.nan) for x in cpdb_res_list['label'].tolist()]
    mcorrres_cp[sample] = cpdb_res_list.groupby('Ligand_Name').apply(lambda df: pd.Series(spearmanr(df[['Commu_Score', sample, 'label']].dropna().groupby('label').median().dropna()),
                                                                                 index = ['R', 'p']))
    mcorrres_cp[sample]['cellpair_num']=cpdb_res_list.groupby('Ligand_Name').apply(lambda df: df.drop_duplicates('label').shape[0]).tolist()
    mcorrres_cp[sample]['comm_num']=cpdb_res_list.groupby('Ligand_Name').apply(lambda df: df.shape[0]).tolist()
    mcorrres_cp[sample]['metabolite'] = [x in met_list for x in mcorrres_cp[sample].index.tolist()]

mcorrres_cp_df = pd.DataFrame()
for s in mcorrres_cp:
    tmp = mcorrres_cp[s]#.query('cellpair_num>10').copy()
    tmp['sample'] = s
    mcorrres_cp_df = pd.concat([mcorrres_cp_df, tmp])
mcorrres_cp_df.groupby('sample').median()
numres_cp_df = pd.DataFrame(numres_cp).T


# In[27]:


## show the comparisone between cellphonedb and mebocost
plot_df1 = mcorr_res_df.copy()
plot_df1['method'] = 'MEBOCOST'
plot_df2 = mcorrres_cp_df.copy() ## ligands in cellphoneDB for performance
plot_df2['method'] = 'CellPhoneDB'

plot_df = pd.concat([plot_df1, plot_df2])

fig, ax = plt.subplots(figsize = (6,5))

sns.boxplot(data = plot_df.query('cellpair_num>10'),
               x = 'sample', y = 'R', order = sample_order,
               hue = 'method', palette = 'Set2', showfliers = False)

ax.hlines(0, *ax.get_xlim(), linestyle = 'dashed', color = 'grey')
ax.tick_params(axis = 'x', rotation = 90)
ax.set(xlabel='', ylabel='Spearman R')
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
sns.despine()
plt.tight_layout()
# fig.savefig('plots/mebocost_cellphonedb_compare.pdf')
plt.show()


# In[61]:


plot_df[plot_df['sample'].isin(sample_order)].query('cellpair_num > 10').groupby('sample').apply(lambda df: ranksums(df.query('method == "MEBOCOST"')['R'].dropna(),
                                                  df.query('method == "CellPhoneDB"')['R'].dropna(),
                                                   alternative = 'greater'))



# In[50]:


## plot total num of comm and num of met
num1 = mcorr_res_df.groupby('sample')['comm_num'].sum()
num2 = mcorrres_cp_df.query('metabolite == True').groupby('sample')['comm_num'].sum()
# num3 = mcorrres_neu_df.query('metabolite == True').groupby('sample')['comm_num'].sum()

plot_df = pd.concat([num1, num2], axis = 1)
plot_df.columns = ['MEBOCOST', 'CellPhoneDB']
plot_df_df = plot_df.reset_index().melt(id_vars = ['sample'])

fig, ax = plt.subplots(figsize = (6.5, 5))
sb = sns.barplot(data = plot_df_df, x = 'sample', y = 'value', hue = 'variable',
           palette = 'Set2', order = sample_order)
for i in sb.containers:
    sb.bar_label(i,)
ax.tick_params(axis = 'x', rotation = 90)
ax.set(xlabel = '', ylabel = '# of communications')
ax.legend(title = '', loc='upper left', bbox_to_anchor=(1, 1))
sns.despine()
plt.tight_layout()
# fig.savefig('plots/comm_num_mebo_cpdb.pdf')
plt.show()

## num of met
num1 = mcorr_res_df.groupby('sample')['comm_num'].count()
num2 = mcorrres_cp_df.query('metabolite == True').groupby('sample')['comm_num'].count()
# num3 = mcorrres_neu_df.query('metabolite == True').groupby('sample')['comm_num'].count()

plot_df = pd.concat([num1, num2], axis = 1)
plot_df.columns = ['MEBOCOST', 'CellPhoneDB']
plot_df_df = plot_df.reset_index().melt(id_vars = ['sample'])

fig, ax = plt.subplots(figsize = (6.5, 5))
sb = sns.barplot(data = plot_df_df, x = 'sample', y = 'value', hue = 'variable',
           palette = 'Set2', order = sample_order)
for i in sb.containers:
    sb.bar_label(i,)
ax.tick_params(axis = 'x', rotation = 90)
ax.set(xlabel = '', ylabel = '# of metabolites')
ax.legend(title = '', loc='upper left', bbox_to_anchor=(1, 1))
sns.despine()
plt.tight_layout()
# fig.savefig('plots/met_num_mebo_cpdb.pdf')
plt.show()

## sensor
x = 'cutoff_prop=0, level_cutoff=0'
numsensor_cp_dict = {}
for s in numsensor_cp:
    numsensor_cp_dict[s] = numsensor_cp[s][True]
plot_df = pd.DataFrame([mcorr_res[x]['num_sensor'], numsensor_cp_dict], 
                      index = ['MEBOCOST', 'CellPhoneDB']).T.reset_index().melt(id_vars = 'index')   
fig, ax = plt.subplots(figsize = (6.5, 5))
sb = sns.barplot(data = plot_df, x = 'index', y = 'value', hue = 'variable',
           palette = 'Set2', order = sample_order)
for i in sb.containers:
    sb.bar_label(i,)
ax.tick_params(axis = 'x', rotation = 90)
ax.set(xlabel = '', ylabel = '# of sensors')
ax.legend(title = '', loc='upper left', bbox_to_anchor=(1, 1))
sns.despine()
plt.tight_layout()
# fig.savefig('plots/sensor_num_mebo_cpdb.pdf')
plt.show()



# ### Cobra Flux

# In[30]:


def _get_commu_res2_(cutoff_prop=0, level_cutoff=0, suffix = 'mebocost_res.pk', folders = sample_foler):
    res_pack_tmp = {}
    x1 = 'sender_transport_flux'
    x2 = 'receiver_transport_flux'
    for sample in folders:
        mebo_path = '%s/%s'%(folders[sample], suffix)
        print(mebo_path)
        mebo_obj = mebocost.load_obj(mebo_path)
        if str(level_cutoff) == 'auto':
            exp_prop, met_prop =  mebo_obj.exp_prop, mebo_obj.met_prop
        else:
            exp_prop, met_prop = mebo_obj._check_aboundance_(cutoff_exp = level_cutoff,
                                                       cutoff_met = level_cutoff)

        comm_res = mebo_obj._filter_lowly_aboundant_(pvalue_res = mebo_obj.original_result.copy(),
                                                        cutoff_prop = cutoff_prop,
                                                        met_prop=met_prop,
                                                        exp_prop=exp_prop,
                                                        min_cell_number = 1
                                                     )
        comm_res = comm_res.sort_values(['Sender', 'Receiver', 'Metabolite', 'Sensor'])
        comm_res['Annotation'] = mebo_obj.original_result.sort_values(['Sender', 'Receiver', 'Metabolite', 'Sensor'])['Annotation'].tolist()
        comm_res['sig'] = (comm_res['permutation_test_fdr'] < 0.05) #& (comm_res['metabolite_prop_in_sender'] > 0.2) & (comm_res['sensor_prop_in_receiver'] > 0.2)

        avg_met_celltype = pd.DataFrame(mebo_obj.avg_met.toarray(),index = mebo_obj.avg_met_indexer,
                               columns = mebo_obj.avg_met_columns)
        avg_sensor_celltype = pd.DataFrame(mebo_obj.avg_exp.toarray(),index = mebo_obj.avg_exp_indexer,
                                       columns = mebo_obj.avg_exp_columns)
        comm_res['met_in_sender'] = [avg_met_celltype.loc[m, c] for c, m in comm_res[['Sender', 'Metabolite']].values.tolist()]
        comm_res['sensor_in_receiver'] = [avg_sensor_celltype.loc[m, c] for c, m in comm_res[['Receiver', 'Sensor']].values.tolist()]
        ## compass flux - exchange and transport
        efflux_mat, influx_mat = _get_compass_reaction_(sample, subsystem = ['Transport, extracellular', 'Exchange/demand reaction'])
        comm_res['sender_transport_flux_exchange_transport'] = [efflux_mat[m][c.replace(' ', '~').replace('/', ':')] if m in efflux_mat else np.nan for c, m in comm_res[['Sender', 'Metabolite_Name']].values.tolist()]
        comm_res['receiver_transport_flux_exchange_transport'] = [influx_mat[m][c.replace(' ', '~').replace('/', ':')] if m in influx_mat else np.nan for c, m in comm_res[['Receiver', 'Metabolite_Name']].values.tolist()]
        ## compass flux - transport
        efflux_mat, influx_mat = _get_compass_reaction_(sample, subsystem = ['Transport, extracellular'])
        comm_res['sender_transport_flux_transport'] = [efflux_mat[m][c.replace(' ', '~').replace('/', ':')] if m in efflux_mat else np.nan for c, m in comm_res[['Sender', 'Metabolite_Name']].values.tolist()]
        comm_res['receiver_transport_flux_transport'] = [influx_mat[m][c.replace(' ', '~').replace('/', ':')] if m in influx_mat else np.nan for c, m in comm_res[['Receiver', 'Metabolite_Name']].values.tolist()]
        ## compass flux - exchange
        efflux_mat, influx_mat = _get_compass_reaction_(sample, subsystem = ['Exchange/demand reaction'])
        comm_res['sender_transport_flux_exchange'] = [efflux_mat[m][c.replace(' ', '~').replace('/', ':')] if m in efflux_mat else np.nan for c, m in comm_res[['Sender', 'Metabolite_Name']].values.tolist()]
        comm_res['receiver_transport_flux_exchange'] = [influx_mat[m][c.replace(' ', '~').replace('/', ':')] if m in influx_mat else np.nan for c, m in comm_res[['Receiver', 'Metabolite_Name']].values.tolist()]

        ## cobra flux
        avg_flux_res = pd.DataFrame()
        tmp = cobra_samples[sample]
        for s in os.listdir(tmp):
            if s.startswith('.'):
                continue
            print(tmp+'/'+s)
            ss = pd.read_csv(tmp+'/'+s, index_col = 0)
            ss.columns = [s.replace('.csv', '')]
            avg_flux_res = pd.concat([avg_flux_res, ss], axis = 1)
        ## cobra flux - exchange and transport
        cobra_efflux_mat, cobra_influx_mat = _cobra_eflux_influx_(comm_res=comm_res, 
                                                                  gem_model=human_gem, 
                                                                  flux_res=avg_flux_res, alias=alias,
                                                                 subsystem=['Exchange/demand reactions', 'Transport reactions'])
        comm_res['cobra_sender_transport_flux_exchange_transport'] = [cobra_efflux_mat[m][c.replace(' ', '~').replace('/', ':')] if m in list(cobra_efflux_mat.keys()) else np.nan for c, m in comm_res[['Sender', 'Metabolite_Name']].values.tolist()]
        comm_res['cobra_receiver_transport_flux_exchange_transport'] = [cobra_influx_mat[m][c.replace(' ', '~').replace('/', ':')] if m in list(cobra_influx_mat.keys()) else np.nan for c, m in comm_res[['Receiver', 'Metabolite_Name']].values.tolist()]
        
        ## cobra flux - transport
        cobra_efflux_mat, cobra_influx_mat = _cobra_eflux_influx_(comm_res=comm_res, 
                                                                  gem_model=human_gem, 
                                                                  flux_res=avg_flux_res, alias=alias,
                                                                 subsystem=['Transport reactions'])
        comm_res['cobra_sender_transport_flux_transport'] = [cobra_efflux_mat[m][c.replace(' ', '~').replace('/', ':')] if m in list(cobra_efflux_mat.keys()) else np.nan for c, m in comm_res[['Sender', 'Metabolite_Name']].values.tolist()]
        comm_res['cobra_receiver_transport_flux_transport'] = [cobra_influx_mat[m][c.replace(' ', '~').replace('/', ':')] if m in list(cobra_influx_mat.keys()) else np.nan for c, m in comm_res[['Receiver', 'Metabolite_Name']].values.tolist()]

        ## cobra flux - exchange
        cobra_efflux_mat, cobra_influx_mat = _cobra_eflux_influx_(comm_res=comm_res, 
                                                                  gem_model=human_gem, 
                                                                  flux_res=avg_flux_res, alias=alias,
                                                                 subsystem=['Exchange/demand reactions'])
        comm_res['cobra_sender_transport_flux_exchange'] = [cobra_efflux_mat[m][c.replace(' ', '~').replace('/', ':')] if m in list(cobra_efflux_mat.keys()) else np.nan for c, m in comm_res[['Sender', 'Metabolite_Name']].values.tolist()]
        comm_res['cobra_receiver_transport_flux_exchange'] = [cobra_influx_mat[m][c.replace(' ', '~').replace('/', ':')] if m in list(cobra_influx_mat.keys()) else np.nan for c, m in comm_res[['Receiver', 'Metabolite_Name']].values.tolist()]

        comm_res['label'] = [' : '.join(sorted(x)) for x in comm_res[['Sender', 'Receiver']].values.tolist()]
        coloc_corr = {}
        for i, j in evaluation_data[sample].iterrows():
            coloc_corr[j['label']] = j['corr']
        comm_res[sample] = [coloc_corr.get(x, np.nan) for x in comm_res['label'].tolist()]
        ## norm flux
        flux_norm = lambda x: (x/np.abs(x)) * np.sqrt(np.abs(x)) if x != 0 else 0
        for x in comm_res.columns[comm_res.columns.str.startswith(x1)].tolist():
            comm_res[x] = [flux_norm(x) for x in comm_res[x].tolist()]
        for x in comm_res.columns[comm_res.columns.str.startswith(x2)].tolist():
            comm_res[x] = [flux_norm(x) for x in comm_res[x].tolist()]
        res_pack_tmp[sample] = comm_res
    return(res_pack_tmp)

### run
sample_foler2 = {'human_heart_s1':'/Users/rongbinzheng/Documents/BCH/ChenLab/Metabolism/update_2023/evaluation/spatial/CCI_datasets/human_heart/sample_s1/',
          'human_heart_s2':'/Users/rongbinzheng/Documents/BCH/ChenLab/Metabolism/update_2023/evaluation/spatial/CCI_datasets/human_heart/sample_s2/',
          'human_intestinal':'/Users/rongbinzheng/Documents/BCH/ChenLab/Metabolism/update_2023/evaluation/spatial/CCI_datasets/human_intestinal/',
          'human_PDAC':'/Users/rongbinzheng/Documents/BCH/ChenLab/Metabolism/update_2023/evaluation/spatial/CCI_datasets/human_paad/PDAC/',
          'human_SCC_s1':'/Users/rongbinzheng/Documents/BCH/ChenLab/Metabolism/update_2023/evaluation/spatial/CCI_datasets/human_scc/sample_s1/',
          'human_SCC_s1':'/Users/rongbinzheng/Documents/BCH/ChenLab/Metabolism/update_2023/evaluation/spatial/CCI_datasets/human_scc/sample_s2/'
          }

# res_pack2 = collections.defaultdict()
# res_pack2['cutoff_prop=%s, level_cutoff=%s'%(0, 0)] = _get_commu_res2_(cutoff_prop=0, level_cutoff=0)
# out = open('data/res_pack_20240208_with_subsystem.pk', 'wb')
# pk.dump(res_pack2, out)
# out.close()

res_pack2 = pk.load(open('data/res_pack_20240208_with_subsystem.pk', 'rb'))
res_pack2.keys()


# In[33]:


## for each metabolie, mCCC by cobra flux
def _extract_cobra_compass_comm_res_(x1, x2, efflux_cut=0, influx_cut = 0):
    mcorr_res={}
    for x in res_pack2:
        res_pack_tmp = res_pack2[x].copy()
        mcorr_res[x] = {}
        for sample in res_pack_tmp:
            if sample == 'mouse_cortex':
                continue
            comm_res = res_pack_tmp[sample].copy()
            comm_res = comm_res[(comm_res['Sender'] != comm_res['Receiver'])].query('sig == True')
            ## base_efflux_influx_cut
            tmp1 = comm_res.query('Annotation != "Receptor"').copy()
            tmp2 = comm_res.query('Annotation == "Receptor"').copy()
            tmp1 = tmp1[(tmp1[x1]>efflux_cut) & (tmp1[x2]>influx_cut)]
            tmp2 = tmp2[(tmp2[x1]>efflux_cut)]
            tmp = pd.concat([tmp1, tmp2])
#             print(tmp.head())
            mcorr_res[x][sample] = tmp.groupby('Metabolite_Name').apply(lambda df: pd.Series(spearmanr(df[['Commu_Score', sample, 'label']].dropna().groupby('label').median().dropna()),
                                                                                         index = ['R', 'p']))
            mcorr_res[x][sample]['cellpair_num']=tmp.groupby('Metabolite_Name').apply(lambda df: df.drop_duplicates('label').shape[0]).tolist()
            mcorr_res[x][sample]['comm_num']=tmp.groupby('Metabolite_Name').apply(lambda df: df.shape[0]).tolist()
    return(mcorr_res)


## mCCC by cobra flux - exchange and transport
x1 = 'cobra_sender_transport_flux_exchange_transport'
x2 = 'cobra_receiver_transport_flux_exchange_transport'
cobra_mcorr_res_ex_tr=_extract_cobra_compass_comm_res_(x1, x2)



# In[34]:


x = 'cutoff_prop=0, level_cutoff=0'
mcorr_res_df = pd.DataFrame()
for s in mcorr_res[x]:
    if s == 'num_sensor':
        continue
    ttmp = mcorr_res[x][s].copy()
    ttmp['sample'] = s
    mcorr_res_df = pd.concat([mcorr_res_df, ttmp])
mbcorr_res_df = pd.DataFrame()
for s in mbcorr_res[x]:
    if s == 'num_sensor':
        continue
    ttmp = mbcorr_res[x][s].copy()
    ttmp['sample'] = s
    mbcorr_res_df = pd.concat([mbcorr_res_df, ttmp])
cobra_mcorr_res_df = pd.DataFrame()
for s in cobra_mcorr_res_ex_tr[x]:
    if s == 'num_sensor':
        continue
    ttmp = cobra_mcorr_res_ex_tr[x][s].copy()
    ttmp['sample'] = s
    cobra_mcorr_res_df = pd.concat([cobra_mcorr_res_df, ttmp])
    
mbcorr_res_df['method'] = 'base'
mcorr_res_df['method'] = 'base+compass_flux'
cobra_mcorr_res_df['method'] = 'base+cobra_flux'

plot_df = pd.concat([mbcorr_res_df, cobra_mcorr_res_df, mcorr_res_df])
## num
nplot_df = plot_df.groupby(['sample', 'method'])['comm_num'].sum().reset_index()

sample_order = ['human_heart_s8', 'human_heart_s10',
       'human_intestinal_A3', 'human_PDAC_B','human_SCC_rep2',
       'human_SCC_rep3']


fig, ax = plt.subplots(ncols = 2, nrows = 1, figsize = (10, 8))
sns.violinplot(data = plot_df.query('cellpair_num > 10'),
               x = 'sample', y = 'R', hue = 'method',
                   order = sample_order, 
               palette = 'Paired',
               ax = ax[0],
#                color = 'darkorange',
               cut = 0)

sns.barplot(data = nplot_df,
               x = 'sample', y = 'comm_num', hue = 'method', ax = ax[1],
                   order = sample_order, 
            palette = 'Paired')
ax[0].hlines(0, *ax[0].get_xlim(), linestyle = 'dashed', color = 'grey')
ax[0].tick_params(axis = 'x', rotation = 90)
ax[0].legend().set_visible(False)
ax[0].legend(loc='upper left', bbox_to_anchor=(0, 1.3), title = 'method')
ax[1].legend(loc='upper left', bbox_to_anchor=(0, 1.3), title = 'method')
ax[0].set(xlabel='', ylabel='Spearmzan R')
# ax[0].set_title(x)
ax[1].tick_params(axis = 'x', rotation = 90)
sns.despine()
plt.tight_layout()
fig.savefig('plots/mebocost_compass_cobra_compare.pdf')
plt.show()



# In[76]:


plot_df.query('cellpair_num > 10').groupby('sample').apply(lambda df: ttest_ind(df.query('method == "a_mean"')['R'],
                                                                               df.query('method == "g_mean"')['R'], alternative = 'greater'))


# #### mean vs product when calculating commu score
# 

# In[78]:


### compare average vs gmean

res_pack_mean_cs = collections.defaultdict()
res_pack_mean_cs['cutoff_prop=%s, level_cutoff=%s'%(0, 0)] = _get_commu_res_(cutoff_prop=0, level_cutoff=0, suffix='mebocost_res_mean_for_commu_score.pk')
out = open('data/res_pack_mean_commu_score_20231030.pk', 'wb')
pk.dump(res_pack_mean_cs, out)
out.close()


## for each metabolie 
x1 = 'sender_transport_flux'
x2 = 'receiver_transport_flux'
cutoff1, cutoff2 = 0.1, -0.4 ## define near and far cell types
efflux_cut, influx_cut = 10, 2
mcorr_res_mean_cs={}

for x in res_pack_mean_cs:
    res_pack_tmp = res_pack_mean_cs[x].copy()
    mcorr_res_mean_cs[x] = {'num_sensor':{}}
    for sample in res_pack_tmp:
        if sample == 'mouse_cortex':
            continue
        comm_res = res_pack_tmp[sample].copy()
        comm_res = comm_res[(comm_res['Sender'] != comm_res['Receiver'])].query('sig == True')

        ## base_efflux_influx_cut
        tmp1 = comm_res.query('Annotation != "Receptor"').copy()
        tmp2 = comm_res.query('Annotation == "Receptor"').copy()
        tmp1 = tmp1[(tmp1[x1]>efflux_cut) & (tmp1[x2]>influx_cut)]
        tmp2 = tmp2[(tmp2[x1]>efflux_cut)]
        tmp = pd.concat([tmp1, tmp2])
        mcorr_res_mean_cs[x][sample] = tmp.groupby('Metabolite_Name').apply(lambda df: pd.Series(spearmanr(df[['Commu_Score', sample, 'label']].dropna().groupby('label').median().dropna()),
                                                                                     index = ['R', 'p']))
        mcorr_res_mean_cs[x][sample]['cellpair_num']=tmp.groupby('Metabolite_Name').apply(lambda df: df.drop_duplicates('label').shape[0]).tolist()
        mcorr_res_mean_cs[x][sample]['comm_num']=tmp.groupby('Metabolite_Name').apply(lambda df: df.shape[0]).tolist()
        mcorr_res_mean_cs[x]['num_sensor'][sample] = np.unique(tmp['Sensor'].dropna()).shape[0]


x = 'cutoff_prop=0, level_cutoff=0'
mcorr_res_mean_cs_df = pd.DataFrame()
for s in mcorr_res_mean_cs[x]:
    if s == 'num_sensor':
        continue
    ttmp = mcorr_res_mean_cs[x][s].copy()
    ttmp['sample'] = s
    mcorr_res_mean_cs_df = pd.concat([mcorr_res_mean_cs_df, ttmp])
    
    


# In[86]:


x = 'cutoff_prop=0, level_cutoff=0'
mcorr_res_df = pd.DataFrame()
for s in mcorr_res[x]:
    if s == 'num_sensor':
        continue
    ttmp = mcorr_res[x][s].copy()
    ttmp['sample'] = s
    mcorr_res_df = pd.concat([mcorr_res_df, ttmp])

mcorr_res_mean_cs_df['method'] = 'mean_commu'
mcorr_res_df['method'] = 'product_commu'

plot_df = pd.concat([mcorr_res_df, mcorr_res_mean_cs_df])
## num
nplot_df = plot_df.groupby(['sample', 'method'])['comm_num'].sum().reset_index()

sample_order = ['human_heart_s8', 'human_heart_s10',
       'human_intestinal_A3', 'human_PDAC_B','human_SCC_rep2',
       'human_SCC_rep3']

fig, ax = plt.subplots(ncols = 2, nrows = 1, figsize = (10, 8))
sns.violinplot(data = plot_df.query('cellpair_num > 10').sort_values('method'),
               x = 'sample', y = 'R', hue = 'method',
                   order = sample_order, 
               palette = 'Set2_r',
               ax = ax[0],
#                color = 'darkorange',
               cut = 0)

sns.barplot(data = nplot_df.sort_values('method'),
               x = 'sample', y = 'comm_num', hue = 'method', ax = ax[1],
                   order = sample_order, 
            palette = 'Set2_r')
ax[0].hlines(0, *ax[0].get_xlim(), linestyle = 'dashed', color = 'grey')
ax[0].tick_params(axis = 'x', rotation = 90)
ax[0].legend().set_visible(False)
ax[0].legend(loc='upper left', bbox_to_anchor=(0, 1.3), title = 'method')
ax[1].legend(loc='upper left', bbox_to_anchor=(0, 1.3), title = 'method')
ax[0].set(xlabel='', ylabel='Spearmzan R')
# ax[0].set_title(x)
ax[1].tick_params(axis = 'x', rotation = 90)
sns.despine()
plt.tight_layout()
fig.savefig('plots/mebocost_product_vs_mean_commu_score.pdf')
plt.show()



# In[87]:


plot_df.query('cellpair_num > 10').groupby('sample').apply(lambda df: ttest_ind(df.query('method == "product_commu"')['R'],
                                                                               df.query('method == "mean_commu"')['R'], alternative = 'greater'))


# ### compare average vs gmean for metabolite enzymes 
# 

# In[100]:


res_pack_gmean_sameReact = collections.defaultdict()
res_pack_gmean_sameReact['cutoff_prop=%s, level_cutoff=%s'%(0, 0)] = _get_commu_res_(cutoff_prop=0, level_cutoff=0, suffix='mebocost_res_product_commu_gmean_sameReact_amean_diffReact.pk')

## for each metabolie 
x1 = 'sender_transport_flux'
x2 = 'receiver_transport_flux'
cutoff1, cutoff2 = 0.1, -0.4 ## define near and far cell types
efflux_cut, influx_cut = 10, 2
mcorr_res_gmean_sameReact={}

for x in res_pack_gmean_sameReact:
    res_pack_tmp = res_pack_gmean_sameReact[x].copy()
    mcorr_res_gmean_sameReact[x] = {'num_sensor':{}}
    for sample in res_pack_tmp:
        if sample == 'mouse_cortex':
            continue
        comm_res = res_pack_tmp[sample].copy()
        comm_res = comm_res[(comm_res['Sender'] != comm_res['Receiver'])].query('sig == True')

        ## base_efflux_influx_cut
        tmp1 = comm_res.query('Annotation != "Receptor"').copy()
        tmp2 = comm_res.query('Annotation == "Receptor"').copy()
        tmp1 = tmp1[(tmp1[x1]>efflux_cut) & (tmp1[x2]>influx_cut)]
        tmp2 = tmp2[(tmp2[x1]>efflux_cut)]
        tmp = pd.concat([tmp1, tmp2])
        mcorr_res_gmean_sameReact[x][sample] = tmp.groupby('Metabolite_Name').apply(lambda df: pd.Series(spearmanr(df[['Commu_Score', sample, 'label']].dropna().groupby('label').median().dropna()),
                                                                                     index = ['R', 'p']))
        mcorr_res_gmean_sameReact[x][sample]['cellpair_num']=tmp.groupby('Metabolite_Name').apply(lambda df: df.drop_duplicates('label').shape[0]).tolist()
        mcorr_res_gmean_sameReact[x][sample]['comm_num']=tmp.groupby('Metabolite_Name').apply(lambda df: df.shape[0]).tolist()
        mcorr_res_gmean_sameReact[x]['num_sensor'][sample] = np.unique(tmp['Sensor'].dropna()).shape[0]


x = 'cutoff_prop=0, level_cutoff=0'
mcorr_res_gmean_sameReact_df = pd.DataFrame()
for s in mcorr_res_gmean_sameReact[x]:
    if s == 'num_sensor':
        continue
    ttmp = mcorr_res_gmean_sameReact[x][s].copy()
    ttmp['sample'] = s
    mcorr_res_gmean_sameReact_df = pd.concat([mcorr_res_gmean_sameReact_df, ttmp])
    


# In[105]:


x = 'cutoff_prop=0, level_cutoff=0'
mcorr_res_df = pd.DataFrame()
for s in mcorr_res[x]:
    if s == 'num_sensor':
        continue
    ttmp = mcorr_res[x][s].copy()
    ttmp['sample'] = s
    mcorr_res_df = pd.concat([mcorr_res_df, ttmp])

mcorr_res_gmean_sameReact_df['method'] = 'g_mean_a_mean'
mcorr_res_df['method'] = 'a_mean'

plot_df = pd.concat([mcorr_res_df, mcorr_res_gmean_sameReact_df])
## num
nplot_df = plot_df.groupby(['sample', 'method'])['comm_num'].sum().reset_index()

sample_order = ['human_heart_s8', 'human_heart_s10',
       'human_intestinal_A3', 'human_PDAC_B','human_SCC_rep2',
       'human_SCC_rep3']

fig, ax = plt.subplots(ncols = 2, nrows = 1, figsize = (10, 8))
sns.violinplot(data = plot_df.query('cellpair_num > 10').sort_values('method'),
               x = 'sample', y = 'R', hue = 'method',
                   order = sample_order, 
               palette = 'Paired',
               ax = ax[0],
#                color = 'darkorange',
               cut = 0)

sns.barplot(data = nplot_df.sort_values('method'),
               x = 'sample', y = 'comm_num', hue = 'method', ax = ax[1],
                   order = sample_order, 
            palette = 'Paired')
ax[0].hlines(0, *ax[0].get_xlim(), linestyle = 'dashed', color = 'grey')
ax[0].tick_params(axis = 'x', rotation = 90)
ax[0].legend().set_visible(False)
ax[0].legend(loc='upper left', bbox_to_anchor=(0, 1.3), title = 'method')
ax[1].legend(loc='upper left', bbox_to_anchor=(0, 1.3), title = 'method')
ax[0].set(xlabel='', ylabel='Spearmzan R')
# ax[0].set_title(x)
ax[1].tick_params(axis = 'x', rotation = 90)
sns.despine()
plt.tight_layout()
fig.savefig('plots/mebocost_gmean_sameReact_amean_diffReact.pdf')
plt.show()



# In[103]:


plot_df.query('cellpair_num > 10').groupby('sample').apply(lambda df: ttest_ind(df.query('method == "a_mean"')['R'],
                                                                               df.query('method == "g_mean_a_mean"')['R'], alternative = 'greater'))


# In[ ]:




