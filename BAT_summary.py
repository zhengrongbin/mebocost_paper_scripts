#!/usr/bin/env python
# coding: utf-8

# In[11]:


import os,sys
import time
import pickle as pk
from datetime import datetime
import numpy as np
import pandas as pd
import traceback
from operator import itemgetter
import statsmodels
import scipy
from scipy import sparse
import collections
import scanpy as sc
import re
import matplotlib.pyplot as plt
import seaborn as sns
from venn import venn
from adjustText import adjust_text
from matplotlib import pyplot as plt
import matplotlib.patheffects as PathEffects
from matplotlib.backends.backend_pdf import PdfPages
from scipy.cluster.hierarchy import dendrogram, linkage
import networkx as nx
import matplotlib

## disable warnings
import warnings
warnings.filterwarnings("ignore")

import multiprocessing

import importlib
import configparser 
from matplotlib.backends.backend_pdf import PdfPages

from mebocost import mebocost

from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib.pyplot import gcf

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
         'VSM': '#9edae5'}
plt.rcParams.update(plt.rcParamsDefault)
rc={"axes.labelsize": 16, "xtick.labelsize": 12, "ytick.labelsize": 12,
    "figure.titleweight":"bold", #"font.size":14,
    "figure.figsize":(5.5,4.2), "font.weight":"regular", "legend.fontsize":10,
    'axes.labelpad':8, 'figure.dpi':300}
plt.rcParams.update(**rc)





# In[12]:


compass_met_ann = pd.read_csv('/Users/rongbinzheng/Documents/test/MEBOCOST/data/Compass/met_md.csv')

compass_rxn_ann = pd.read_csv('/Users/rongbinzheng/Documents/test/MEBOCOST/data/Compass/rxn_md.csv')

met_ann = pd.read_csv('/Users/rongbinzheng/Documents/github/MEBOCOST/data/mebocost_db/common/metabolite_annotation_HMDB_summary.tsv',
                     sep = '\t')
alias = {str(i).upper():[str(x).upper() for x in str(j).split('; ')] for i,j in met_ann[['metabolite', 'synonyms_name']].values.tolist()}


def _cobra_eflux_influx_(comm_res, gem_model, flux_res, alias):
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
    transport_r_flux_d = flux_res.loc[transport_r['reaction_id'],].apply(lambda col: col * transport_r['direction'].tolist())
    transport_r_flux_d.index = transport_r.index.tolist()
    transport_r = pd.concat([transport_r, transport_r_flux_d], axis = 1)
    exchange_r = transport_r.query('subsystem == "Exchange/demand reactions"')
    transport_r = transport_r.query('subsystem == "Transport reactions"')
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


def _get_compass_flux_(compass_folder):  
    uptake_path = os.path.join(compass_folder, 'uptake.tsv')
    secret_path = os.path.join(compass_folder, 'secretions.tsv')
    reaction_path = os.path.join(compass_folder, 'reactions.tsv')

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

def _get_commu_res_(mebo_path, compass_folder, cutoff_prop=0, exp_cutoff=0, met_cutoff=0, efflux_cut = 'auto', influx_cut='auto'):
    res_pack_tmp = {}
    x1 = 'sender_transport_flux'
    x2 = 'receiver_transport_flux'
    mebo_obj = mebocost.load_obj(mebo_path)
    if str(exp_cutoff) == 'auto' and str(met_cutoff) == 'auto':
        exp_prop, met_prop =  mebo_obj.exp_prop, mebo_obj.met_prop
    else:
        exp_prop, met_prop = mebo_obj._check_aboundance_(cutoff_exp = exp_cutoff,
                                                   cutoff_met = met_cutoff)

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
    ## compass
    efflux_mat, influx_mat = _get_compass_flux_(compass_folder = compass_folder)
    x1 = 'sender_transport_flux'
    x2 = 'receiver_transport_flux'
    comm_res[x1] = [efflux_mat.loc[m,c.replace(' ', '~').replace('/', ':')] if m in efflux_mat.index.tolist() else np.nan for c, m in comm_res[['Sender', 'Metabolite_Name']].values.tolist()]
    comm_res[x2] = [influx_mat.loc[m,c.replace(' ', '~').replace('/', ':')] if m in influx_mat.index.tolist() else np.nan for c, m in comm_res[['Receiver', 'Metabolite_Name']].values.tolist()]
    flux_norm = lambda x: (x/np.abs(x)) * np.sqrt(np.abs(x)) if x != 0 else 0
    comm_res[x1] = [flux_norm(x) for x in comm_res[x1].tolist()]
    comm_res[x2] = [flux_norm(x) for x in comm_res[x2].tolist()] 
    if efflux_cut == 'auto':
        efflux_cut = np.percentile(comm_res[x1], 25)
    if influx_cut == 'auto':
        influx_cut = np.percentile(comm_res[x2], 25)
    print('efflux_cut:', efflux_cut)
    print('influx_cut:', influx_cut)
    ## base_efflux_influx_cut
    tmp_na = comm_res[pd.isna(comm_res[x1]) | pd.isna(comm_res[x2])].query('sig == True')
    tmp1 = comm_res.query('sig == True and Annotation != "Receptor"').copy()
    tmp2 = comm_res.query('sig == True and Annotation == "Receptor"').copy()
    combine_score = lambda x, xx: ((x*xx)/np.abs(x*xx)) * np.sqrt(np.abs(x*xx))
    tmp1 = tmp1[(tmp1[x1]>efflux_cut) & (tmp1[x2]>influx_cut)]
    tmp2 = tmp2[(tmp2[x1]>efflux_cut)]
    tmp = pd.concat([tmp1, tmp2, tmp_na])
    return(mebo_obj, tmp, efflux_mat, influx_mat)

        
        


# ## Cold2
# 

# In[13]:


## cold2 BAT scRNA-seq analyzed by MEBOCOST v1.0.2
# cold2_mebo = mebocost.load_obj('../../mebocost_BAT_v1.0.2/cold2/scBAT_mebocost_cold2.pk')
ec = 0.866
mc = 0.141

# exp_prop, met_prop = cold2_mebo._check_aboundance_(cutoff_exp = ec,
#                                                     cutoff_met = mc)
# cold2_mebo.commu_res = cold2_mebo._filter_lowly_aboundant_(pvalue_res = cold2_mebo.original_result,
#                                                     cutoff_prop = 0.15,
#                                                     met_prop=met_prop,
#                                                     exp_prop=exp_prop)




# In[14]:


cold2_mebo, cold2_commu_res, efflux_mat, influx_mat = _get_commu_res_(mebo_path='../../mebocost_BAT_v1.0.2/cold2/scBAT_mebocost_cold2.pk',
                                        compass_folder='../flux_2023/compass/avg_exp_compass/mBAT_cold2/', 
                                        cutoff_prop=0.15, exp_cutoff = ec, met_cutoff=mc,
                                        efflux_cut = 'auto', influx_cut='auto')
cold2_mebo.commu_res = cold2_commu_res.copy()


# In[71]:


cold2_mebo.eventnum_bar(
                    sender_focus=[],
                    metabolite_focus=[],
                    sensor_focus=[],
                    receiver_focus=[],
                    and_or='and',
                    pval_method='permutation_test_fdr',
                    pval_cutoff=0.05,
                    comm_score_col='Commu_Score',
                    comm_score_cutoff = None,
                    cutoff_prop = 0.15,
                    figsize='auto',
                    save='../updated_plots/0822/Cold2_commu_num.pdf',
                    show_plot=True,
                    include=['sender-receiver'],
                    group_by_cell=True,
                    colorcmap='tab20',
                    return_fig=False
                )



# In[15]:


def _matrix_commu_(comm_res, pval_method, pval_cutoff, 
                   comm_score_col = 'Commu_Score',
                   comm_score_cutoff = None,
                   cutoff_prop = None 
                  ):
    cols = ['Metabolite_Name', 'Sensor', 'Sender', 'Receiver',
            comm_score_col, pval_method, 'metabolite_prop_in_sender', 'sensor_prop_in_receiver']
    commu_df = comm_res[cols]
    ## especially for permutation test, p value could be 0, so set to 2.2e-16
    commu_df.loc[commu_df[pval_method]==0,pval_method] = 2.2e-16
    # print(communication_res_tidy.head())
    ## focus on significant ones, and communication score should be between 0 and 1
    if not comm_score_cutoff:
        comm_score_cutoff = 0
    if not cutoff_prop:
        cutoff_prop = 0
    commu_df = commu_df[(commu_df[pval_method] <= pval_cutoff) &
                        (commu_df[comm_score_col] > comm_score_cutoff) &
                        (commu_df['metabolite_prop_in_sender'] > cutoff_prop) &
                        (commu_df['sensor_prop_in_receiver'] > cutoff_prop)
                       ]
                                # (commu_df[comm_score_col] <= 1) &
                                # (commu_df[comm_score_col] >= 0)]
    commu_df['Signal_Pair'] = commu_df['Metabolite_Name'] + '~' + commu_df['Sensor']
    commu_df['Cell_Pair'] = commu_df['Sender'] + '→' + commu_df['Receiver']
    commu_df['-log10(pvalue)'] = -np.log10(commu_df[pval_method])
    return commu_df

def _make_comm_event_(commu_res,
                        pval_method = 'permutation_test_fdr',
                        pval_cutoff = 0.05,
                        comm_score_col = 'Commu_Score',
                        comm_score_cutoff = None,
                        cutoff_prop = None
                     ):
    plot_tmp = _matrix_commu_(commu_res, pval_method, pval_cutoff, comm_score_col, comm_score_cutoff, cutoff_prop)
    if plot_tmp.shape[0] == 0:
#         info('No communication events under pval_cutoff:{} and comm_score_cutoff:{}, try to tune them!'.format(pval_cutoff, comm_score_cutoff))
        return None
    ## visulize the communication frequency between cells
    count_df = pd.DataFrame(plot_tmp.groupby('Cell_Pair')['Cell_Pair'].count())
    count_df.index = count_df.index.tolist()
    count_df = pd.concat([count_df, 
                          pd.DataFrame(count_df.index.str.split('→').tolist(), index = count_df.index)], axis = 1)
    count_df.columns = ['Count', 'Sender', 'Receiver']
    ## communicate event summary
    comm_event = {}
    for x in plot_tmp['Cell_Pair'].tolist():
        tmp = plot_tmp[plot_tmp['Cell_Pair']==x]['-log10(pvalue)'] # all the p values
        comm_event[x] = [len(tmp), -sum(-tmp)] ## sum of log p = the product of p
    comm_event = pd.DataFrame.from_dict(comm_event, orient = 'index')
    comm_event = pd.concat([comm_event, 
                            pd.DataFrame(comm_event.index.str.split('→').tolist(), index = comm_event.index)], axis = 1)
    comm_event.columns = ['Count', '-log10(pvalue)', 'Sender', 'Receiver']
    return(comm_event)


def _commu_network_plot_(commu_res,
                        sender_focus = [],
                        metabolite_focus = [],
                        sensor_focus = [],
                        receiver_focus = [],
                        remove_unrelevant = False,
                        and_or = 'and',
                        pval_method = 'permutation_test_fdr',
                        pval_cutoff = 0.05,
                        node_cmap = 'tab20',
                        figsize = 'auto',
                        line_cmap = 'RdBu_r',
                        line_color_vmin = None,
                        line_color_vmax = None,
                        line_width_col = 'Count',
                        linewidth_norm = (0.1, 1),
                        node_size_norm = (50, 300),
                        node_size_range = (None, None),
                        linewidth_range = (None, None),
                        adjust_text_pos_node = True,
                        node_text_hidden = False,
                        node_text_font = 10,
                        pdf = None,
                        save_plot = True,
                        show_plot = False,
                        comm_score_col = 'Commu_Score',
                        comm_score_cutoff = None,
                        cutoff_prop = None,
                        text_outline = False,
                        return_fig = False):
    """
    plot network figure to show the interactions between cells
    --------------
    comm_event: a data frame with the format like this:
                                    Count   -log10(pvalue)  Sender  Receiver
        Malignant_0->Malignant_0    13  17.864293   Malignant_0 Malignant_0
        Malignant_12->Malignant_0   16  21.788151   Malignant_12    Malignant_0
        Malignant_1->Malignant_0    10  13.598459   Malignant_1 Malignant_0

    line_cmap: line color map, usually for the overall score, a sum of -log10(pvalue) for all metabolite-sensor communications to the connection
    node_cmap: node color map, usually for different type of cells
    figsize: a tuple to indicate the width and height for the figure, default is automatically estimate
    sender_col: column names for sender cells
    receiver_col: column names for sender cells
    """
#     info('show communication in cells by network plot')

    ## clean
    plt.close()
    
    sender_col = 'Sender'
    receiver_col = 'Receiver'
    metabolite_col = 'Metabolite_Name'
    sensor_col = 'Sensor'
    line_color_col = '-log10(pvalue)'
    
    rad = .2
    conn_style = f'arc3,rad={rad}'

    ## adjust by filter
    focus_commu = commu_res.copy()
    if and_or == 'and':
        if sender_focus:
            focus_commu = focus_commu[(focus_commu[sender_col].isin(sender_focus))]
        if receiver_focus:
            focus_commu = focus_commu[(focus_commu[receiver_col].isin(receiver_focus))]
        if metabolite_focus:
            focus_commu = focus_commu[(focus_commu[metabolite_col].isin(metabolite_focus))]
        if sensor_focus:
            focus_commu = focus_commu[(focus_commu[sensor_col].isin(sensor_focus))]
    else:
        if sender_focus or receiver_focus or metabolite_focus or sensor_focus:
            focus_commu = focus_commu[(focus_commu[sender_col].isin(sender_focus)) |
                                     (focus_commu[receiver_col].isin(receiver_focus)) |
                                     (focus_commu[metabolite_col].isin(metabolite_focus)) |
                                     (focus_commu[sensor_col].isin(sensor_focus))]
    if focus_commu.shape[0] == 0:
        info('No communication detected under the filtering')
                                     
    if remove_unrelevant is True:
        ## make comm_event
        comm_event = _make_comm_event_(
                                commu_res = focus_commu,
                                pval_method = pval_method,
                                pval_cutoff = pval_cutoff,
                                comm_score_col = comm_score_col,
                                comm_score_cutoff = comm_score_cutoff,
                                cutoff_prop = cutoff_prop
                            )
    else:
        ## make comm_event
        comm_event = _make_comm_event_(
                                commu_res = focus_commu,
                                pval_method = pval_method,
                                pval_cutoff = pval_cutoff,
                                comm_score_col = comm_score_col,
                                comm_score_cutoff = comm_score_cutoff,
                                cutoff_prop = cutoff_prop
                            )
        
    if comm_event is None:
        print(1)
        return
    
    if figsize == 'auto' or not figsize:
        node_num = len(set(comm_event[sender_col].tolist()+comm_event[receiver_col].tolist()))
        figsize = (2.8+node_num*0.2, 1.8+node_num * 0.1)

    fig = plt.figure(constrained_layout=True, figsize=figsize)

    subfigs = fig.add_gridspec(2, 3, width_ratios=[5.2, .7, .7])
    leftfig = fig.add_subplot(subfigs[:, 0])
    midfig = [fig.add_subplot(subfigs[0, 1]), fig.add_subplot(subfigs[1, 1])]
    rightfig = fig.add_subplot(subfigs[:, 2])

    ## get the node cells
    total_count = collections.Counter(comm_event[receiver_col].tolist()+comm_event[sender_col].tolist())
    G = nx.DiGraph(directed = True)

    ## add node
    for n in sorted(list(total_count.keys())):
        G.add_node(n)
        
    ## node size and color
    if node_size_range[0] and node_size_range[1]:
        node_size_norm_fun = lambda x, y: node_size_norm[0]+((x-node_size_range[0]) / (node_size_range[1] - node_size_range[0]) * (node_size_norm[1]-node_size_norm[0])) if node_size_range[0] != node_size_range[1] else node_size_norm[0]+((x-node_size_range[0]) / node_size_range[1] * (node_size_norm[1]-node_size_norm[0])) 
    else:
        node_size_norm_fun = lambda x, y: node_size_norm[0]+((x-min(y)) / (max(y) - min(y)) * (node_size_norm[1]-node_size_norm[0])) if max(y) != min(y) else node_size_norm[0]+((x-min(y)) / max(y) * (node_size_norm[1]-node_size_norm[0])) 
    
    node_size = [node_size_norm_fun(total_count.get(x, 0), total_count.values()) for x in G.nodes()] 

    if type(node_cmap) == type(dict()):
        node_col = np.array([node_cmap.get(x) for x in G.nodes()])
    else:
        node_col = np.array([plt.cm.get_cmap(node_cmap)(i) for i in range(len(G.nodes()))])


    if linewidth_range[0] and linewidth_range[1]:
        linewidth_norm_fun = lambda x, y: linewidth_norm[0]+((x-linewidth_range[0]) / (linewidth_range[1] - linewidth_range[0]) * (linewidth_norm[1]-linewidth_norm[0])) if linewidth_range[1] != linewidth_range[0] else linewidth_norm[0]+((x-linewidth_range[0]) / linewidth_range[1] * (linewidth_norm[1]-linewidth_norm[0]))
    else:
        linewidth_norm_fun = lambda x, y: linewidth_norm[0]+((x-min(y)) / (max(y) - min(y)) * (linewidth_norm[1]-linewidth_norm[0])) if max(y) != min(y) else linewidth_norm[0]+((x-min(y)) / max(y) * (linewidth_norm[1]-linewidth_norm[0]))
        
    edge_color_norm = matplotlib.colors.Normalize(vmin = line_color_vmin if line_color_vmin else comm_event[line_color_col].min(), vmax = line_color_vmax if line_color_vmax else comm_event[line_color_col].max())

    if focus_commu.shape[0] == 0:
        # Custom the nodes:
        pos = nx.circular_layout(G)
        nx.draw(G, pos, #with_labels=with_labels,
                font_size = node_text_font, node_color=node_col,
                node_size=node_size, connectionstyle=conn_style,
                cmap = node_cmap, ax = leftfig, alpha = .9)
    else:
        comm_event_filter = _make_comm_event_(
                                        commu_res = focus_commu,
                                        pval_method = pval_method,
                                        pval_cutoff = pval_cutoff,
                                        comm_score_col = comm_score_col,
                                        comm_score_cutoff = comm_score_cutoff
                                    )
        if comm_event_filter is None:
            return
        
        for i,line in comm_event_filter.iterrows():
            sender = line[sender_col]
            receiver = line[receiver_col]
            G.add_edge(sender, receiver)

        # Custom the nodes:
        pos = nx.circular_layout(G)

        if not line_color_vmin:
            line_color_vmin = np.percentile(comm_event[line_color_col], 0)
        if not line_color_vmax:
            line_color_vmax = np.percentile(comm_event[line_color_col], 100)

        edge_color = [plt.cm.get_cmap(line_cmap)(edge_color_norm(comm_event[(comm_event[sender_col]==x[0]) &
                                  (comm_event[receiver_col]==x[1])].loc[:, line_color_col])) for x in G.edges]

        linewidth = [linewidth_norm_fun(comm_event[(comm_event[sender_col]==x[0]) & 
                                                   (comm_event[receiver_col]==x[1])].loc[:, line_width_col],
                                        comm_event[line_width_col]) for x in G.edges]
        nx.draw(G, pos, 
#                 with_labels=with_labels,
                arrows = True, 
                arrowstyle = '-|>',
                font_size = node_text_font,
                node_color=node_col,
                node_size=node_size, 
                edge_color=edge_color, 
                width=linewidth,
#                 cmap = node_cmap, 
                connectionstyle=conn_style,
                ax = leftfig, alpha = .8)
#         nx.draw_networkx_edges(G, pos=pos, edgelist=G.edges(), edge_color='black',
#                        connectionstyle=conn_style)
        
    if node_text_hidden is False or node_text_hidden is None:
        if adjust_text_pos_node:
            text = []
            for x in pos.keys():
                txt = leftfig.text(pos[x][0], pos[x][1], x, size = node_text_font)
                if text_outline:
                    txt.set_path_effects([PathEffects.withStroke(linewidth=2, foreground='w')])
                text.append(txt)

            adjust_text(text, 
                        # arrowprops=dict(arrowstyle="-", color = 'k'),
                        ax = leftfig
                        )   
        else:
            for x in pos.keys():
                txt = leftfig.text(pos[x][0], pos[x][1], x, size = node_text_font, ha = 'center')
                if text_outline:
                    txt.set_path_effects([PathEffects.withStroke(linewidth=2, foreground='w')])

    ## node size
    if node_size_range[0] and node_size_range[1]:
        node_size_ann = sorted(list(set([np.min(node_size_range),
                    np.percentile(node_size_range, 50),
                    np.max(node_size_range)])))
    else:
        node_size_ann = sorted(list(set([np.percentile(list(total_count.values()), 10),
                    np.percentile(list(total_count.values()), 50),
                    np.percentile(list(total_count.values()), 90)])))
    ## legend for connected node size
    for label in node_size_ann:
        midfig[0].scatter([],[],
                color = 'black',
                facecolors='none',
                s = node_size_norm_fun(label, node_size_ann),
                label=int(label))
    ## legend for communication evens, line width
    if linewidth_range[0] and linewidth_range[1]:
        line_ann = sorted(list(set([np.min(linewidth_range),
                    np.percentile(linewidth_range, 50),
                    np.max(linewidth_range)])))
    else:
        line_ann = sorted(list(set([np.min(comm_event[line_width_col]),
                    np.percentile(comm_event[line_width_col], 50),
                    np.max(comm_event[line_width_col])])))

    for label in line_ann:
        midfig[1].plot([],[],'g',
                color = 'black',
                linewidth = linewidth_norm_fun(label, line_ann),
                label=int(label))
    midfig[0].axis('off')
    midfig[1].axis('off')
    midfig[0].legend(title = '# of\nConnected\nNodes', loc='center', frameon = False)
    midfig[1].legend(title = '# of\nCommunication\nEvents', loc='center', frameon = False)
    ## legend for communication confidence, line color
    sm = matplotlib.cm.ScalarMappable(cmap=plt.cm.get_cmap(line_cmap), norm = edge_color_norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, shrink = .5, location = 'left')
    cbar.set_label(label='Overall Score',fontsize = 10)
    rightfig.axis('off')
    pdf.savefig(fig) if pdf else None
    if show_plot:
        plt.show()
    plt.close()
    if return_fig:
        return(fig)


    


# In[72]:


## circle plot to show communications between cell groups
fig1 = _commu_network_plot_(
                    commu_res = cold2_commu_res,
                    sender_focus = [],
                    metabolite_focus = [],
                    sensor_focus = [],
                    receiver_focus = [],
                    remove_unrelevant = False,
                    and_or = 'and',
                    pval_method = 'permutation_test_fdr',
                    pval_cutoff = 0.05,
                    node_cmap = colmap,
                    figsize = (6.1, 3.5),
                    line_cmap = 'bwr',
                    line_color_vmin = None,
                    line_color_vmax = 200,
                    line_width_col = 'Count',
                    linewidth_norm = (0.2, 1),
                    node_size_norm = (50, 200),
                    adjust_text_pos_node = True,
                    node_text_hidden = False,
                    node_text_font = 10,
                    pdf = None,
                    save_plot = True,
                    show_plot = True,
                    comm_score_col = 'Commu_Score',
                    comm_score_cutoff = None,
                    cutoff_prop = None,
                    text_outline = False,
                    return_fig = True)


fig2 = _commu_network_plot_(
                    commu_res = cold2_commu_res,
                    sender_focus = [],
                    metabolite_focus = [],
                    sensor_focus = [],
                    receiver_focus = [],
                    remove_unrelevant = False,
                    and_or = 'and',
                    pval_method = 'permutation_test_fdr',
                    pval_cutoff = 0.05,
                    node_cmap = colmap,
                    figsize = (6.1, 3.5),
                    line_cmap = 'bwr',
                    line_color_vmin = None,
                    line_color_vmax = 200,
                    line_width_col = 'Count',
                    linewidth_norm = (0.2, 1),
                    node_size_norm = (50, 200),
                    adjust_text_pos_node = True,
                    node_text_hidden = True,
                    node_text_font = 10,
                    pdf = None,
                    save_plot = True,
                    show_plot = True,
                    comm_score_col = 'Commu_Score',
                    comm_score_cutoff = None,
                    cutoff_prop = None,
                    text_outline = False,
                    return_fig = True)
pdf = PdfPages('../updated_plots/0822/Cold2_network.pdf')
pdf.savefig(fig1)
pdf.savefig(fig2)
pdf.close()


# In[73]:


### dot plot to show the number of communications between cells

cold2_mebo.count_dot_plot(
                        pval_method='permutation_test_fdr',
                        pval_cutoff=0.05,
                        cmap='bwr',
                        figsize='auto',
                        save='../updated_plots/0822/Cold2_count_dot.pdf',
                        dot_size_norm=(20, 200),
                        dot_color_vmin=None,
                        dot_color_vmax=200,
                        show_plot=True,
                        comm_score_col='Commu_Score',
                        comm_score_cutoff=None,
                        return_fig = False
                    )




# In[74]:


### dot-plot show detailed communication
cold2_mebo.commu_dotmap(
                sender_focus=[],
                metabolite_focus=[],
                sensor_focus=[],
                receiver_focus=['Adipocytes'],
                and_or='or',
                pval_method='permutation_test_fdr',
                pval_cutoff=0.05,
                figsize=(10, 7),
                cmap='bwr',
                node_size_norm=(10, 150),
                save='../updated_plots/0822/Cold2_dotmap_adipocyte_receiver.pdf',
                show_plot=True,
                comm_score_col='Commu_Score',
                comm_score_cutoff=None,
                swap_axis = False,
                return_fig = False
                )



# In[40]:


pdf = PdfPages('../updated_plots/0822/enzyme_sensor_violin.pdf')
fig1 = cold2_mebo.violin_plot(sensor_or_met=['L-Glutamine'],
                  cell_focus=cold2_commu_res.query('Receiver == "Adipocytes"')['Sender'].unique().tolist(), 
                       return_fig = True)

fig2 = cold2_mebo.violin_plot(sensor_or_met=['Slc1a5'],
                      cell_focus = cold2_commu_res.query('Receiver == "Adipocytes"')['Sender'].unique().tolist(),
                      return_fig = True)
pdf.savefig(fig1)
pdf.savefig(fig2)
pdf.close()


# In[21]:


cold2_mebo.FlowPlot(
                pval_method='permutation_test_fdr',
                pval_cutoff=0.05,
                sender_focus=['EC'],
                metabolite_focus=['L-Glutamine'],
                sensor_focus=[],
                receiver_focus=['Adipocytes', 'Pdgfra_APC'],
                remove_unrelevant = True,
                and_or='and',
                node_label_size=12,
                node_alpha=0.6,
                figsize=(9, 6),
                node_cmap='Set1',
                line_cmap='bwr',
                line_vmin = None,
                line_vmax = 15.5,
                node_size_norm=(20, 150),
                linewidth_norm=(0.5, 5),
                save='../updated_plots/0822/Cold2_flowmap_EC_to_adipocyte.pdf',
                show_plot=True,
                comm_score_col='Commu_Score',
                comm_score_cutoff=None,
                text_outline=False,
                return_fig = False
            )



# ## other condition
# 

# In[26]:


TN_mebo, TN_commu_res, efflux_mat, influx_mat = _get_commu_res_(mebo_path='../../mebocost_BAT_v1.0.2/other_cond/scBAT_mebocost_TN.pk',
                                        compass_folder='../flux_2023/compass/avg_exp_compass/mBAT_TN/', 
                                        cutoff_prop=0.15, exp_cutoff = ec, met_cutoff=mc,
                                        efflux_cut = 'auto', influx_cut='auto')
TN_mebo.commu_res = TN_commu_res.copy().query('Sender != "Adipocytes" and Receiver != "Adipocytes"')


RT_mebo, RT_commu_res, efflux_mat, influx_mat = _get_commu_res_(mebo_path='../../mebocost_BAT_v1.0.2/other_cond/scBAT_mebocost_RT.pk',
                                        compass_folder='../flux_2023/compass/avg_exp_compass/mBAT_RT/', 
                                        cutoff_prop=0.15, exp_cutoff = ec, met_cutoff=mc,
                                        efflux_cut = 'auto', influx_cut='auto')
RT_mebo.commu_res = RT_commu_res.copy().query('Sender != "Adipocytes" and Receiver != "Adipocytes"')


cold7_mebo, cold7_commu_res, efflux_mat, influx_mat = _get_commu_res_(mebo_path='../../mebocost_BAT_v1.0.2/other_cond/scBAT_mebocost_cold7.pk',
                                        compass_folder='../flux_2023/compass/avg_exp_compass/mBAT_cold7/', 
                                        cutoff_prop=0.15, exp_cutoff = ec, met_cutoff=mc,
                                        efflux_cut = 'auto', influx_cut='auto')
cold7_mebo.commu_res = cold7_commu_res.copy().query('Sender != "Adipocytes" and Receiver != "Adipocytes"')



# In[26]:


## output to a excel file
with pd.ExcelWriter('detected_communication_conditions.xlsx') as writer:
    tmp = TN_mebo.commu_res.copy()
    cols = ['Sender', 'Metabolite', 'Metabolite_Name', 'Receiver', 'Sensor', 'Commu_Score', 'permutation_test_fdr']
    tmp = tmp[tmp['permutation_test_fdr']<0.05][cols].groupby('permutation_test_fdr').apply(lambda df: df.sort_values(['Commu_Score'], ascending = False))
    tmp.to_excel(writer, sheet_name='TN', index = False)
    
    tmp = RT_mebo.commu_res.copy()
    cols = ['Sender', 'Metabolite', 'Metabolite_Name', 'Receiver', 'Sensor', 'Commu_Score', 'permutation_test_fdr']
    tmp = tmp[tmp['permutation_test_fdr']<0.05][cols].groupby('permutation_test_fdr').apply(lambda df: df.sort_values(['Commu_Score'], ascending = False))
    tmp.to_excel(writer, sheet_name='RT', index = False)
    
    tmp = cold2_mebo.commu_res.copy()
    cols = ['Sender', 'Metabolite', 'Metabolite_Name', 'Receiver', 'Sensor', 'Commu_Score', 'permutation_test_fdr']
    tmp = tmp[tmp['permutation_test_fdr']<0.05][cols].groupby('permutation_test_fdr').apply(lambda df: df.sort_values(['Commu_Score'], ascending = False))
    tmp.to_excel(writer, sheet_name='Cold2', index = False)
    
    tmp = cold7_mebo.commu_res.copy()
    cols = ['Sender', 'Metabolite', 'Metabolite_Name', 'Receiver', 'Sensor', 'Commu_Score', 'permutation_test_fdr']
    tmp = tmp[tmp['permutation_test_fdr']<0.05][cols].groupby('permutation_test_fdr').apply(lambda df: df.sort_values(['Commu_Score'], ascending = False))
    tmp.to_excel(writer, sheet_name='Cold7', index = False)
    
    


# In[80]:


res1 = pd.DataFrame()
res2 = pd.DataFrame()
for cond, comm_res in [['TN', TN_mebo.commu_res], ['RT', RT_mebo.commu_res], 
             ['cold2', cold2_mebo.commu_res], ['cold7', cold7_mebo.commu_res]]:
    print(cond)
    tmp1 = comm_res.query('sig == True and ((Sender == "EC")|(Sender == "Lymph_EC")) and ((Receiver == "Adipocytes")|(Receiver == "Pdgfra_APC"))')
    tmp1['cond'] = cond
    tmp2 = comm_res.query('sig == True and ((Receiver == "EC")|(Receiver == "Lymph_EC")) and ((Sender == "Adipocytes")|(Sender == "Pdgfra_APC"))')
    tmp2['cond'] = cond
    res1 = pd.concat([res1, tmp1])
    res2 = pd.concat([res2, tmp2])
    
res1['label'] = res1['Sender']+'~'+res1['Metabolite_Name']+'~'+res1['Sensor']+'~'+res1['Receiver']
# res1['base_flux'] = res1[['base_efflux_influx_cobra', 'base_efflux_influx_compass']].T.max()
res1_tmp = res1.pivot_table(index = 'label', columns = 'cond', values = 'Commu_Score').fillna(0)[['TN', 'RT', 'cold2', 'cold7']]
res1_tmp = pd.concat([pd.DataFrame(res1_tmp.index.str.split('~').tolist(),
            index = res1_tmp.index, columns = ['Sender', 'Metabolite', 'Sensor', 'Receiver']),
                                   res1_tmp], axis = 1)
res1_tmp.to_csv('../experimental_validation/EC_to_Adipocytes_update.csv', index = None)

res2['label'] = res2['Sender']+'~'+res2['Metabolite_Name']+'~'+res2['Sensor']+'~'+res2['Receiver']
res2_tmp = res2.pivot_table(index = 'label', columns = 'cond', values = 'Commu_Score').fillna(0)[['TN', 'RT', 'cold2', 'cold7']]
res2_tmp = pd.concat([pd.DataFrame(res2_tmp.index.str.split('~').tolist(),
            index = res2_tmp.index, columns = ['Sender', 'Metabolite', 'Sensor', 'Receiver']),
                                   res2_tmp], axis = 1)
res2_tmp.to_csv('../experimental_validation/Adipocytes_to_EC_update.csv', index = None)


# In[27]:


## number of communication across conditions
## remove adipocytes in TN, RT, and Cold7, since the adipocyte number in those condition is too few
df = pd.Series({
    'TN': TN_mebo.commu_res[TN_mebo.commu_res['permutation_test_fdr']<0.05].query('Sender != "Adipocytes" and Receiver != "Adipocytes"').shape[0],
    'RT': RT_mebo.commu_res[RT_mebo.commu_res['permutation_test_fdr']<0.05].query('Sender != "Adipocytes" and Receiver != "Adipocytes"').shape[0],
    'Cold2': cold2_mebo.commu_res[(cold2_mebo.commu_res['permutation_test_fdr']<0.05)].shape[0],
    'Cold7': cold7_mebo.commu_res[cold7_mebo.commu_res['permutation_test_fdr']<0.05].query('Sender != "Adipocytes" and Receiver != "Adipocytes"').shape[0]
})
# df = df[['TN', 'RT', 'cold2', 'cold7']]

fig, ax = plt.subplots(figsize = (4,3))
ax.bar(df.index, df, color = '#FFA500', width = .5)
for i,j in [[0, df['TN']],[1, df['RT']],[2, df['Cold2']],[3, df['Cold7']]]:
    ax.text(i, j+5, j, ha = 'center')
ax.set_ylabel('Number of Communication')
sns.despine()
plt.tight_layout()
fig.savefig('../updated_plots/0822/num_of_commu.pdf')
plt.show()



# In[29]:


## circle plot to show communications between cell groups
fig1 = _commu_network_plot_(
                    commu_res = TN_commu_res.query('Sender != "Adipocytes" and Receiver != "Adipocytes"'),
                    sender_focus = [],
                    metabolite_focus = [],
                    sensor_focus = [],
                    receiver_focus = [],
                    remove_unrelevant = False,
                    and_or = 'and',
                    pval_method = 'permutation_test_fdr',
                    pval_cutoff = 0.05,
                    node_cmap = colmap,
                    figsize = (6.1, 3.5),
                    line_cmap = 'bwr',
                    line_color_vmin = None,
                    line_color_vmax = 200,
                    line_width_col = 'Count',
                    linewidth_norm = (0.2, 1),
                    linewidth_range = (1, 13),
                    node_size_norm = (50, 200),
                    node_size_range = (23, 40),
                    adjust_text_pos_node = True,
                    node_text_hidden = False,
                    node_text_font = 10,
                    pdf = None,
                    save_plot = True,
                    show_plot = True,
                    comm_score_col = 'Commu_Score',
                    comm_score_cutoff = None,
                    cutoff_prop = None,
                    text_outline = False,
                    return_fig = True)


fig2 = _commu_network_plot_(
                    commu_res = TN_commu_res.query('Sender != "Adipocytes" and Receiver != "Adipocytes"'),
                    sender_focus = [],
                    metabolite_focus = [],
                    sensor_focus = [],
                    receiver_focus = [],
                    remove_unrelevant = False,
                    and_or = 'and',
                    pval_method = 'permutation_test_fdr',
                    pval_cutoff = 0.05,
                    node_cmap = colmap,
                    figsize = (6.1, 3.5),
                    line_cmap = 'bwr',
                    line_color_vmin = None,
                    line_color_vmax = 200,
                    line_width_col = 'Count',
                    linewidth_norm = (0.2, 1),
                    node_size_norm = (50, 200),
                    linewidth_range = (1, 13),
                    node_size_range = (23, 40),
                    adjust_text_pos_node = True,
                    node_text_hidden = True,
                    node_text_font = 10,
                    pdf = None,
                    save_plot = True,
                    show_plot = True,
                    comm_score_col = 'Commu_Score',
                    comm_score_cutoff = None,
                    cutoff_prop = None,
                    text_outline = False,
                    return_fig = True)
pdf = PdfPages('../updated_plots/0822/TN_network.pdf')
pdf.savefig(fig1)
pdf.savefig(fig2)
pdf.close()



TN_mebo.eventnum_bar(
                    sender_focus=[],
                    metabolite_focus=[],
                    sensor_focus=[],
                    receiver_focus=[],
                    and_or='and',
                    pval_method='permutation_test_fdr',
                    pval_cutoff=0.05,
                    comm_score_col='Commu_Score',
                    comm_score_cutoff = None,
                    cutoff_prop = 0.15,
                    figsize='auto',
                    save='../updated_plots/0822/TN_commu_num.pdf',
                    show_plot=True,
                    include=['sender-receiver'],
                    group_by_cell=True,
                    colorcmap='tab20',
                    return_fig=False
                )





# In[30]:


## circle plot to show communications between cell groups
fig1 = _commu_network_plot_(
                    commu_res = RT_commu_res.query('Sender != "Adipocytes" and Receiver != "Adipocytes"'),
                    sender_focus = [],
                    metabolite_focus = [],
                    sensor_focus = [],
                    receiver_focus = [],
                    remove_unrelevant = False,
                    and_or = 'and',
                    pval_method = 'permutation_test_fdr',
                    pval_cutoff = 0.05,
                    node_cmap = colmap,
                    figsize = (6.1, 3.5),
                    line_cmap = 'bwr',
                    line_color_vmin = None,
                    line_color_vmax = 200,
                    line_width_col = 'Count',
                    linewidth_norm = (0.2, 1),
                    node_size_norm = (50, 200),
                    linewidth_range = (1, 13),
                    node_size_range = (23, 40),
                    adjust_text_pos_node = True,
                    node_text_hidden = False,
                    node_text_font = 10,
                    pdf = None,
                    save_plot = True,
                    show_plot = True,
                    comm_score_col = 'Commu_Score',
                    comm_score_cutoff = None,
                    cutoff_prop = None,
                    text_outline = False,
                    return_fig = True)


fig2 = _commu_network_plot_(
                    commu_res = RT_commu_res.query('Sender != "Adipocytes" and Receiver != "Adipocytes"'),
                    sender_focus = [],
                    metabolite_focus = [],
                    sensor_focus = [],
                    receiver_focus = [],
                    remove_unrelevant = False,
                    and_or = 'and',
                    pval_method = 'permutation_test_fdr',
                    pval_cutoff = 0.05,
                    node_cmap = colmap,
                    figsize = (6.1, 3.5),
                    line_cmap = 'bwr',
                    line_color_vmin = None,
                    line_color_vmax = 200,
                    line_width_col = 'Count',
                    linewidth_norm = (0.2, 1),
                    node_size_norm = (50, 200),
                    linewidth_range = (1, 13),
                    node_size_range = (23, 40),
                    adjust_text_pos_node = True,
                    node_text_hidden = True,
                    node_text_font = 10,
                    pdf = None,
                    save_plot = True,
                    show_plot = True,
                    comm_score_col = 'Commu_Score',
                    comm_score_cutoff = None,
                    cutoff_prop = None,
                    text_outline = False,
                    return_fig = True)
pdf = PdfPages('../updated_plots/0822/RT_network.pdf')
pdf.savefig(fig1)
pdf.savefig(fig2)
pdf.close()



RT_mebo.eventnum_bar(
                    sender_focus=[],
                    metabolite_focus=[],
                    sensor_focus=[],
                    receiver_focus=[],
                    and_or='and',
                    pval_method='permutation_test_fdr',
                    pval_cutoff=0.05,
                    comm_score_col='Commu_Score',
                    comm_score_cutoff = None,
                    cutoff_prop = 0.15,
                    figsize='auto',
                    save='../updated_plots/0822/RT_commu_num.pdf',
                    show_plot=True,
                    include=['sender-receiver'],
                    group_by_cell=True,
                    colorcmap='tab20',
                    return_fig=False
                )





# In[31]:


## circle plot to show communications between cell groups
fig1 = _commu_network_plot_(
                    commu_res = cold7_commu_res.query('Sender != "Adipocytes" and Receiver != "Adipocytes"'),
                    sender_focus = [],
                    metabolite_focus = [],
                    sensor_focus = [],
                    receiver_focus = [],
                    remove_unrelevant = False,
                    and_or = 'and',
                    pval_method = 'permutation_test_fdr',
                    pval_cutoff = 0.05,
                    node_cmap = colmap,
                    figsize = (6.1, 3.5),
                    line_cmap = 'bwr',
                    line_color_vmin = None,
                    line_color_vmax = 200,
                    line_width_col = 'Count',
                    linewidth_norm = (0.2, 1),
                    node_size_norm = (50, 200),
                    linewidth_range = (1, 13),
                    node_size_range = (23, 40),
                    adjust_text_pos_node = True,
                    node_text_hidden = False,
                    node_text_font = 10,
                    pdf = None,
                    save_plot = True,
                    show_plot = True,
                    comm_score_col = 'Commu_Score',
                    comm_score_cutoff = None,
                    cutoff_prop = None,
                    text_outline = False,
                    return_fig = True)


fig2 = _commu_network_plot_(
                    commu_res = cold7_commu_res.query('Sender != "Adipocytes" and Receiver != "Adipocytes"'),
                    sender_focus = [],
                    metabolite_focus = [],
                    sensor_focus = [],
                    receiver_focus = [],
                    remove_unrelevant = False,
                    and_or = 'and',
                    pval_method = 'permutation_test_fdr',
                    pval_cutoff = 0.05,
                    node_cmap = colmap,
                    figsize = (6.1, 3.5),
                    line_cmap = 'bwr',
                    line_color_vmin = None,
                    line_color_vmax = 200,
                    line_width_col = 'Count',
                    linewidth_norm = (0.2, 1),
                    node_size_norm = (50, 200),
                    linewidth_range = (1, 13),
                    node_size_range = (23, 40),
                    adjust_text_pos_node = True,
                    node_text_hidden = True,
                    node_text_font = 10,
                    pdf = None,
                    save_plot = True,
                    show_plot = True,
                    comm_score_col = 'Commu_Score',
                    comm_score_cutoff = None,
                    cutoff_prop = None,
                    text_outline = False,
                    return_fig = True)
pdf = PdfPages('../updated_plots/0822/cold7_network.pdf')
pdf.savefig(fig1)
pdf.savefig(fig2)
pdf.close()



cold7_mebo.eventnum_bar(
                    sender_focus=[],
                    metabolite_focus=[],
                    sensor_focus=[],
                    receiver_focus=[],
                    and_or='and',
                    pval_method='permutation_test_fdr',
                    pval_cutoff=0.05,
                    comm_score_col='Commu_Score',
                    comm_score_cutoff = None,
                    cutoff_prop = 0.15,
                    figsize='auto',
                    save='../updated_plots/0822/cold7_commu_num.pdf',
                    show_plot=True,
                    include=['sender-receiver'],
                    group_by_cell=True,
                    colorcmap='tab20',
                    return_fig=False
                )





# ## to identify cold-sensitive communication, run mebocost by pooling all the cells but label by condition
# 

# In[23]:


### run MEBOCOST by total cells from all conditions
cellall_mebo = mebocost.load_obj(path = '../../mebocost_BAT_v1.0.2/other_cond/scBAT_mebocost_allcond.pk') 

ec = 0.866
mc = 0.141

exp_prop, met_prop = cellall_mebo._check_aboundance_(cutoff_exp = ec,
                                                    cutoff_met = mc)

cellall_mebo.commu_res = cellall_mebo._filter_lowly_aboundant_(pvalue_res = cellall_mebo.original_result.copy(),
                                                            cutoff_prop = 0.15,
                                                            met_prop=met_prop,
                                                            exp_prop=exp_prop)


# In[27]:


cold2_commu_res['label'] = cold2_commu_res['Sender']+'~'+cold2_commu_res['Metabolite_Name']+'~'+cold2_commu_res['Sensor']+'~'+cold2_commu_res['Receiver']
TN_commu_res['label'] = TN_commu_res['Sender']+'~'+TN_commu_res['Metabolite_Name']+'~'+TN_commu_res['Sensor']+'~'+TN_commu_res['Receiver']
RT_commu_res['label'] = RT_commu_res['Sender']+'~'+RT_commu_res['Metabolite_Name']+'~'+RT_commu_res['Sensor']+'~'+RT_commu_res['Receiver']
cold7_commu_res['label'] = cold7_commu_res['Sender']+'~'+cold7_commu_res['Metabolite_Name']+'~'+cold7_commu_res['Sensor']+'~'+cold7_commu_res['Receiver']

all_sig = np.unique(cold2_commu_res['label'].tolist()+TN_commu_res['label'].tolist()+RT_commu_res['label'].tolist()+cold7_commu_res['label'].tolist())



# In[28]:


### exclude communication if sender and receiver in different condition
commu_res = cellall_mebo.commu_res.copy()
commu_res['sender_cond'] = [x.split('~')[-1] for x in commu_res['Sender'].tolist()]
commu_res['receiver_cond'] = [x.split('~')[-1] for x in commu_res['Receiver'].tolist()]
commu_res_new = pd.DataFrame()
for c in commu_res['sender_cond'].unique().tolist():
    tmp = commu_res[(commu_res['sender_cond'] == c) & (commu_res['receiver_cond'] == c)]
    commu_res_new = pd.concat([commu_res_new, tmp])

cellall_mebo.commu_res = commu_res_new

## focus on events at least siginificant in one condition
commu_res_new['sender_cond'] = [x.split('~')[-1] for x in commu_res_new['Sender'].tolist()]
commu_res_new['receiver_cond'] = [x.split('~')[-1] for x in commu_res_new['Receiver'].tolist()]
commu_res_new['label'] = commu_res_new['Sender'].apply(lambda x: x.split('~')[0])+'~'+commu_res_new['Metabolite_Name']+'~'+commu_res_new['Sensor']+'~'+commu_res_new['Receiver'].apply(lambda x: x.split('~')[0])
significant = commu_res_new[commu_res_new['permutation_test_fdr'] < 0.05]
## exclude Adipocytes, since has been talked in one figure specifically
significant = significant[~significant['Sender'].str.startswith('Adipocytes') & 
                         ~significant['Receiver'].str.startswith('Adipocytes')]
significant = significant[significant['label'].isin(all_sig)]
# significant
commu_res_need = commu_res_new[commu_res_new['label'].isin(significant['label'])]

commu_res_need_mat = commu_res_need.pivot_table(index = 'label', columns = 'sender_cond', values = 'Commu_Score')

## Index of despersion
IOD = commu_res_need_mat.apply(lambda row: np.var(row)/np.mean(row), axis = 1).sort_values(ascending=False)
fig, ax = plt.subplots()
ax.scatter(x = range(IOD.shape[0]), y = IOD, s= 1)
ax.vlines(100, *ax.get_ylim())
ax.hlines(np.quantile(IOD, 0.75), *ax.get_xlim())
ax.set_ylim(0, 7)
plt.show()



# In[10]:


## top 100 variable communications
most_var_commu = commu_res_need_mat.loc[IOD[IOD>np.percentile(IOD, 95)].index]
# most_var_commu = commu_res_need_mat.loc[IOD.head(100).index]
most_var_commu = pd.concat([most_var_commu,
                            pd.DataFrame(most_var_commu.index.str.split('~').tolist(),
                                         index = most_var_commu.index, 
                                         columns = ['Sender', 'Met', 'Sensor', 'Receiver'])],
                          axis = 1)


# In[11]:


## K-means for Q1 communication events
top_q1 = IOD[IOD>np.quantile(IOD, 0.95)].index

top_q1_commu = commu_res_need_mat.loc[top_q1,]
top_q1_commu = top_q1_commu.apply(lambda row: (row - np.mean(row))/np.std(row), axis = 1)

# top_q1_commu = pd.concat([top_q1_commu,
#                             pd.DataFrame(top_q1_commu.index.str.split('~').tolist(),
#                                          index = top_q1_commu.index, 
#                                          columns = ['Sender', 'Met', 'Sensor', 'Receiver'])],
#                           axis = 1)


# In[70]:


from sklearn.cluster import KMeans
## determine k
cost =[]
for i in range(1, 11):
    KM = KMeans(n_clusters = i, max_iter = 500)
    KM.fit(top_q1_commu)
    # calculates squared error
    # for the clustered points
    cost.append(KM.inertia_)    
    
# plot the cost against K values
plt.plot(range(1, 11), cost, color ='g', linewidth ='3')
plt.xlabel("Value of K")
plt.ylabel("Squared Error (Cost)")
plt.show() # clear the plot



# In[12]:


## K-means
from sklearn.cluster import KMeans
# top_q1_commu_row = KMeans(n_clusters=5, random_state=1000).fit(top_q1_commu.apply(lambda x: (x-np.mean(x))/np.std(x)))
# out = open('top_q1_commu_row_kmeans.pk', 'wb')
# pk.dump(top_q1_commu_row, out)
# out.close()
out = open('top_q1_commu_row_kmeans.pk', 'rb')
top_q1_commu_row = pk.load(out)
out.close()
## two-column df, index ~ cluster
clusters = pd.Series(top_q1_commu_row.labels_+1,
            index = top_q1_commu.index).T.sort_values()


# In[13]:


top_q1_commu['cluster'] = clusters[top_q1_commu.index.tolist()]
top_q1_commu.index.name = None
top_q1_commu.columns.name = None

# top_q1_commu.to_csv('most_var_comm.tsv',sep = '\t')


# In[43]:


## pattern plot
ddf = commu_res_need_mat.loc[top_q1,].copy()
ddf = ddf.apply(lambda row: (row - np.mean(row))/np.std(row), axis = 1)
ddf['cluster'] = clusters[ddf.index.tolist()]

fig,ax = plt.subplots(nrows = 5, ncols = 1, sharex = True,
                     figsize = (4, 8))

cluster_text = {}
cluster_order =  [1,5,4,3,2]
for i in cluster_order:
    plot_df = ddf.loc[ddf['cluster']==i,
                                        ['TN', 'RT', 'cold2', 'cold7']]
    print(plot_df.shape[0])
    coldf = pd.DataFrame(plot_df.index.str.split('~').tolist(),
            index = plot_df.index, columns = ['Sender', 'Metabolite', 'Sensor', 'Receiver'])[['Sender', 'Receiver']]
    coldf['Sender'] = [colmap.get(x) for x in coldf['Sender'].tolist()]
    coldf['Receiver'] = [colmap.get(x) for x in coldf['Receiver'].tolist()]
    ax[cluster_order.index(i)].plot(plot_df.mean(), marker = 'o', linestyle = 'dashed')
    ax[cluster_order.index(i)].fill_between(['TN', 'RT', 'cold2', 'cold7'],
                                            plot_df.min(), plot_df.max(),alpha=.33, 
                                            color = 'lightgrey', edgecolor = 'none')
    ax[cluster_order.index(i)].set_title('Cluster '+str(cluster_order.index(i)+1))

sns.despine()
plt.tight_layout()
plt.savefig('../updated_plots/0822/most_var_commu_pattern_dot.pdf')
plt.show()


    


# In[22]:


cluster_text = {}
for i in range(1, 6):
    plot_df = top_q1_commu.loc[top_q1_commu['cluster']==i,
                                        ['TN', 'RT', 'cold2', 'cold7']]
    
    coldf = pd.DataFrame(plot_df.index.str.split('~').tolist(),
            index = plot_df.index, columns = ['Sender', 'Metabolite', 'Sensor', 'Receiver'])[['Sender', 'Receiver']]
    coldf['Sender'] = [colmap.get(x) for x in coldf['Sender'].tolist()]
    coldf['Receiver'] = [colmap.get(x) for x in coldf['Receiver'].tolist()]


    g = sns.clustermap(plot_df,
                center = 0,
                square=True,
                linewidth = .7,
#                 z_score = 0,
                cmap = 'bwr',
                row_cluster=True,
                col_cluster=False,
                xticklabels=True,
                yticklabels=True,
                row_colors=coldf,
                vmax = 1.5, 
                vmin = -1.5,
    #             figsize = (10,22),
    #             cbar_pos=(1, .1, .02, .2),
    #             cbar_kws=dict(orientation='horizontal'),
                      )
    cluster_text[str(i)] = [x.get_text() for x in g.ax_heatmap.get_yticklabels()]
    for label in colmap.keys():
        g.ax_col_dendrogram.bar(0, 0, color=colmap[label],
                                label=label, linewidth=0)
#     l1 = g.ax_col_dendrogram.legend(loc="center",
#                                     ncol=2, 
#                                     title = 'Celltype',
#                                     frameon = False,
#                                     bbox_to_anchor=(1.2, .54),
#                                     bbox_transform=gcf().transFigure)

    g.ax_heatmap.set_ylabel('')
    g.ax_heatmap.set_xlabel('')
    g.ax_cbar.set_title('Relative Communication Score')

    plt.show()


# In[28]:


cluster_text_order = cluster_text['1']+cluster_text['5']+cluster_text['4']+cluster_text['3']+cluster_text['2']
plot_df = top_q1_commu.loc[cluster_text_order,
                                        ['TN', 'RT', 'cold2', 'cold7']]

coldf = pd.DataFrame(plot_df.index.str.split('~').tolist(),
            index = plot_df.index, columns = ['Sender', 'Metabolite', 'Sensor', 'Receiver'])[['Sender', 'Receiver']]
coldf['Sender'] = [colmap.get(x) for x in coldf['Sender'].tolist()]
coldf['Receiver'] = [colmap.get(x) for x in coldf['Receiver'].tolist()]


g = sns.clustermap(plot_df,
            center = 0,
            square=True,
            linewidth = .7,
            cmap = 'bwr',
            row_cluster=False,
            col_cluster=False,
            xticklabels=True,
            yticklabels=True,
            row_colors=coldf,
            vmax = 1.5, 
            vmin = -1.5,
            figsize = (12,22),
            cbar_pos=(.1, .9, .2, .02),
            cbar_kws=dict(orientation='horizontal'),
                  )
for label in colmap.keys():
        g.ax_col_dendrogram.bar(0, 0, color=colmap[label],
                                label=label, linewidth=0)
l1 = g.ax_col_dendrogram.legend(loc="center",
                                ncol=2, 
                                title = 'Celltype',
                                frameon = False,
                                bbox_to_anchor=(1.2, .54),
                                bbox_transform=gcf().transFigure)

g.ax_heatmap.set_ylabel('')
g.ax_heatmap.set_xlabel('')
g.ax_cbar.set_title('Relative Communication Score')
plt.tight_layout()
plt.savefig('../updated_plots/0822/most_var_commu_heatmap.pdf')

plt.show()
plt.close()

