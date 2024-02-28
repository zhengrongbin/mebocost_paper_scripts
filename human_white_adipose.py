#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
from scipy.stats import spearmanr, pearsonr, ranksums, wilcoxon, ttest_ind, chisquare, kstest, mannwhitneyu

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

plt.rcParams.update(plt.rcParamsDefault)
rc={"axes.labelsize": 16, "xtick.labelsize": 12, "ytick.labelsize": 12,
    "figure.titleweight":"bold", #"font.size":14,
    "figure.figsize":(5.5,4.2), "font.weight":"regular", "legend.fontsize":10,
    'axes.labelpad':8, 'figure.dpi':120}
plt.rcParams.update(**rc)





# In[2]:


compass_met_ann = pd.read_csv('/lab-share/Cardio-Chen-e2/Public/rongbinzheng/software/Compass/compass/Resources/Recon2_export/met_md.csv')

compass_rxn_ann = pd.read_csv('/lab-share/Cardio-Chen-e2/Public/rongbinzheng/software/Compass/compass/Resources/Recon2_export/rxn_md.csv')

met_ann = pd.read_csv('/lab-share/Cardio-Chen-e2/Public/rongbinzheng/software/MEBOCOST/data/mebocost_db/common/metabolite_annotation_HMDB_summary.tsv',
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

def _get_commu_res_(mebo_path, compass_folder, cutoff_prop=0, exp_cutoff=0, met_cutoff=0,min_cell_number=20, efflux_cut = 'auto', influx_cut='auto'):
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
                                                    min_cell_number = min_cell_number
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
    comm_res[x1] = [efflux_mat.loc[m,c] if m in efflux_mat.index.tolist() else np.nan for c, m in comm_res[['Sender', 'Metabolite_Name']].values.tolist()]
    comm_res[x2] = [influx_mat.loc[m,c] if m in influx_mat.index.tolist() else np.nan for c, m in comm_res[['Receiver', 'Metabolite_Name']].values.tolist()]
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
    return(mebo_obj, tmp)

        

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


    


# In[3]:


def grouped_obs_mean(adata, group_key, layer=None, gene_symbols=None):
    if layer is not None:
        getX = lambda x: x.layers[layer]
    else:
        getX = lambda x: x.X
    if gene_symbols is not None:
        new_idx = adata.var[idx]
    else:
        new_idx = adata.var_names

    grouped = adata.obs.groupby(group_key)
    out = pd.DataFrame(
        np.zeros((adata.shape[1], len(grouped)), dtype=np.float64),
        columns=list(grouped.groups.keys()),
        index=adata.var_names
    )

    for group, idx in grouped.indices.items():
        X = getX(adata[idx])
        out[group] = np.ravel(X.mean(axis=0, dtype=np.float64))
    return out


# In[3]:


adata = sc.read_10x_mtx(path = './snRNA/normalized/')
meta = pd.read_csv('./snRNA/Hs.metadata.tsv', sep = '\t', index_col = 0)
umap = pd.read_csv('./snRNA/Hs.umap.scp.tsv', sep = '\t', index_col = 0)
adata.obs = meta.reindex(index = adata.obs_names)
adata.obsm['X_umap'] = np.array(umap.reindex(index = adata.obs_names))
adata.write('PMID35296864_snRNA_normalized_exp.h5ad')


# In[48]:


sc.pl.umap(adata = adata, color = 'cell_type__custom')
sc.pl.umap(adata = adata, color = 'depot__ontology_label')
sc.pl.umap(adata = adata, color = 'bmi__group')


# In[47]:


adata.obs.drop_duplicates(['donor_id', 'bmi__group']).groupby(['bmi__group', 'depot__ontology_label'])['donor_id'].count()


# In[49]:


sc.pl.umap(adata = adata[adata.obs['depot__ontology_label']=='omental fat pad'], color = 'cell_type__custom')
sc.pl.umap(adata = adata[adata.obs['depot__ontology_label']=='omental fat pad'], color = 'bmi__group')



# In[51]:


adata[adata.obs['depot__ontology_label']=='omental fat pad'].obs.groupby('bmi__group')['donor_id'].count()


# In[76]:


## set color map
celltype_list = adata.obs['cell_type__custom'].unique().tolist()
asg_id = pd.Series(range(0, len(celltype_list))).astype('str')+': '+pd.Series(celltype_list)
asg_id.index = celltype_list

colmap1 = {x: matplotlib.colors.rgb2hex(plt.cm.get_cmap('tab20')(celltype_list.index(x))) for x in celltype_list}
colmap2 = {x: matplotlib.colors.rgb2hex(plt.cm.get_cmap('tab20')(asg_id.tolist().index(x))) for x in asg_id.tolist()}


# In[77]:


plot_df = adata.obs.copy()
plot_df['UMAP_1'] = adata.obsm['X_umap'][:,0]
plot_df['UMAP_2'] = adata.obsm['X_umap'][:,1]

plot_df['celltype'] = [asg_id[x] for x in plot_df['cell_type__custom'].tolist()]
label_df = plot_df.groupby(['celltype'])[['UMAP_1', 'UMAP_2']].mean()
label_df['label'] = [x.split(': ')[0] for x in label_df.index.tolist()]

pdf = PdfPages('Figures/adipose_bmi_umap.pdf')
fig, ax = plt.subplots(figsize = (7, 4.5))
sns.scatterplot(data = plot_df, x = 'UMAP_1', y = 'UMAP_2',
               hue = 'celltype', edgecolor = 'none',
               palette = colmap2, #palette='tab20', 
                s = 2, alpha = .6, zorder = 100)
for i, line in label_df.iterrows():
    ax.text(line['UMAP_1'], line['UMAP_2'], line['label'], zorder = 100)
sns.despine()
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax.set_title('All')
# plt.grid(zorder = -5)
plt.tight_layout()
pdf.savefig(fig)
plt.show()
## omental fat BMI 20-30
fig, ax = plt.subplots(figsize = (7, 4.5))
sns.scatterplot(data = plot_df[(plot_df['depot__ontology_label']=='omental fat pad') & 
                        (plot_df['bmi__group']=='20-30')], x = 'UMAP_1', y = 'UMAP_2',
               hue = 'celltype', edgecolor = 'none',
               palette=colmap2, s = 2, alpha = .6, zorder = 100)
for i, line in label_df.iterrows():
    ax.text(line['UMAP_1'], line['UMAP_2'], line['label'], zorder = 100)
sns.despine()
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax.set_title('BMI: 20-30')
# plt.grid(zorder = -5)
plt.tight_layout()
pdf.savefig(fig)
plt.show()

## omental fat BMI 40-50
fig, ax = plt.subplots(figsize = (7, 4.5))
sns.scatterplot(data = plot_df[(plot_df['depot__ontology_label']=='omental fat pad') & 
                        (plot_df['bmi__group']=='40-50')], x = 'UMAP_1', y = 'UMAP_2',
               hue = 'celltype', edgecolor = 'none',
               palette=colmap2, s = 2, alpha = .6, zorder = 100)
for i, line in label_df.iterrows():
    ax.text(line['UMAP_1'], line['UMAP_2'], line['label'], zorder = 100)
sns.despine()
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax.set_title('BMI: 40-50')
# plt.grid(zorder = -5)
plt.tight_layout()
pdf.savefig(fig)
plt.show()
pdf.close()


# In[5]:


adata_bmi_low = adata[(adata.obs['depot__ontology_label']=='omental fat pad') & 
                        (adata.obs['bmi__group']=='20-30')]
adata_bmi_high = adata[(adata.obs['depot__ontology_label']=='omental fat pad') & 
                        (adata.obs['bmi__group']=='40-50')]


# In[69]:


dat1 = grouped_obs_mean(adata = adata_bmi_high, group_key = 'cell_type__custom')
dat1.apply(lambda col: np.exp(col)-1).to_csv('PMID35296864_snRNA_ct_avg_exp_unlog_BMI_high.tsv', sep = '\t')

dat2 = grouped_obs_mean(adata = adata_bmi_low, group_key = 'cell_type__custom')
dat2.apply(lambda col: np.exp(col)-1).to_csv('PMID35296864_snRNA_ct_avg_exp_unlog_BMI_low.tsv', sep = '\t')



# ### MEBOCOST result

# In[5]:


## get mebocost result, constrained by compass flux
mebo_path='./adipose_bmi_20-30_40-50_mebocost_res.pk'

cutoff_prop=0.15
exp_cutoff='auto'
met_cutoff='auto'
min_cell_number=20
efflux_cut = 'auto'
influx_cut='auto'

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
                                                min_cell_number = min_cell_number
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
comm_res['sender_cond'] = [x.split('_')[-1] for x in comm_res['Sender'].tolist()]
comm_res['receiver_cond'] = [x.split('_')[-1] for x in comm_res['Receiver'].tolist()]
comm_res = comm_res[comm_res['sender_cond'] == comm_res['receiver_cond']]

## compass
efflux_mat_2030, influx_mat_2030 = _get_compass_flux_(compass_folder = './PMID35296864_snRNA_ct_avg_exp_unlog_BMI_low_res/')
efflux_mat_4050, influx_mat_4050 = _get_compass_flux_(compass_folder = './PMID35296864_snRNA_ct_avg_exp_unlog_BMI_high_res/')
efflux_mat_2030.columns = efflux_mat_2030.columns+'_20-30'
influx_mat_2030.columns = influx_mat_2030.columns+'_20-30'
efflux_mat_4050.columns = efflux_mat_4050.columns+'_40-50'
influx_mat_4050.columns = influx_mat_4050.columns+'_40-50'
efflux_mat = pd.concat([efflux_mat_2030, efflux_mat_4050], axis = 1)
influx_mat = pd.concat([influx_mat_2030, influx_mat_4050], axis = 1)

x1 = 'sender_transport_flux'
x2 = 'receiver_transport_flux'
comm_res[x1] = [efflux_mat.loc[m,c] if m in efflux_mat.index.tolist() else np.nan for c, m in comm_res[['Sender', 'Metabolite_Name']].values.tolist()]
comm_res[x2] = [influx_mat.loc[m,c] if m in influx_mat.index.tolist() else np.nan for c, m in comm_res[['Receiver', 'Metabolite_Name']].values.tolist()]
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

low_commu_res = tmp[tmp['sender_cond']=='20-30']
low_commu_res['Sender'] = ['_'.join(x.split('_')[:-1]) for x in low_commu_res['Sender'].tolist()]
low_commu_res['Receiver'] = ['_'.join(x.split('_')[:-1]) for x in low_commu_res['Receiver'].tolist()]

high_commu_res = tmp[tmp['sender_cond']=='40-50']
high_commu_res['Sender'] = ['_'.join(x.split('_')[:-1]) for x in high_commu_res['Sender'].tolist()]
high_commu_res['Receiver'] = ['_'.join(x.split('_')[:-1]) for x in high_commu_res['Receiver'].tolist()]


# In[6]:


lcell = pd.Series(collections.Counter(mebo_obj.cell_ann.query('bmi__group == "20-30"')['cell_type__custom']))
hcell = pd.Series(collections.Counter(mebo_obj.cell_ann.query('bmi__group == "40-50"')['cell_type__custom']))
comm_cell = np.intersect1d(lcell[lcell>50].index, hcell[hcell>50].index)

low_commu_res_cc = low_commu_res[low_commu_res['Sender'].isin(comm_cell) &
                                low_commu_res['Receiver'].isin(comm_cell)]
high_commu_res_cc = high_commu_res[high_commu_res['Sender'].isin(comm_cell) &
                                high_commu_res['Receiver'].isin(comm_cell)]


# In[79]:


pdf = PdfPages('Figures/BMI_low_VAT_network_plot.pdf')
fig1 = _commu_network_plot_(
                    commu_res = low_commu_res_cc,
                    sender_focus = list(comm_cell),
                    metabolite_focus = [],
                    sensor_focus = [],
                    receiver_focus = list(comm_cell),
                    remove_unrelevant = False,
                    and_or = 'and',
                    pval_method = 'permutation_test_fdr',
                    pval_cutoff = 0.05,
                    node_cmap = colmap1,
                    figsize = (6.4, 3.5),
                    line_cmap = 'RdBu_r',
                    line_color_vmin = None,
                    line_color_vmax = 100,
                    line_width_col = 'Count',
                    linewidth_norm = (0.2, 1),
                    node_size_norm = (50, 200),
                    node_size_range= (11, 27),
                    linewidth_range=(1, 7),
                    adjust_text_pos_node = True,
                    node_text_hidden = False,
                    node_text_font = 10,
                    pdf = pdf,
                    save_plot = True,
                    show_plot = True,
                    comm_score_col = 'Commu_Score',
                    comm_score_cutoff = None,
                    cutoff_prop = None,
                    text_outline = False,
                    return_fig = True)
pdf.close()


# In[80]:


pdf = PdfPages('Figures/BMI_high_VAT_network_plot.pdf')

fig1 = _commu_network_plot_(
                    commu_res = high_commu_res_cc,
                    sender_focus = list(comm_cell),
                    metabolite_focus = [],
                    sensor_focus = [],
                    receiver_focus = list(comm_cell),
                    remove_unrelevant = False,
                    and_or = 'and',
                    pval_method = 'permutation_test_fdr',
                    pval_cutoff = 0.05,
                    node_cmap = colmap1,
                    figsize = (6.4, 3.5),
                    line_cmap = 'RdBu_r',
                    line_color_vmin = None,
                    line_color_vmax = 100,
                    line_width_col = 'Count',
                    linewidth_norm = (0.2, 1),
                    node_size_norm = (50, 200),
                    node_size_range= (11, 27),
                    linewidth_range=(1, 7),
                    adjust_text_pos_node = True,
                    node_text_hidden = False,
                    node_text_font = 10,
                    pdf = pdf,
                    save_plot = True,
                    show_plot = True,
                    comm_score_col = 'Commu_Score',
                    comm_score_cutoff = None,
                    cutoff_prop = None,
                    text_outline = False,
                    return_fig = True)
pdf.close()


# In[18]:


low_comm_event = _make_comm_event_(commu_res = low_commu_res[low_commu_res['Sender'].isin(comm_cell) & low_commu_res['Receiver'].isin(comm_cell)])
high_comm_event = _make_comm_event_(commu_res = high_commu_res[high_commu_res['Sender'].isin(comm_cell) & high_commu_res['Receiver'].isin(comm_cell)])

comb_comm_event = pd.concat([low_comm_event, high_comm_event])
comb_comm_event['label'] = ['BMI_Low'] * low_comm_event.shape[0] + ['BMI_High']*high_comm_event.shape[0]


comp_comm_count = comb_comm_event.pivot_table(index = ['Sender', 'Receiver'], columns = ['label'], values = ['Count']).fillna(0)
comp_comm_count.columns = comp_comm_count.columns.get_level_values(1)
comp_comm_count['delta'] = comp_comm_count['BMI_High'] - comp_comm_count['BMI_Low']
comp_comm_count = comp_comm_count.reset_index().sort_values('delta')



# In[81]:


pdf = PdfPages('Figures/comp_comm_number_barplot.pdf')

fig, axs = plt.subplots(ncols = 1, nrows = comp_comm_count['Sender'].unique().shape[0], 
                      sharex = True, figsize = (6, 12), gridspec_kw = {'hspace':.8})
# receivers = comp_comm_count['Receiver'].unique().tolist()
for x in comm_cell.tolist():
    tmp = comp_comm_count[comp_comm_count['Receiver'] == x]
    tmp = tmp[['Sender', 'BMI_High', 'BMI_Low']].melt(id_vars = ['Sender'])
    tmp['label'] = pd.Categorical(tmp['label'].tolist(), ['BMI_Low', 'BMI_High'])
    ax = axs[comm_cell.tolist().index(x)]
    sns.barplot(data = tmp, x = 'Sender', y = 'value', hue = 'label', ax = ax, order = comm_cell)
    ax.set(xlabel = '', ylabel = '')
    ax.set_title('Receiver: '+x, size = 8)
    sns.despine()
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize = 5)
axs[-1].tick_params(axis = 'x', rotation = 90)
# plt.subplots_adjust(hspace=0)
plt.tight_layout()
pdf.savefig(fig)
plt.show()
plt.close()
pdf.close()



# In[8]:


def _count_dot_plot_(commu_res, 
                        pval_method='permutation_test_fdr', 
                        pval_cutoff=0.05, 
                        cmap='RdBu_r', 
                        figsize = 'auto',
                        pdf = None,
                        dot_size_norm = (5, 100),
                        dot_color_vmin = None,
                        dot_color_vmax = None,
                        show_plot = False,
                        comm_score_col = 'Commu_Score',
                        comm_score_cutoff = None,
                        cutoff_prop = None,
                        cellgroup_order = [],
                        show_dendrogram = False,
                        return_fig = False):
    """
    plot dot plot where y axis is receivers, x axis is senders
    -----------
    commu_res: data frame 
        Cell_Pair  -log10(pvalue)
        Malignant_0->Malignant_0    1.378965
        Malignant_12->Malignant_0   1.378965
        Malignant_1->Malignant_0    1.359778
        CD8Tex_3->Malignant_0   1.347078

    cmap: colormap for dot color representing overall score of communication between cells
    """
    ## clean
    plt.close()
    
#     info('plot dot plot to show communication in cell type level')
    comm_event = _make_comm_event_(commu_res = commu_res,
                                    pval_method = pval_method,
                                    pval_cutoff = pval_cutoff,
                                    comm_score_col = comm_score_col,
                                    comm_score_cutoff = comm_score_cutoff,
                                    cutoff_prop = cutoff_prop
                                  )
    if comm_event is None:
        return
    ## plot setting
    if not dot_color_vmin:
        dot_color_vmin = np.percentile(comm_event['-log10(pvalue)'], 0)
    if not dot_color_vmax:
        dot_color_vmax = np.percentile(comm_event['-log10(pvalue)'], 100)

    ## plot dot plot with dendrogram
    if figsize == 'auto':
        sender_num = comm_event['Sender'].unique().shape[0]
        receiver_num = comm_event['Receiver'].unique().shape[0]
        figsize = (6.5+receiver_num*0.27, 4+sender_num*0.16)
    df = comm_event.pivot_table(index = 'Sender', columns = 'Receiver', values = ['Count']).fillna(0)
    fig, axes = plt.subplots(nrows=2, ncols=4, 
                             gridspec_kw={'height_ratios': [.8, 5],
                                        'width_ratios':[5, 1.8, 0.2, 1.8],
                                        'hspace':0,
                                        'wspace':0},
                            figsize=figsize)
    leftupper = fig.add_subplot(axes[0, 0])
    leftlower = fig.add_subplot(axes[1, 0])
    midupper = fig.add_subplot(axes[0, 1])
    midlower = fig.add_subplot(axes[1, 1])
    rightupper = fig.add_subplot(axes[0, 2])
    rightlower = fig.add_subplot(axes[1, 2])
    sideupper = fig.add_subplot(axes[0, 3])
    sidelower = fig.add_subplot(axes[1, 3])

    ## off axis
    midupper.axis('off')
    rightupper.axis('off')
    rightlower.axis('off')
    leftupper.axis('off')
    midlower.axis('off')
    sideupper.axis('off')
    sidelower.axis('off')
    
    ## main scatter
    if not cellgroup_order and show_dendrogram:
        ## top dendrogram
        linkage_top = linkage(df)
        plt.rcParams['lines.linewidth'] = 1
        den_top = dendrogram(linkage_top, 
                             ax=leftupper, color_threshold=0, above_threshold_color='k')
        leftupper.tick_params(axis = 'x', rotation = 90)
        den_top_order = df.index[[int(x) for x in den_top['ivl']]]

        ## side dendrogram
        linkage_side = linkage(df.T)
        den_side = dendrogram(linkage_side,
                              ax=midlower, color_threshold=0, above_threshold_color='k',
                  orientation='right')
        den_side_order = df.columns.get_level_values(1)[[int(x) for x in den_side['ivl']]]
        ## force order
        x_order = den_top_order.tolist()
        y_order = den_side_order.tolist()
#         comm_event['Sender'] = comm_event['Sender'].astype('category').cat.set_categories(den_top_order)
#         comm_event['Receiver'] = comm_event['Receiver'].astype('category').cat.set_categories(den_side_order)
    elif cellgroup_order:
        x_order = cellgroup_order
        y_order = cellgroup_order
    else:
        pass
    
    ### dot norm
    dot_size_norm_fun = lambda x, y: dot_size_norm[0]+((x-min(y)) / (max(y) - min(y)) * (dot_size_norm[1]-dot_size_norm[0])) if max(y) != min(y) else dot_size_norm[0]+((x-min(y)) / max(y) * (dot_size_norm[1]-dot_size_norm[0])) 
    dot_size = [dot_size_norm_fun(x, comm_event['Count']) for x in comm_event['Count'].tolist()]
    comm_event['dot_size'] = dot_size
#     sp = leftlower.scatter(x = comm_event['Sender'], 
#                            y = comm_event['Receiver'],
#                            c = comm_event['-log10(pvalue)'], 
#                            s = dot_size,
#                            vmax = dot_color_vmax,
#                            vmin = dot_color_vmin,
#                            cmap = cmap)
    for se in x_order:
        for r in y_order:
            tmp = comm_event[(comm_event['Sender'] == se) & (comm_event['Receiver'] == r)]
            if tmp.shape[0] != 0:
                x, y = x_order.index(se), y_order.index(r)
                c, s = tmp['-log10(pvalue)'], tmp['dot_size']
                sp=leftlower.scatter(x = [x], y = [y], c = c, s = s,
                                  vmax = dot_color_vmax, 
                                  vmin = dot_color_vmin,
                                  cmap = cmap)
    leftlower.set_xticks(range(len(x_order)), x_order)
    leftlower.set_yticks(range(len(y_order)), y_order)
    
    leftlower.tick_params(axis = 'x', rotation = 90)
    leftlower.set_xlabel('Sender')
    leftlower.set_ylabel('Receiver')
    

    ## legend for color
    cbar = plt.colorbar(sp, ax = sidelower,
                        location = 'top',
                       shrink = .7)
    cbar.set_label(label = 'Overall Score', 
                   fontsize = 10)

    ## legend for dot size
    dot_values = sorted(list(set([
                 np.min(comm_event['Count']),
#                  np.percentile(comm_event['Count'], 10),
                 np.percentile(comm_event['Count'], 50),
#                  np.percentile(comm_event['Count'], 90),
                 np.max(comm_event['Count'])
                ])))

    for label in dot_values:
        sidelower.scatter([],[],
                color = 'black',
                facecolors='none',
                s = dot_size_norm_fun(label, comm_event['Count']),
                label=int(label))
    sidelower.legend(title = '# of Communication Event', 
                     loc='upper center',
                     fontsize = 10,
                     frameon = False)
    nrow = len(comm_event['Sender'].unique())
    ncol = len(comm_event['Receiver'].unique())
    leftlower.set_xlim(-0.5, nrow-0.5)
    leftlower.set_ylim(-0.5, ncol-0.5)

    plt.tight_layout()
    pdf.savefig(fig) if pdf else None
    if show_plot:
        plt.show()
    plt.close()
    
    if return_fig:
        return(fig)


# In[82]:


celltype_order = ['adipocyte', 'ASPC', 'endothelial', 'LEC', 'SMC', 'pericyte', 'mesothelium',
                  'macrophage', 'monocyte', 'dendritic_cell', 'mast_cell', 'nk_cell', 't_cell']

pdf = PdfPages('Figures/BMI_low_comm_number_dotplot.pdf')
_count_dot_plot_(commu_res=low_commu_res[low_commu_res['Sender'].isin(comm_cell) & low_commu_res['Receiver'].isin(comm_cell)], 
                        pval_method='permutation_test_fdr', 
                        pval_cutoff=0.05, 
                        cmap='RdBu_r', 
                        figsize = 'auto',
                        pdf = pdf,
                        dot_size_norm = (5, 100),
                        dot_color_vmin = None,
                        dot_color_vmax = 110,
                        show_plot = True,
                        comm_score_col = 'Commu_Score',
                        comm_score_cutoff = None,
                        cutoff_prop = None,
                        cellgroup_order = celltype_order,
                        show_dendrogram=False,
                        return_fig = False)
pdf.close()

pdf = PdfPages('Figures/BMI_high_comm_number_dotplot.pdf')
_count_dot_plot_(commu_res=high_commu_res[high_commu_res['Sender'].isin(comm_cell) & high_commu_res['Receiver'].isin(comm_cell)], 
                        pval_method='permutation_test_fdr', 
                        pval_cutoff=0.05, 
                        cmap='RdBu_r', 
                        figsize = 'auto',
                        pdf = pdf,
                        dot_size_norm = (5, 100),
                        dot_color_vmin = None,
                        dot_color_vmax = 110,
                        show_plot = True,
                        comm_score_col = 'Commu_Score',
                        comm_score_cutoff = None,
                        cutoff_prop = None,
                        cellgroup_order = celltype_order,
                        show_dendrogram=False,
                        return_fig = False)
pdf.close()



# In[9]:


low_commu_res_cc['label'] = low_commu_res_cc['Sender']+'~'+low_commu_res_cc['Metabolite_Name']+'~'+low_commu_res_cc['Sensor']+'~'+low_commu_res_cc['Receiver']
high_commu_res_cc['label'] = high_commu_res_cc['Sender']+'~'+high_commu_res_cc['Metabolite_Name']+'~'+high_commu_res_cc['Sensor']+'~'+high_commu_res_cc['Receiver']
comb_comm = pd.concat([low_commu_res_cc[['label', 'Commu_Score', 'permutation_test_fdr']],
                      high_commu_res_cc[['label', 'Commu_Score', 'permutation_test_fdr']]])
comb_comm['group'] = ['BMI_Low']*low_commu_res_cc.shape[0]+['BMI_High']*high_commu_res_cc.shape[0]

comb_comm_score = comb_comm.pivot_table(index = 'label', columns = 'group', values = 'Commu_Score').fillna(0)
comb_comm_score['delta'] = comb_comm_score['BMI_High'] - comb_comm_score['BMI_Low']
comb_comm_score['mean'] = comb_comm_score[['BMI_High', 'BMI_Low']].T.mean()
comb_comm_score = comb_comm_score.sort_values('delta')
comb_comm_score.sort_values('delta')


# In[84]:


with pd.ExcelWriter('Figures/mCCC_BMI_high_low.xlsx') as writer:
    comb_comm_score.sort_values(['delta']).to_excel(writer, sheet_name='comm_score_comp')
    comb_comm.sort_values(['Commu_Score']).to_excel(writer, sheet_name='all')


# In[23]:


### for EC related

# pdf = PdfPages('Figures/comp_comm_score_scatterplot.pdf')
plot_df = pd.DataFrame(comb_comm_score.index.str.split('~').tolist(),
            index = comb_comm_score.index,
            columns = ['Sender', 'Metabolite', 'Sensor', 'Receiver'])
plot_df = pd.concat([plot_df, comb_comm_score], axis = 1)
## no autocrine and between ASPC and adipocyte, since the main purpose is to explore ASPC/Adipocyte with other 
plot_df = plot_df.query('Sender != Receiver')
# plot_df = plot_df.query('~(Sender == "macr" and Receiver == "ASPC") and ~(Sender == "ASPC" and Receiver == "adipocyte")')


df1 = plot_df.query('(Sender == "macrophage" or Receiver == "macrophage") and abs(delta) > 2').sort_values(['delta', 'Sender', 'Receiver'])
df2 = plot_df.query('(Sender == "endothelial" or Receiver == "endothelial") and abs(delta) > 2').sort_values(['delta', 'Sender', 'Receiver'])
df3 = plot_df.query('(Sender == "pericyte" or Receiver == "pericyte") and abs(delta) > 2').sort_values(['delta', 'Sender', 'Receiver'])
df = pd.concat([df1, df2, df3]).drop_duplicates()

fig, ax = plt.subplots(figsize = (10, 10))
sns.heatmap(df[['BMI_High','BMI_Low']], cmap = 'Purples', linewidth = .2, linecolor = 'lightgrey',
           ax = ax)
plt.tight_layout()
fig.savefig('Figures/macrophage_endothlial_pericyte_high_changed_mCCC.pdf')
plt.show()


# In[43]:


## for macrophage related
fig, ax = plt.subplots(figsize = (4, 4))
sns.heatmap(df1.query('delta > 0').sort_values('BMI_High', ascending = False)[['BMI_High','BMI_Low']], 
            cmap = 'Blues', linewidth = .2, linecolor = 'lightgrey', ax = ax)
plt.tight_layout()
fig.savefig('Figures/macrophage_high_changed_mCCC.pdf')
plt.show()



# In[ ]:




