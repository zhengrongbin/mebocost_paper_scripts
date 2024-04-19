import os,sys
import json, re, time
import urllib.request, urllib.parse, urllib.error
import traceback
from datetime import datetime
import pickle
import subprocess
import random
import importlib
import collections
try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET

from xml.dom.minidom import parseString
import pandas as pd
import numpy as np
import xml.dom.minidom

## =========== 1. transporter =========
transporter_met = pd.read_csv('need_curate/transporter_metabolite_pmid_summary_manual_curated.tsv', sep = '\t')
transporter_met = transporter_met[(transporter_met['curate'] == 'T') & 
                                 (~transporter_met['direction'].isin(['Efflux']))]
transporter_met['metName'] = [x.split(';')[0] for x in transporter_met['metabolite_transporter'].tolist()]
transporter_met['transporter'] = [x.split(';')[1] for x in transporter_met['metabolite_transporter'].tolist()]

def _get_hmdb_id_(met):
    compass_metabolite
    if met in hmdb_res['metabolite'].tolist():
        Id = hmdb_res[hmdb_res['metabolite'] == met]['HMDB_ID'].tolist()[0]
    elif met in scFEA_compound_ids_hmdb['metabolite_name'].tolist():
        Id = scFEA_compound_ids_hmdb[scFEA_compound_ids_hmdb['metabolite_name'] == met]['HMDB_ID'].tolist()[0]
        if Id in hmdb_res['Secondary_HMDB_ID'].tolist():
            Id = hmdb_res[hmdb_res['Secondary_HMDB_ID'] == Id]['HMDB_ID'].tolist()[0]
        else:
            Id = None
    else:
        Id = None
    return(Id)

transporter_met['HMDB_ID'] = list(map(_get_hmdb_id_, transporter_met['metName']))

## =========== 2. receptor ========
receptor_met = pd.read_csv('need_curate/receptor_metabolite_pmid_summary_manual_curated.tsv', sep = '\t')
receptor_met = receptor_met[(receptor_met['Curate'] == 'T') & 
                            (~receptor_met['direct'].isin(['Secretion', 'Efflux']))]
receptor_met['metName'] = [x.split(';')[0] for x in receptor_met['metabolite_receptor'].tolist()]
receptor_met['receptor'] = [x.split(';')[1] for x in receptor_met['metabolite_receptor'].tolist()]
receptor_met['HMDB_ID'] = list(map(_get_hmdb_id_, receptor_met['metName']))

## concat transporter and receptor in one dict
met_transp_recp = pd.DataFrame()
##### transporter
for i, line in transporter_met.iterrows():
    hmdbid, metName, transporter, protein, pmid = line['HMDB_ID'], line['metName'], line['transporter'], line['protein_name'], line['pmid']
    standard_metName = hmdb_res[hmdb_res['HMDB_ID'] == hmdbid]['metabolite'].tolist()[0]
    tmp = pd.Series({'HMDB_ID':hmdbid, 'metName':metName, 'gene':transporter+'[Transporter]', 'protein':protein, 'pmid':pmid})
    met_transp_recp = pd.concat([met_transp_recp, tmp], axis = 1)
##### receptor
for i, line in receptor_met.iterrows():
    hmdbid, metName, receptor, protein, pmid, direct = line['HMDB_ID'], line['metName'], line['receptor'], line['protein_name'], line['pmid'], line['direct']
    standard_metName = hmdb_res[hmdb_res['HMDB_ID'] == hmdbid]['metabolite'].tolist()[0]
    Type = 'Transporter' if direct in ['Transport', 'Uptake'] else 'Receptor'
    tmp = pd.Series({'HMDB_ID':hmdbid, 'metName':metName, 'gene':receptor+'[%s]'%Type, 'protein':protein, 'pmid':pmid})
    met_transp_recp = pd.concat([met_transp_recp, tmp], axis = 1)

## === add manual collection
manual_collected_before = pd.read_csv('scFEA_metabolite_transporter_manual_add.txt', sep = '\t')
manual_collected_before = manual_add[(~pd.isna(manual_collected_before['transporter']) | 
                        ~pd.isna(manual_collected_before['receptor'])) &
                       (manual_collected_before['comments'] != 'ER') & 
                       (manual_collected_before['direction'] != 'Efflux')]
manual_collected_before['HMDB_ID'] = [_get_hmdb_id_(x) for x in manual_collected_before['metabolite'].tolist()]

for i, line in manual_collected_before[~pd.isna(manual_collected_before['HMDB_ID'])].iterrows():
    metName = line['metabolite']
    hmdbid = line['HMDB_ID']
    standard_metName = hmdb_res[hmdb_res['HMDB_ID'] == hmdbid]['metabolite'].tolist()[0]
    if not pd.isna(line['transporter']):
        gene = line['transporter']
        protein = uniprot_all.loc[gene,'primary_protein_name']
        gene = gene+'[Transporter]'
        pmid = line['publication'].replace('https://pubmed.ncbi.nlm.nih.gov/', '').rstrip('/')
        tmp = pd.Series({'HMDB_ID':hmdbid, 'metName':metName, 'gene':gene, 'protein':protein, 'pmid':pmid})
        met_transp_recp = pd.concat([met_transp_recp, tmp], axis = 1)
    if not pd.isna(line['receptor']):
        gene = line['receptor']
        protein = uniprot_all.loc[gene,'primary_protein_name']
        gene = gene+'[Receptor]'
        pmid = line['publication'].replace('https://pubmed.ncbi.nlm.nih.gov/', '').rstrip('/')
        tmp = pd.Series({'HMDB_ID':hmdbid, 'metName':metName, 'gene':gene, 'protein':protein, 'pmid':pmid})
        met_transp_recp = pd.concat([met_transp_recp, tmp], axis = 1)

## =========== 3. interacting protein from MIDAS ========
## interactome derived from a bioRxiv paper at https://www.biorxiv.org/content/10.1101/2021.08.28.458030v1
pro_met_interact_midas = pd.read_csv('metabolite-protein-interaction/Jared_Rutter/media-4.txt', sep = '\t')
met_ann_midas = pd.read_csv('metabolite-protein-interaction/Jared_Rutter/media-1.txt', sep = '\t', encoding= 'unicode_escape')

## interacting protein, ingore enzymes, since those could be endogenous reaction
metabolite_associated_gene = pd.read_csv('metabolite_associated_gene_reaction_HMDB_summary.tsv', sep = '\t')

enzymes_all = [x.replace('[Enzyme]', '') for x in metabolite_associated_gene[metabolite_associated_gene['gene'].str.endswith('[Enzyme]')]['gene'].tolist()]
enzymes_all = list(set(enzymes_all))

pro_met_interact_midas = pd.merge(pro_met_interact_midas, met_ann_midas[['Metabolite', 'KEGG_ID', 'HMDB_ID']],
        left_on = 'Metabolite', right_on = 'Metabolite', how = 'left')
good_interact = pro_met_interact_midas[pro_met_interact_midas['HMDB_ID'].isin(hmdb_res['Secondary_HMDB_ID'].tolist()) &
                      (pro_met_interact_midas['q_value'] <= 0.1) & 
                      (pro_met_interact_midas['Log2(fold_change)'] >= np.log2(1.5)) &
                      (~good_interact['Gene'].isin(enzymes_all))]#.sort_values('Log2(corrected_fold_change)')

protein_ann = {}
for p in good_interact['Protein'].unique().tolist():
    pp = p.replace('-', '').upper()
    tmp = uniprot_all[(uniprot_all['Gene'].str.replace('-', '').str.upper() == pp) |
       (uniprot_all['primary_protein_name'].str.replace('-', '').str.upper() == pp)]
    if tmp.shape[0] != 0:
        protein_ann[p] = tmp.iloc[0,:]
# good_interact['gene_name'] = [protein_ann.get(x) if x in protein_ann else np.nan for x in good_interact['Protein'].tolist()]
good_interact = pd.merge(good_interact, pd.DataFrame(protein_ann).T,
                        left_on = 'Protein', right_index = True)
for i, line in good_interact.iterrows():
    metName = line['Metabolite']
    hmdbid = line['HMDB_ID']
    hmdbid = hmdb_res[hmdb_res['Secondary_HMDB_ID'] == hmdbid]['HMDB_ID'].tolist()[0]
    standard_metName = hmdb_res[hmdb_res['HMDB_ID'] == hmdbid]['metabolite'].tolist()[0]
    gene = line['Gene']
    protein = uniprot_all.loc[gene,'primary_protein_name']
    gene = gene+'[Interacting]'
    pmid = 'https://www.biorxiv.org/content/10.1101/2021.08.28.458030v1'
    tmp = pd.Series({'HMDB_ID':hmdbid, 'metName':metName, 'gene':gene, 'protein':protein, 'pmid':pmid})
    met_transp_recp = pd.concat([met_transp_recp, tmp], axis = 1)

## tanspose dataframe
met_transp_recp = met_transp_recp.T


#### standard name of metabolite
met_transp_recp['standard_metName'] = [hmdb_res[hmdb_res['HMDB_ID'] == x]['metabolite'].tolist()[0] for x in met_transp_recp['HMDB_ID'].tolist()]
### unique by standard_metName, genes
met_transp_recp_uniq = {}
for i, line in met_transp_recp.iterrows():
    key = '%s | %s | %s | %s'%(line['HMDB_ID'], line['gene'], line['protein'], line['standard_metName'])
    pmid = line['pmid']
    metName = line['metName']
    if key not in met_transp_recp_uniq:
        met_transp_recp_uniq[key] = [metName, pmid]
    else:
        met_transp_recp_uniq[key][0] = '; '.join(list(set(met_transp_recp_uniq[key][0].split('; ')+[metName])))
        met_transp_recp_uniq[key][1] = '; '.join(list(set(met_transp_recp_uniq[key][1].split('; ')+[pmid])))
met_transp_recp_uniq = pd.DataFrame.from_dict(met_transp_recp_uniq, orient = 'index')
met_transp_recp_uniq = pd.concat([pd.DataFrame([x.split(' | ') for x in met_transp_recp_uniq.index.tolist()]),
                                 pd.DataFrame(met_transp_recp_uniq.values.tolist())], axis = 1)
met_transp_recp_uniq.columns = ['HMBD_ID', 'Gene_name', 'Protein_name', 'standard_metName', 'metName', 'Evidence']

met_transp_recp_uniq.to_csv('metabolite_sensors_annotation.csv', sep = '\t', index = None)


## === for mouse ====

homology = pd.read_csv('/Users/rongbinzheng/Documents/github/Metabolic_Communication_V1/software/data/mascot_db/common/human_mouse_homology_gene_pair.csv')
met_sensor = pd.read_csv('/Users/rongbinzheng/Documents/github/Metabolic_Communication_V1/software/data/mascot_db/human/metabolite_sensors_annotation.tsv',
                        sep = '\t')
    
met_sensor_new = []
for i, line in met_sensor.iterrows():
    gene = line['Gene_name']
    gene_name, gtype = gene.rstrip(']').split('[')
    if gene_name not in homology['human_gene'].tolist():
        continue
    else:
        mouse_gene = homology[homology['human_gene'] == gene_name]['mouse_gene'].tolist()
        for mg in mouse_gene:
            new_gene = '{}[{}]'.format(mg, gtype)
            new_line = line.copy()
            new_line['Gene_name'] = new_gene
            met_sensor_new.append(new_line)
met_sensor_new = pd.DataFrame(met_sensor_new)
met_sensor_new.to_csv('/Users/rongbinzheng/Documents/github/Metabolic_Communication_V1/software/data/mascot_db/mouse/metabolite_sensors_annotation_mouse.tsv',
                     sep = '\t', index = None)


### == update =====
predb = pd.read_csv('db_update_202204/met_sen_curation_update.tsv', sep = '\t')
manual_coll = pd.read_csv('db_update_202204/manual_collect.tsv', sep = '\t')
### add recon collection
recon_coll = pd.read_csv('db_update_202204/Recon2_infflux_met_gene.tsv', sep = '\t')
recon_coll_new = pd.DataFrame()
for i, line in recon_coll.iterrows():
    hmdb_id = line['HMDB_ID']
    standard_metName = line['metabolite']
    metName = line['metName']
    evidence = 'Recon2'
    annotation = 'Transporter'
    genes = line['infflux_genes'].split('; ')
    for g in genes:
        if g in uniprot_all.index.tolist():
            protein = uniprot_all.loc[g, 'primary_protein_name']
            if hmdb_id+'~'+g not in (predb['HMBD_ID']+'~'+predb['Gene_name']).tolist():
                ## add a new
                tmp = pd.DataFrame([hmdb_id, standard_metName, metName, g, protein, evidence, annotation],
                                  index = ['HMBD_ID', 'standard_metName', 'metName',
                                           'Gene_name', 'Protein_name', 'Evidence', 'Annotation']).T
                predb = pd.concat([predb, tmp])
            else: 
                ## add one more evidence if existed
                predb.loc[(predb['HMBD_ID'] == hmdb_id) & (predb['Gene_name'] == g), 'Evidence'] = predb.loc[(predb['HMBD_ID'] == hmdb_id) & (predb['Gene_name'] == g), 'Evidence']+'; Recon2'
            

## add manual collection
manual_coll = pd.read_csv('db_update_202204/manual_collect.tsv', sep = '\t')
for i, line in manual_coll.iterrows():
    standard_metName = line['met']
    metName = standard_metName
    evidence = line['evidence']
    annotation = line['annotation']
    gene = line['gene']
    if standard_metName in need['metabolite'].tolist():
        hmdb_id = need[need['metabolite'] == standard_metName]['HMDB_ID'].tolist()[0]
        protein = uniprot_all.loc[g, 'primary_protein_name']
        if hmdb_id+'~'+gene not in (predb['HMBD_ID']+'~'+predb['Gene_name']).tolist():
            ## add a new
            tmp = pd.DataFrame([hmdb_id, standard_metName, metName, gene, protein, evidence, annotation],
                              index = ['HMBD_ID', 'standard_metName', 'metName',
                                       'Gene_name', 'Protein_name', 'Evidence', 'Annotation']).T
            predb = pd.concat([predb, tmp])
        else: 
            ## add one more evidence if existed
            predb.loc[(predb['HMBD_ID'] == hmdb_id) & (predb['Gene_name'] == g), 'Evidence'] = predb.loc[(predb['HMBD_ID'] == hmdb_id) & (predb['Gene_name'] == g), 'Evidence']+'; '+evidence

### focus on predictable
url = 'https://raw.githubusercontent.com/zhengrongbin/zhengrongbin.github.io/master/filles/met_re.tsv'
reaction = pd.read_csv(url, sep = '\t')
predb_predict = predb[predb['HMBD_ID'].isin(reaction['HMDB_ID'])]
predb_predict.columns = ['HMDB_ID', 'standard_metName', 'metName', 'Gene_name', 'Protein_name',
       'Evidence', 'Annotation']
predb_predict.to_csv('db_update_202204/met_sen_final_updated_20220401.tsv', sep = '\t')

## to mouse
homology = pd.read_csv('/Users/rongbinzheng/Documents/github/Metabolic_Communication_V1/software/data/mascot_db/common/human_mouse_homology_gene_pair.csv')
 
met_sensor_new = []
for i, line in predb_predict.iterrows():
    gene_name = line['Gene_name']
    gtype = line['Annotation']
    if gene_name not in homology['human_gene'].tolist():
        continue
    else:
        mouse_gene = homology[homology['human_gene'] == gene_name]['mouse_gene'].tolist()
        for mg in mouse_gene:
            new_gene = mg#'{}[{}]'.format(mg, gtype)
            new_line = line.copy()
            new_line['Gene_name'] = new_gene
            met_sensor_new.append(new_line)
met_sensor_new = pd.DataFrame(met_sensor_new)
met_sensor_new.to_csv('mouse_met_sen_May-18-2023.tsv', sep = '\t')







