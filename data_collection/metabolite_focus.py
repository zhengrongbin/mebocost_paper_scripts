import os,sys
import json, re, time
import urllib.request, urllib.parse, urllib.error
import traceback
from datetime import datetime
import pickle
import subprocess
from operator import itemgetter
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
from matplotlib import pyplot as plt

from scipy.stats import spearmanr, pearsonr
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib
import seaborn as sns
from adjustText import adjust_text
from pingouin import partial_corr as pcorr
import networkx as nx
from matplotlib_venn import venn2, venn2_circles, venn2_unweighted
from matplotlib_venn import venn3, venn3_circles
import statsmodels
import statsmodels.api as sm
import scipy
import pickle as pk
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.stats import kde



## ------------- 1. collect metabolites for focus -------------
#### we collect metabolite from two softwares called scFEA and Compass, mainly they collected metabolites from Recon2 and KEGG. In addition, we used HMDB database to annotate those can be located in extracellular space
## scFEA
scFEA_metabolite = pd.read_csv('../scFEA/Human_M168_information.symbols.csv', index_col = 0)

## Compass
compass_metabolite = pd.read_csv('../Compass/recon2_md/met_md.csv')
compass_react_ann = pd.read_csv('./data_collection/Compass/recon2_md/rxn_md.csv', index_col = 0)
compass_met_re_in = {}
compass_met_re_out = {}

for i, line in compass_react_ann.iterrows():
    in_ms = line['rxn_formula'].split(' --> ')[0].split(' + ')
    in_ms = list(set([j.split(' [')[0].split(' * ')[-1].strip() for j in in_ms]))
    out_ms = line['rxn_formula'].split(' --> ')[-1].replace('\nNo genes', '').split(' + ')
    out_ms = list(set([j.split(' [')[0].split(' * ')[-1].strip() for j in out_ms]))
    for m in in_ms:
        if m in compass_met_re_in:
            compass_met_re_in[m] += '; '+i
        else:
            compass_met_re_in[m] = i
    for m in out_ms:
        if m in compass_met_re_out:
            compass_met_re_out[m] += '; '+i
        else:
            compass_met_re_out[m] = i
met_all = list(set(list(compass_met_re_in.keys())+list(compass_met_re_out.keys())))


## ================== collect HMDBID for scFEA metabolites, they have KEGG ID ========

## in order to get the metabolite location in HMDB, we parse KEGG site to get HMDB ID for scFEA metabolite
## for scFEA metabolite, only KEGG ID was provided
compound_ids = {}
for i,x in scFEA_metabolite.iterrows():
    in_name, in_Id, out_name, out_Id = x['Compound_IN_name'], x['Compound_IN_ID'], x['Compound_OUT_name'], x['Compound_OUT_ID']
    if ',' not in in_Id and '+' not in in_Id:
        if in_Id in compound_ids:
            compound_ids[in_Id] += '; '+in_name
        else:
            compound_ids[in_Id] = in_name
    if ',' not in out_Id and '+' not in out_Id:
        if out_Id in compound_ids:
            compound_ids[out_Id] += '; '+out_name
        else:
            compound_ids[out_Id] = out_name
## unique
compound_ids = {x:'; '.join(set(compound_ids[x].split('; '))) for x in compound_ids}
compound_ids = pd.Series(compound_ids)

## HMDB id parser from KEGG
HMDB_id = {}
for Id in compound_ids.index.tolist():
    print(Id)
    url = 'https://www.genome.jp/dbget-bin/get_linkdb?-t+hmdb+cpd:%s'%Id
    urlf = urllib.request.urlopen(url)
    context = urlf.read()
    context = context.decode(encoding='utf-8',errors='ignore')
    urlf.close()
    HMDB_id[Id] = list(set(re.findall('HMDB[0-9]*', context)) - set(['HMDB']))
HMDB_id = pd.Series({x:';'.join(HMDB_id[x]) for x in HMDB_id})
## cat compound Id and HMDB Id
scFEA_compound_ids_hmdb = pd.concat([pd.DataFrame(compound_ids), pd.DataFrame(HMDB_id)], axis = 1).reset_index()
scFEA_compound_ids_hmdb.columns = ['Compound_ID', 'metabolite_name', 'HMDB_ID']
## HMDB00744;HMDB00156 is the same, just keep HMDB00156
scFEA_compound_ids_hmdb.loc[scFEA_compound_ids_hmdb['HMDB_ID'] == 'HMDB00744;HMDB00156','HMDB_ID'] = 'HMDB00156'
## correct name problem for some
scFEA_compound_ids_hmdb.loc[scFEA_compound_ids_hmdb['metabolite_name'] == 'G6P; Glucose-6-phosphate', 'metabolite_name'] = "Glucose-6-phosphate"
scFEA_compound_ids_hmdb.loc[scFEA_compound_ids_hmdb['metabolite_name'] == 'Acetyl-CoA; Acetyl-Coa', 'metabolite_name'] = "Acetyl-CoA"
scFEA_compound_ids_hmdb.loc[scFEA_compound_ids_hmdb['metabolite_name'] == 'Serine; serine', 'metabolite_name'] = "Serine"
scFEA_compound_ids_hmdb.loc[scFEA_compound_ids_hmdb['metabolite_name'] == 'glutamate; Glutamate', 'metabolite_name'] = "Glutamate"
scFEA_compound_ids_hmdb.loc[scFEA_compound_ids_hmdb['metabolite_name'] == 'phenylalanine; Phenylalanine', 'metabolite_name'] = "Phenylalanine"
scFEA_compound_ids_hmdb.loc[scFEA_compound_ids_hmdb['metabolite_name'] == 'Lysine; lysine', 'metabolite_name'] = "Lysine"
scFEA_compound_ids_hmdb.loc[scFEA_compound_ids_hmdb['metabolite_name'] == 'Putrescine; Putresine', 'metabolite_name'] = "Putrescine"
scFEA_compound_ids_hmdb = scFEA_compound_ids_hmdb.drop(20)

scFEA_compound_ids_hmdb.to_csv('scFEA_metabolite_HMDBid.csv', index=None)


## ------------------ 2. functions for parser in HMDB -------------
### hmdb parser for metabolite annotation 
def _hmdb_parser_(hmdbid):
    Hmdbid = ''
    try:
        ## get official HMDB ID
        url = 'https://hmdb.ca/metabolites/%s'%hmdbid
        urlf = urllib.request.urlopen(url)
        context = urlf.read()
        context = context.decode(encoding='utf-8',errors='ignore')
        urlf.close()
        Hmdbid = re.findall('HMDB ID</th><td>HMDB[0-9]*', context)[0].replace('HMDB ID</th><td>', '')
        url = 'https://hmdb.ca/metabolites/%s.xml'%Hmdbid
        urlf = urllib.request.urlopen(url)
        context = urlf.read()
        context = context.decode(encoding='utf-8',errors='ignore')
        urlf.close()
        rt = ET.fromstring(context)
    except:
        os.system('echo ' + hmdbid)
        return(Hmdbid, '', '', {}, {}, [])

    metabolite_name = rt.find('name').text
    synonyms_name = '; '.join([x.text for x in rt.find('synonyms')])
    ## metabolite class
    class_ann = {}
    try:
        class_ann["kingdom"] = rt.find('taxonomy').find('kingdom').text
    except:
        class_ann["kingdom"] = ''
    try:
        class_ann['super_class'] = rt.find('taxonomy').find('super_class').text
    except:
        class_ann['super_class'] = ''
    try:
        class_ann['sub_class'] = rt.find('taxonomy').find('sub_class').text
    except:
        class_ann['sub_class'] = ''
    try:
        class_ann['class'] = rt.find('taxonomy').find('class').text
    except:
        class_ann['class'] = ''
    ## biological location
    bio_loc = {}
    for r in rt.find('ontology').findall('root'):
        for x in r.find('descendants'):
            if x.find('term').text == 'Biological location':
                for i in x.find('descendants').findall('descendant'):
                    term = i.find('term').text
#                     print('1. %s'%term)
                    bio_loc[term] = {}
                    descendant = i.find('descendants').findall('descendant')
                    subterms = ''
                    for d in descendant:
                        dterm = d.find('term').text
#                         print('2. %s'%dterm)
                        synonym = d.find('synonyms').findall('synonym')
                        synonyms = [y.text for y in synonym] if synonym else []
#                         print('3. %s'%synonyms)
                        subterms += ' | '+dterm+' : '+' - '.join(synonyms) if synonyms else ' | '+dterm
                    bio_loc[term] = subterms.rstrip(' - ').lstrip(' | ')
    ## parser for protein associations
    protein_ann = []
    if rt.find('protein_associations'):
        proteins = rt.find('protein_associations').findall('protein') 
        for one in proteins:
            try:
                protein_accession = one.find('protein_accession').text
            except:
                protein_accession = ''
            try:
                gene_name = one.find('gene_name').text
            except:
                gene_name = ''
            try:
                protein_uniprot_id = one.find('uniprot_id').text
            except:
                protein_uniprot_id = ''
            try:
                protein_type = one.find('protein_type').text
            except:
                protein_type = ''
            protein_ann.append({'accession':protein_accession,
                               'gene_name':gene_name,
                               'uniprot':protein_uniprot_id,
                               'type':protein_type})
    
    return(Hmdbid, metabolite_name, synonyms_name, bio_loc, class_ann, protein_ann)

### hmdb parser for annotation for metabolite associated genes
def _parse_reaction_and_function_(hmdbpId):
    try:
        url = 'https://hmdb.ca/proteins/%s'%hmdbpId
        urlf = urllib.request.urlopen(url)
        context = urlf.read()
        context = context.decode(encoding='utf-8',errors='ignore')
        urlf.close()
    except:
        print('problem: ' + hmdbpId)
        return(['', '', ''])

    reactions = re.findall(r'\n                <td>.*</td>', context)
    if reactions:
        reactions = [x.replace('\n                <td>', '').replace('</td>', '') for x in reactions]
    general_function = re.findall(r'<th>General Function</td>\n      <td>.*</td>', context)
    if general_function:
        general_function = [x.replace('<th>General Function</td>\n      <td>', '').replace('</td>', '') for x in general_function]
    specific_function = re.findall(r'<th>Specific Function</td>\n      <td>.*\n</td>', context)
    if specific_function:
        specific_function = [x.replace('<th>Specific Function</td>\n      <td>', '').replace('\n</td>', '') for x in specific_function]
    return(['; '.join(general_function), ' | '.join(reactions), ' | '.join(specific_function)])

## ---------------- 3. parser info for metabolite from HMDB, excutive --------
### metabolites in HMDB ID used for HMDB parser
hmdbid_all = list(set(scFEA_compound_ids_hmdb['HMDB_ID'].dropna().unique().tolist() + compass_metabolite['hmdbID'].dropna().unique().tolist()))

### each metabolite save to json file
for hmdbid in hmdbid_all:
   os.system('echo ' + hmdbid)
   time.sleep(10)
   Hmdbid, metabolite_name, synonyms_name, bio_loc, class_ann, protein_ann = _hmdb_parser_(hmdbid)
   json.dump({'Secondary_HMDB_ID':hmdbid, 'HMDB_ID':Hmdbid, "metabolite":metabolite_name,
          "synonyms_name":synonyms_name,
          'biological_location':bio_loc,
          'class_ann':class_ann,
          'protein_association':protein_ann}, open('hmdb_result/%s.json'%hmdbid, 'w'))

### collect metabolite info from each json file
hmdb_res = pd.DataFrame()
## collect protein association
hmdb_protein_ass = pd.DataFrame()
for hmdb_id in hmdbid_all:
    print(hmdb_id)
    path = os.path.join('hmdb_result', hmdb_id+'.json')
    cont = json.load(open(path))
    cont.update(cont['biological_location'])
    cont.pop('biological_location')
    cont.update(cont['class_ann'])
    cont.pop('class_ann')
    cont = pd.DataFrame.from_dict(cont, orient = 'index')
    hmdb_res = pd.concat([hmdb_res, cont.T])
    ## protein ann
    if cont['protein_association']:
        protein_cont = pd.DataFrame(cont['protein_association'])#.drop(['accession'], axis = 1)
        protein_cont['metabolite_hmdb'] = cont['HMDB_ID']
        protein_cont['metabolite'] = cont['metabolite']
        hmdb_protein_ass = pd.concat([hmdb_protein_ass, protein_cont])

hmdb_protein_ass.to_csv('HMDB_parser_protein_association_result.csv')

### --------------- 4. parser info for metabolite associated protein from HMDB ---------
hmdbps = hmdb_protein_ass['accession'].unique().tolist()
protein_ann = pd.DataFrame()
for hmdbpId in hmdbps:
    res = _parse_reaction_and_function_(hmdbpId)
    protein_ann = pd.concat([protein_ann, pd.DataFrame([hmdbpId]+list(res),
                                                    index = ['HMDBPID', 'general_function', 
                                                             'reaction', 'specific_function']).T])
protein_ann.to_csv('HMDB_parser_protein_association_annotation.csv', index = None)


### ---------------- 5. concat all info to a big table ---------------------
### Kegg ID to HMID
hmdb_kegg_id = {}
for i, line in scFEA_compound_ids_hmdb[['Compound_ID', 'HMDB_ID']].dropna().iterrows():
    compound_id = line['Compound_ID']
    hmdb_id = line['HMDB_ID']
    if hmdb_id in hmdb_kegg_id:
        hmdb_kegg_id[hmdb_id] += ', '+compound_id
    else:
        hmdb_kegg_id[hmdb_id] = compound_id
        
for i, line in compass_metabolite[['keggID', 'hmdbID']].dropna().iterrows():
    compound_id = line['keggID']
    hmdb_id = line['hmdbID']
    if hmdb_id in hmdb_kegg_id:
        hmdb_kegg_id[hmdb_id] += ', '+compound_id
    else:
        hmdb_kegg_id[hmdb_id] = compound_id
## unique
hmdb_kegg_id = {x:', '.join(list(set(hmdb_kegg_id[x].split(', ')))) for x in hmdb_kegg_id}
hmdb_kegg_id['HMDB00618'] = 'C00199' ## remove C03736 

## add `Kegg ID` for metabolite to hmdb info
hmdb_res['Kegg_ID'] = [hmdb_kegg_id.get(x) if x in hmdb_kegg_id else None for x in hmdb_res['Secondary_HMDB_ID'].tolist()]
## add `associated gene` for metabolite 
hmdb_res['associated_gene'] = [', '.join(y) if y else None for y in [[i['gene_name'] for i in x if i['gene_name']] for x in hmdb_res['protein_association'].tolist()]]
hmdb_res = hmdb_res.drop('protein_association', axis = 1)

## from scFEA or Compass
def _check_from_scFEA_or_Compass_(hmdbid):
    _from = []
    if hmdbid in scFEA_compound_ids_hmdb['HMDB_ID'].dropna().tolist():
        _from.append('scFEA')
    if hmdbid in compass_metabolite['hmdbID'].dropna().tolist():
        _from.append('Compass')
    return(';'.join(_from) if _from else None)
hmdb_res['Software_Predicted'] = list(map(_check_from_scFEA_or_Compass_, hmdb_res['Secondary_HMDB_ID']))

### check biological location
def _check_bio_location_(line):
    loc = []
    if 'Extracellular' in str(line['Cell and elements']):
        loc.append('Extracellular')
    if 'Blood' in str(line['Biofluid and excreta']):
        loc.append('Blood')
    if "Cerebrospinal fluid" in str(line['Biofluid and excreta']):
        loc.append('Cerebrospinal fluid')
    if not loc:
        loc = ['Other']
    return(' | '.join(loc))
## we only annotate extracellular, blood, and spinal fluid    
hmdb_res['BioLocation_Summary'] = [_check_bio_location_(line) for i, line in hmdb_res.iterrows()]

## write out
hmdb_res.to_csv('metabolite_annotation_HMDB_summary.tsv', sep = '\t', index = None)

#### ------------------ 6. metabolite ~ reaction ~ gene --------
hmdb_protein_ass = pd.merge(hmdb_protein_ass, protein_ann, left_on = 'accession', right_on = 'HMDBPID', how = 'left')
met_gene_pos = {} ## product
met_gene_neg = {} ## substrate
## gene to metabolite, see the reaction and check whether is product of the reaction or substrate 
for i, line in hmdb_protein_ass[~pd.isna(hmdb_protein_ass['reaction']) & 
                                ~pd.isna(hmdb_protein_ass['gene_name'])].iterrows():
    reaction = line['reaction'].split(' | ')
    metabolite = line['metabolite']
    metabolite_hmdb = line['metabolite_hmdb']
    gtype = line['type']
    gene_name = line['gene_name']
    for r in reaction:
        substrate = r.split(' → ')[0].split(' + ')
        product = r.split(' → ')[1].split(' + ')
        if metabolite in substrate:
            if metabolite+':'+metabolite_hmdb in met_gene_neg:
                met_gene_neg[metabolite+' : '+metabolite_hmdb+' : ' + r] += '; '+gene_name+'['+gtype+']'
            else:
                met_gene_neg[metabolite+' : '+metabolite_hmdb+' : ' + r] = gene_name+'['+gtype+']'
        if metabolite in product:
            if metabolite+':'+metabolite_hmdb in met_gene_pos:
                met_gene_pos[metabolite+' : '+metabolite_hmdb+' : ' + r] += '; '+gene_name+'['+gtype+']'
            else:
                met_gene_pos[metabolite+' : '+metabolite_hmdb+' : ' + r] = gene_name+'['+gtype+']'
        
met_gene_pos = pd.DataFrame([x.split(' : ')+[met_gene_pos[x]] for x in met_gene_pos.keys()])
met_gene_pos['direction'] = 'product'
met_gene_neg = pd.DataFrame([x.split(' : ')+[met_gene_neg[x]] for x in met_gene_neg.keys()])
met_gene_neg['direction'] = 'substrate'

metabolite_associated_gene = pd.concat([met_gene_pos, met_gene_neg])
metabolite_associated_gene.columns = ['metabolite', 'HMDB_ID', 'reaction', 'gene', 'direction']
metabolite_associated_gene = metabolite_associated_gene.sort_values(['metabolite', 'reaction'])

metabolite_associated_gene.to_csv('metabolite_associated_gene_reaction_HMDB_summary.tsv', sep = '\t', index = None)


#### ===== for mouse version =======
homology = pd.read_csv('/Users/rongbinzheng/Documents/github/Metabolic_Communication_V1/software/data/mascot_db/human_mouse_homology_gene_pair.csv')
metabolite_associated_gene = pd.read_csv('/Users/rongbinzheng/Documents/github/Metabolic_Communication_V1/software/data/mascot_db/metabolite_associated_gene_reaction_HMDB_summary.tsv',
                        sep = '\t')

met_gene_new = []
for i, line in met_enzyme.iterrows():
    genes = line['gene'].split('; ')
    mann = []
    for g in genes:
        gene_name, gtype = g.rstrip(']').split('[')
        if gene_name not in homology['human_gene'].tolist():
            continue
        else:
            mouse_gene = homology[homology['human_gene'] == gene_name]['mouse_gene'].tolist()
            for mg in mouse_gene:
                mann.append('{}[{}]'.format(mg, gtype))
    if mann:
        new_line = line.copy()
        new_line['gene'] = '; '.join(mann)
        met_gene_new.append(new_line)
met_gene_new = pd.DataFrame(met_gene_new)

met_gene_new.to_csv('metabolite_associated_gene_reaction_HMDB_summary_mouse.tsv', sep = '\t', index = None)


