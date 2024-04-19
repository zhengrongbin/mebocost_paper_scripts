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

### === uniprot protein annotation
## common
uniprot_all = pd.read_csv('../../data_collection/UniProt/uniprot-filtered-reviewed_yes+AND+organism__Homo+sapiens+(Human)+[96--.tab',
                     sep = '\t')
uniprot_all = uniprot_all[~uniprot_all.iloc[:,8].duplicated()]
uniprot_all['primary_protein_name'] = [x.split(' (')[0] for x in uniprot_all['Protein names'].tolist()]
uniprot_all = uniprot_all.iloc[:,[8, 7, 10]]
uniprot_all.columns = ['Gene', 'location', 'primary_protein_name']
uniprot_all.index = uniprot_all.loc[:,'Gene'].tolist()


## -------------------- transporter -------------------- 
#### we collect transporters from different resources
# * TCDB at https://www.tcdb.org/hgnc_explore.php
# * HMDB at https://hmdb.ca/downloads, protein data were downloaded and the ones labeled as Transporter will be extracted

# ============= 1. transporter from HMDB ================
## tidy all the protein in HMDB
path = '../Human_Metabolome_Database/hmdb_proteins.xml'
tree = ET.parse(path)
root = tree.getroot()

hmb_protein_res = []
for child in root:
    gene_name = child.find('{http://www.hmdb.ca}gene_name').text
    protein_type = child.find('{http://www.hmdb.ca}protein_type').text
    general_function = child.find('{http://www.hmdb.ca}general_function').text
    metabolite_associations = child.find('{http://www.hmdb.ca}metabolite_associations')
    accession = []
    metabolites = []
    for metabolite in metabolite_associations:
        accession.append(metabolite.find('{http://www.hmdb.ca}accession').text)
        metabolites.append(metabolite.find('{http://www.hmdb.ca}name').text)
    collect = [gene_name, protein_type, general_function, ' | '.join(accession), ' | '.join(metabolites)]
    hmb_protein_res.append(collect)
hmb_protein_res = pd.DataFrame(hmb_protein_res)
hmb_protein_res.columns = ['gene_name', 'protein_type', 'general_function', 'metabolite_accession', 'metabolite']
# hmb_protein_res.to_csv('../Human_Metabolome_Database/hmdb_proteins_clean.csv', index = None)


## ============= 2. transporter from TCDB ================
tcdb_transporter = pd.read_csv('./TCDB_transporters.csv')
## =========== merge transporters ========
transporter = pd.DataFrame(list(set(hmdb_transporter['gene_name'].tolist()+tcdb_transporter['Gene'].tolist())),
            columns = ['gene_name'])

## ============= cell membrane transporter, by checking protein location =======
transporter = pd.merge(transporter, uniprot_all[['Gene', 'location']], left_on = 'gene_name', right_on = 'Gene').drop('Gene', axis = 1)
transporter['cell_membrane'] = transporter['location'].str.contains('Cell membrane')
transporter['protein_name'] = uniprot_all.loc[transporter['gene_name'].tolist(),'primary_protein_name'].tolist()
cell_membrane_transporter = transporter[transporter['cell_membrane'] == True]
## write out
cell_membrane_transporter.to_csv('cell_membrane_transporter.csv',index = None)

## -------------------- receptor -------------------- 

# ## receptor
# #### we collect receptors from  two parts
# * search `receptor` in uniprot, produces the result from link at https://www.uniprot.org/uniprot/?query=name%3Areceptor+reviewed%3Ayes+organism%3A%22Homo+sapiens+%28Human%29+%5B9606%5D%22&sort=score
# * get receptor list from NichNet which collects one of the most comprehensive ligand-receptor resources 

## uniprot
receptor_uniprot = pd.read_csv('../UniProt/uniprot-name_receptor+reviewed_yes+organism__Homo+sapiens+(Human)+%5--.tab',
                              sep = '\t')

## nichenet: collect recetpors from KEGG and other databases
receptor_nichenet = pd.read_csv('../NicheNet/receptor_all.txt', sep = '\t', header = None)

## merge
receptor = pd.DataFrame(list(set(receptor_uniprot.iloc[:,7].tolist()+receptor_nichenet[0].tolist())),
                       columns = ['gene_name']).dropna()
## locate at cell membrane
receptor = pd.merge(receptor, uniprot_all[['Gene', 'location']], left_on = 'gene_name', right_on = 'Gene').drop('Gene', axis = 1)
receptor['cell_membrane'] = receptor['location'].str.contains('Cell membrane')
receptor['protein_name'] = uniprot_all.loc[receptor['gene_name'].tolist(),'primary_protein_name'].tolist()
cell_membrane_receptor = receptor[receptor['cell_membrane'] == True]
## write out
cell_membrane_receptor.to_csv('cell_membrane_receptor.csv',index = None)









