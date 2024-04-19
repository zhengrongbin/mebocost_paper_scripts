import os,sys
import re, json
import pandas as pd
import numpy as np

def info(txt):
    os.system('echo "%s"'%txt)

def _parse_abstract_(path):
    res = {}
    ## open the file and save the info in dict
    with open(path) as f:
        for line in f:
            if line == '\n':
                continue
            if not line.startswith(' ') and '- ' in line:
                ## each category was started by xx- xxx
                line = line.rstrip().split('- ')
                key = line[0].strip()
                value = line[1]
            else:
                value = ' '+line.strip()
            ## concat to dict format
            if key in res:
                res[key] += ';{}'.format(value)
            else:
                res[key] = value
    return(res)

def _collect_info_pmid_(pmid, pubmed_path, output, known_metabolite):
    path = os.path.join('{}/{}.txt'.format(pubmed_path, pmid))
    res = _parse_abstract_(path)
    ## check the jornal type
    journal_title = res['JT'].replace('\t', ' ') if 'JT' in res else ''
    publication_type = res['PT'].replace('\t', ' ') if 'PT' in res else ''
    title = res['TI'].replace('\t', ' ') if 'TI' in res else ''
    abstract = res['AB'].replace('\t', ' ').replace(';', ' ') if 'AB' in res else ''
    pub_date = res['DEP'].replace('\t', ' ') if 'DEP' in res else res['LR'].replace('\t', ' ')
    ## iterate to check metabolites
    m_include = []
    content = (title + ' ' + abstract).upper()
    for m in known_metabolite:
        if (' {} '.format(m.upper()) in content or ' {},'.format(m.upper()) in content or ' {}.'.format(m.upper()) in content) and m.upper()+' KINASE' not in content:
            m_include.append(m)
    ## check specific terms: Signaling transduction, transporter, metabolic communication, celluar response
    specific_terms = ['Signal Transduction', 'transporter', 'metabolic communication', 'celluar response']
    sp_include = []
    for sp in specific_terms:
        if ' {} '.format(sp.upper()) in content or ' {},'.format(sp.upper()) in content or ' {}.'.format(sp.upper()) in content:
            sp_include.append(sp)
    sp_include = 'NULL' if not sp_include else ';'.join(sp_include)
    ## write out
    if m_include:
    #     return([pmid, ';'.join(m_include), journal_title, publication_type, pub_date, title, sp_include])
    # else:
    #     return(None)
        ### write out pmid and metabolite that has been mentioned
        out = open(output, 'a')
        out.write('\t'.join([pmid, ';'.join(m_include), journal_title, publication_type, pub_date, title, sp_include])+'\n')
        out.close()

### ==========================
output = 'scFEA_metabolite_pmid.txt'
pubmed_path = './pubmed'
metabolite_path = 'scFEA_extracell_blood_fluid_metabolite.csv'
#
#pmids = [x.split('.txt')[0] for x in os.listdir(pubmed_path)]
### pmids = [x.rstrip() for x in open('pmid_need.txt')]
### pmid = '27729975'
#
### known metabolites
#metabolites = pd.read_csv(metabolite_path)
## km = [x.rstrip() for x in open(metabolite_path)]
#km = metabolites['metabolite'].tolist()
#for pmid in pmids:
#    info(pmid)
#    try:
#        _collect_info_pmid_(pmid, pubmed_path, output, km)
#    except:
#        info('problem %s'%pmid)
#

## === parsing transporter or receptor in metabolite related pmid using protein name
receptor_path = 'cell_membrane_receptor.csv'
transporter_path = 'cell_membrane_transporter.csv'
receptors = pd.read_csv(receptor_path)['gene_name'].unique().tolist()#.gene_name.unique().tolist()
transporters = pd.read_csv(transporter_path)['gene_name'].unique().tolist()#.gene_name.unique().tolist()

metabolite_pmids = [x.rstrip().split('\t') for x in open(output)]


for line in metabolite_pmids:
    pmid = line[0]
    metabolites = line[1].split(';')
    path = os.path.join('{}/{}.txt'.format(pubmed_path, pmid))
    res = _parse_abstract_(path)
    mesh_words = res['OT'] if 'OT' in res else ''
    title = res['TI'] if 'TI' in res else ''
    abstract = res['AB'].replace(';', '') if 'AB' in res else ''
    ## for receptors
    r_include = []
    content = (title + '. ' + abstract + '. ' + mesh_words).upper()
    for recep in receptors:
        if ' {} '.format(recep.upper()) in content or ' {},'.format(recep.upper()) in content or ' {}.'.format(recep.upper()) in content:
            r_include.append(recep)
    ## for transporter
    t_include = []
    content = (title + '. ' + abstract + '. ' + mesh_words).upper()
    for transp in transporters:
        if ' {} '.format(transp.upper()) in content or ' {},'.format(transp.upper()) in content or ' {}.'.format(transp.upper()) in content:
            t_include.append(transp)
    ## direction
    d_include = []
    for dw in ['influx', 'efflux']:
        if ' {} '.format(dw.upper()) in content or ' {},'.format(dw.upper()) in content or ' {}.'.format(dw.upper()) in content:
            d_include.append(dw)

    ## record
    if r_include:
        line.append(';'.join(r_include))
    else:
        line.append('NULL')
    if t_include:
        line.append(';'.join(t_include))
    else:
        line.append('NULL')
    if d_include:
        line.append(';'.join(d_include))
    else:
        line.append('NULL')

    ## check whether metabolite and transporter in the same sentence, but exclude the last one
    content = (title + '. ' + abstract).upper()
    metabolite_get = line[1].split(';')
    receptor_get = line[7].split(';')
    transporter_get= line[8].split(';')
    t_pairs = [] ## metabolite-transporter pair
    t_pairs_sentence = []
    r_pairs = [] ## metabolite-receptor pair
    r_pairs_sentence = []
    for m in metabolites:
        for sentence in content.split('. ')[:-1]:
            if m.upper() in sentence:
                for t in transporter_get:
                    if t == 'NULL':
                        continue
                    if t.upper() in sentence:
                        t_pairs.append('{};{}'.format(m, t))
                        t_pairs_sentence.append(sentence.lower())
                for r in receptor_get:
                    if r == 'NULL':
                        continue
                    if r.upper() in sentence:
                        r_pairs.append('{};{}'.format(m, r))
                        r_pairs_sentence.append(sentence.lower())

    if r_pairs:
        line.append(' | '.join(r_pairs))
       # line.append(' | '.join(list(set(r_pairs_sentence))))
        line.append(' | '.join(r_pairs_sentence))
    else:
        line.append('NULL')
        line.append('NULL')
    if t_pairs:
        line.append(' | '.join(t_pairs))
        #line.append(' | '.join(list(set(t_pairs_sentence))))
        line.append(' | '.join(t_pairs_sentence))
    else:
        line.append('NULL')
        line.append('NULL')

    if r_include or t_include:
        ## write out
        print(pmid)
        out = open(output+'_transporter_receptor_direction_gene_name.txt', 'a')
        out.write('\t'.join(line)+'\n')
        out.close()


## === parsing transporter or receptor in metabolite related pmid using protein name
receptor_path = 'cell_membrane_receptor.csv'
transporter_path = 'cell_membrane_transporter.csv'
receptors = pd.read_csv(receptor_path)['protein_name'].unique().tolist()#.gene_name.unique().tolist()
transporters = pd.read_csv(transporter_path)['protein_name'].unique().tolist()#.gene_name.unique().tolist()

metabolite_pmids = [x.rstrip().split('\t') for x in open(output)]


for line in metabolite_pmids:
    pmid = line[0]
    metabolites = line[1].split(';')
    path = os.path.join('{}/{}.txt'.format(pubmed_path, pmid))
    res = _parse_abstract_(path)
    mesh_words = res['OT'] if 'OT' in res else ''
    title = res['TI'] if 'TI' in res else ''
    abstract = res['AB'].replace(';', '') if 'AB' in res else ''
    ## for receptors
    r_include = []
    content = (title + '. ' + abstract + '. ' + mesh_words).upper()
    for recep in receptors:
        if ' {} '.format(recep.upper()) in content or ' {},'.format(recep.upper()) in content or ' {}.'.format(recep.upper()) in content:
            r_include.append(recep)
    ## for transporter
    t_include = []
    content = (title + '. ' + abstract + '. ' + mesh_words).upper()
    for transp in transporters:
        if ' {} '.format(transp.upper()) in content or ' {},'.format(transp.upper()) in content or ' {}.'.format(transp.upper()) in content:
            t_include.append(transp)
    ## direction
    d_include = []
    for dw in ['influx', 'efflux']:
        if ' {} '.format(dw.upper()) in content or ' {},'.format(dw.upper()) in content or ' {}.'.format(dw.upper()) in content:
            d_include.append(dw)

    ## record
    if r_include:
        line.append(';'.join(r_include))
    else:
        line.append('NULL')
    if t_include:
        line.append(';'.join(t_include))
    else:
        line.append('NULL')
    if d_include:
        line.append(';'.join(d_include))
    else:
        line.append('NULL')

    ## check whether metabolite and transporter in the same sentence, but exclude the last one
    content = (title + '. ' + abstract).upper()
    metabolite_get = line[1].split(';')
    receptor_get = line[7].split(';')
    transporter_get= line[8].split(';')
    t_pairs = [] ## metabolite-transporter pair
    t_pairs_sentence = []
    r_pairs = [] ## metabolite-receptor pair
    r_pairs_sentence = []
    for m in metabolites:
        for sentence in content.split('. ')[:-1]:
            if m.upper() in sentence:
                for t in transporter_get:
                    if t == 'NULL':
                        continue
                    if t.upper() in sentence:
                        t_pairs.append('{};{}'.format(m, t))
                        t_pairs_sentence.append(sentence.lower())
                for r in receptor_get:
                    if r == 'NULL':
                        continue
                    if r.upper() in sentence:
                        r_pairs.append('{};{}'.format(m, r))
                        r_pairs_sentence.append(sentence.lower())

    if r_pairs:
        line.append(' | '.join(r_pairs))
       # line.append(' | '.join(list(set(r_pairs_sentence))))
        line.append(' | '.join(r_pairs_sentence))
    else:
        line.append('NULL')
        line.append('NULL')
    if t_pairs:
        line.append(' | '.join(t_pairs))
#        line.append(' | '.join(list(set(t_pairs_sentence))))
        line.append(' | '.join(t_pairs_sentence))
    else:
        line.append('NULL')
        line.append('NULL')

    if r_include or t_include:
        ## write out
        print(pmid)
        out = open(output+'_transporter_receptor_direction_protein_name.txt', 'a')
        out.write('\t'.join(line)+'\n')
        out.close()


