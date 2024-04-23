import os,sys
import time

# pmid_path = sys.argv[1]
os.system('wget -O pmid.txt "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term=%28metabolism%5BTitle%2FAbstract%5D%29&retmax=15000000"')
pmid_path = 'pmid.txt'
for Id in open(pmid_path).readlines():
    Id = Id.rstrip()
    print(Id)
    if os.path.exists('pubmed/%s.txt'%Id):
            continue
    url = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id=%s&rettype=medline&retmode=txt'%Id
    if not os.path.exists('pubmed/%s.txt'%Id):
        cmd = 'wget -O pubmed/%s.txt "%s"'%(Id, url)
        os.system(cmd)
        time.sleep(3)
