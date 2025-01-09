import PCA
sites = ['HEAL',
        'BART',
        'CLBJ',
        'KONZ',
        'NIWO',
        'ONAQ',
        'OSBS',
        'PUUM',
        'SERC',
        'SRER',
        'TALL',
        'TEAK',
        'TOOL',
        'UNDE',
        'WOOD',
        'WREF',
        'YELL']

for site in sites:
  PCA.pca_workflow(site)
