import PCA
import Spectral_richness
import PCA_specrich_null

sites = [
        'BART',
        'HEAL',
        'CLBJ',
        'KONZ',
        'NIWO',
        #'ONAQ', # 0.3 NDVI
        'SERC',
        #'SRER', # 0.3 NDVI
        #'TALL', # 0.45 NDVI
        #'TEAK', # 0.35 NDVI
        'TOOL',
        #'UNDE', # 0.45 NDVI
        'WOOD',
        'WREF',
        'YELL'
       ]

for site in sites:
  #PCA.pca_workflow(site)
  #Spectral_richness.process_spectral_richness(site)
  PCA_specrich_null.pca_specdiv_workflow(site)
