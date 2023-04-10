import os, pickle, warnings, itertools
from Connectome_Spatial_Smoothing import CSS as css
from scipy import sparse
import hcpalign_utils as hutils
from hcpalign_utils import sizeof, ospath
import matplotlib.pyplot as plt, hcp_utils as hcp, tkalign_utils as utils
from tkalign import x
from tkalign_utils import full, values2ranks as ranks, regress as reg, identifiability as ident
from joblib import Parallel, delayed
import numpy as np
from scipy.stats import spearmanr as sp

alignfile= 'hcpalign_movie_temp_scaled_orthogonal_5-4-7_TF_0_0_0_FFF_S300_False'

aligner_file = f'{hutils.intermediates_path}/alignpickles/{alignfile}.p'
all_aligners = pickle.load( open( ospath(aligner_file), "rb" )) 

for i in range(10):
    print(x)