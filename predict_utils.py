from joblib import Parallel, delayed
import numpy as np
import pandas as pd


def get_and_cache_data():
    """
    Get data from HCP and cache it in the intermediates folder
    """
    import hcpalign_utils as hutils
    c=hutils.clock()  
    df = hutils.get_hcp_behavioral_data()
    cognitive_measures, rows_with_cognitive, rows_with_3T_taskfMRI, rows_with_3T_rsfmri, rows_with_7T_rsfmri, rows_with_7T_movie = hutils.get_rows_in_behavioral_data()
    eligible_rows = rows_with_3T_rsfmri & rows_with_3T_taskfMRI
    eligible_subjects = [str(i) for i in df.loc[eligible_rows,'Subject']]  
    print(len(eligible_rows))
    sub_slices = [slice(i, i + 10) for i in range(0, 1200, 10)]
    for sub_slice in sub_slices:
        subs = eligible_subjects[sub_slice]
        print(f'{c.time()} {hutils.memused()} {sub_slice} Get data start')
        imgs_X, X_string = hutils.get_movie_or_rest_data(subs,'rest_FC',runs=[0,1,2,3],fwhm=0,clean=True,MSMAll=False,FC_parcellation_string='S1000')
        #imgs_X,X_string = hutils.get_task_data(subs,hutils.tasks,MSMAll=MSMAll)
        print(f'{c.time()} {sub_slice} Get data end')


def get_task_data_and_align(subs, MSMAll, func_align_file, c, t):
    import hcpalign_utils as hutils
    from hcpalign_utils import ospath
    t.print(f'{c.time()} get aligners: start')
    original_imgs,original_imgs_string = hutils.get_task_data(subs,hutils.tasks,MSMAll=MSMAll)
    X_string = f"{func_align_file}{original_imgs_string}"
    t.print(f'{c.time()} get original img end, transform img start')
    prefix = ospath(f'{hutils.intermediates_path}/alignpickles3/{func_align_file}')
    imgs = Parallel(n_jobs=-1,prefer='processes')(delayed(apply_aligner_to_img)(original_img,prefix,sub) for original_img,sub in zip(original_imgs,subs))
    t.print(f'{c.time()} transform img end')
    return imgs, X_string


import pickle
import warnings
from sklearn.exceptions import InconsistentVersionWarning
def apply_aligner_to_img(original_img,prefix,sub):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
        load_sub_aligner = lambda sub: pickle.load(open(f'{prefix}/{sub}.p',"rb"))
        aligner = load_sub_aligner(sub)
    return aligner.transform(original_img)

def get_regress_function(classifier):
    from sklearn.svm import SVR
    from sklearn.linear_model import Ridge, RidgeCV, Lasso, LassoCV
    if classifier=='svr':
        reg = SVR(kernel='linear',C=1) 
    elif classifier=='ridge':
        reg = Ridge(alpha=1.0)
    elif classifier=='ridgeCV':
        reg = RidgeCV(alphas=(1e-4,1e-3,1e-2,1e-1,1e0,1e1,1e2,1e3,1e4))
    elif classifier=='lasso':
        reg = Lasso(alpha=1.0)
    elif classifier=='lassoCV':
        reg = LassoCV(n_alphas=10)    
    return reg

def construct_pipeline(X_impute,X_StandardScaler,X_pca_features,classifier):
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.pipeline import make_pipeline
    pipelist = []
    if X_impute: 
        pipelist.append(SimpleImputer(missing_values=np.nan, strategy='mean'))
    if X_StandardScaler: 
        pipelist.append(StandardScaler())
    if X_pca_features: 
        pipelist.append(PCA(n_components=0.5))
    pipelist.append(get_regress_function(classifier))
    pipeline = make_pipeline(*pipelist)
    return pipeline

def reshape(list_of_arrays):
    reshape = lambda arr: arr.reshape(1,-1)
    return np.vstack(Parallel(n_jobs=-1,prefer="threads")(delayed(reshape)(arr) for arr in list_of_arrays))

def do_prediction(pipeline,X, y, n_jobs=-1):
    from sklearn.model_selection import cross_val_score, KFold
    kf = KFold(n_splits=10, shuffle=True, random_state=0)
    r2_scores = cross_val_score(pipeline, X, y, cv=kf, scoring='r2',n_jobs=n_jobs)
    return r2_scores

def print_r2_result(t,r2_scores):
    t.print(f'r2 mean {np.mean(r2_scores):.3f}: {[round(i,3) for i in r2_scores]}')

