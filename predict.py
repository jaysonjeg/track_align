"""
Code to predict behavior such as IQ from MRI data in Human Connectome Project
"""
import numpy as np, pandas as pd
import hcpalign_utils as hutils
from hcpalign_utils import ospath
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
import predict_utils as putils

sub_slice = slice(0,400)
parcellation_string = 'S300' #for de-meaning each parcel, or making a parcel-specific classifier
MSMAll=False
y_quantile_transform = False #quantile transform output data
classifier = 'ridgeCV' #svr, ridge, lasso, ridgeCV, lassoCV

X_impute = False #impute missing values for input data using means method
X_pca_features = False # Retain components explaining 50% of variance of features
X_StandardScaler = False #normalize each input data feature to have mean 0 and variance 1
X_demean_parcelwise = True #subtract the mean value from each parcel

X_use_single_parcel = True #use data from a single cortical parcel
single_parcel = 1 #which parcel


resultsfilepath=ospath(f'{hutils.results_path}/r{hutils.datetime_for_filename()}.txt')
c=hutils.clock()   
resultsfile = open(resultsfilepath, 'w')

try:
    t=hutils.cprint(resultsfile) 

    df=pd.read_csv(hutils.ospath(f'{hutils.intermediates_path}/BehavioralData.csv'))
    df.loc[df['Subject']==179548,'3T_Full_Task_fMRI']  = False #does not have MSMAll data for WM task

    cognitive_measures = ['Flanker_AgeAdj', 'CardSort_AgeAdj', 'PicSeq_AgeAdj', 'ListSort_AgeAdj', 'ProcSpeed_AgeAdj','PicVocab_AgeAdj', 'ReadEng_AgeAdj','PMAT24_A_CR','IWRD_TOT','VSPLOT_TC'] 
    rows_with_cognitive = ~df[cognitive_measures].isna().any(axis=1)
    rows_with_3T_taskfMRI = (df['3T_Full_Task_fMRI']==True)
    rows_with_3T_rsfmri = (df['3T_RS-fMRI_Count']==4)
    rows_with_7T_rsfmri = (df['7T_RS-fMRI_Count']==4)
    rows_with_7T_movie = (df['fMRI_Movie_Compl']==True)
    eligible_rows = rows_with_3T_rsfmri & rows_with_3T_taskfMRI & rows_with_cognitive
    eligible_subjects = [str(i) for i in df.loc[eligible_rows,'Subject']]  
    #eligible_subjects = list(set.intersection(set(hutils.all_subs), set(eligible_subjects)))   
    
    """
    sub_slices = [slice(i, i + 25) for i in range(700, 1000, 25)]
    #sub_slices = [slice(0,3)]
    for sub_slice in sub_slices:
        subs = eligible_subjects[sub_slice]
        print(f'{c.time()} {hutils.memused()} {sub_slice} Get decode start')
        imgs_X, X_string = hutils.get_movie_or_rest_data(subs,'rest_FC',runs=[0,1,2,3],fwhm=0,clean=True,MSMAll=MSMAll,FC_parcellation_string='S1000')
        #imgs_X,X_string = hutils.get_task_data(subs,hutils.tasks,MSMAll=MSMAll)
        print(f'{c.time()} {sub_slice} Get decode end')
    assert(0)   
    """ 

    sub_slice_string = f'sub{sub_slice.start}to{sub_slice.stop}'
    print(sub_slice_string)
    subs = eligible_subjects[sub_slice]

    clustering = hutils.parcellation_string_to_parcellation(parcellation_string)
    parc_matrix = hutils.parcellation_string_to_parcmatrix(parcellation_string)
    pipeline = putils.construct_pipeline(X_impute,X_StandardScaler,X_pca_features,classifier)
    df_subs = hutils.dataframe_get_subs(df,subs)
    y = PCA(n_components=1).fit_transform(df_subs[cognitive_measures]).squeeze().astype(np.float32)
    if y_quantile_transform:
        y = hutils.do_quantile_transform(y)

    for option in ['anat_align','func_align']:
        if option == 'anat_align':
            imgs_X,X_string = hutils.get_task_data(subs,hutils.tasks,MSMAll=MSMAll)
        elif option == 'func_align':
            X_string = 'Aresfcf0123t0S1000_D7tasksf&ms_S300_Tresfcf0123t0S1000sub0to20_G1ffrr_TempScal__0'
            imgs_X = hutils.get_saved_task_data(X_string,subs)
        #imgs_X, X_string = hutils.get_movie_or_rest_data(subs,'rest_FC',runs=[0,1,2,3],fwhm=0,clean=True,MSMAll=MSMAll,FC_parcellation_string='S1000')
        print(f'{c.time()} Get X data end')


        log2str = hutils.logical2str
        output_string = f"{sub_slice_string}_{parcellation_string}_{X_string}_{classifier}_y{log2str[y_quantile_transform]}_X{log2str[X_impute]}{log2str[X_pca_features]}{log2str[X_StandardScaler]}{log2str[X_demean_parcelwise]}_{log2str[X_use_single_parcel]}{single_parcel}"
        t.print(output_string)
            
        if len(imgs_X)==2:
            print('pairwise')
        else:
            t.print(f'{c.time()} template or anat')

            if X_demean_parcelwise:
                imgs_X = Parallel(n_jobs=-1, prefer='threads')(delayed(hutils.standardize_image_parcelwise)(img, clustering, parc_matrix,demean=True, unit_variance=False) for img in imgs_X)
            t.print(f'{c.time()} ?demeaned done')
            if X_use_single_parcel:
                get_single_parcel = lambda img: img[:,clustering==single_parcel]
                imgs_X = Parallel(n_jobs=-1,prefer='threads')(delayed(get_single_parcel)(img) for img in imgs_X)
            t.print(f'{c.time()} ?singleparcel done')

            putils.reshape_and_cross_validate(c, t, pipeline,imgs_X, y)




finally:
    resultsfile.close()