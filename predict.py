"""
Code to predict behavior such as IQ from MRI data in Human Connectome Project
"""
import numpy as np, pandas as pd
import hcpalign_utils as hutils
from hcpalign_utils import ospath
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
import predict_utils as putils

### ADJUSTABLE PARAMETERS ###

sub_slice = slice(50,450)
parcellation_string = 'S300' #for de-meaning each parcel, or making a parcel-specific classifier
MSMAll=False
y_quantile_transform = False #quantile transform output data
classifier = 'ridgeCV' #svr, ridge, lasso, ridgeCV, lassoCV, lassolarsCV, sgd, adaboost, GBR, HistGBR, grouped. Slow ones are SVR, Lasso, GBR
what_feature_groups = None #how feature are grouped. None, 'contrasts','parcels','both'

X_impute = False #impute missing values for input data using means method
X_pca_features = False # Retain components explaining 50% of variance of features
X_StandardScaler = False #normalize each input data feature to have mean 0 and variance 1
X_demean_parcelwise = True #subtract the mean value from each parcel. False if doing 'both' alignment option

which_parcel = 1 #an integer for a specific parcel number (e.g. 1), 'w' for Whole brain, or 'a' for All parcels one at a time
align_options = ['func']
save_r2 = False
n_jobs_cross_val = -1


add_parcelwise_mean = False
X_data_pre_aligned = True
X_data = 'task' #'task', 'rest_FC'. Irrelevant if X_data_pre_aligned

if X_data_pre_aligned:
    ### Folders with pre-aligned input data ###
    #pre_aligned_folder = 'Amovf0t0_D7tasksf&ms_S300_Tmovf0t0sub0to5_G1ffrr_TempScal__0' #test with hutils.all_subs
    #pre_aligned_folder = 'Aresfcf0123t0S1000_D7tasksf&ms_S300_Tresfcf0123t0S1000sub0to20_G1ffrr_TempScal__0'
    #pre_aligned_folder = 'Aresfcf0123t0S1000_D7tasksf&ms_S300_Tresfcf0123t0S1000sub0to20_G1ffrr_TempScal__0CIRCSHIFT'
    pre_aligned_folder = 'Aresfcf0123t0S1000_D7tasksf&ms_S300_Tresfcf0123t0S1000sub0to50_L_TempRidg_alphas[1000]_0'
    #xx
else:
    ### Files with subjects' aligners, that needs to be applied to input data ###
    #func_aligner_folder = 'Amovf0t0_S300_Tmovf0t0sub0to5_G1ffrr_TempScal_'
    func_aligner_folder = 'Aresfcf0123t0S1000_S300_Tresfcf0123t0S1000sub0to50_L_TempRidg_alphas[1000]'
    pass


### SOME CHECKS ###
if add_parcelwise_mean: 
    assert(which_parcel=='w')
    assert('func' in align_options)
    assert(X_demean_parcelwise == False)
if 'both' in align_options: 
    assert(X_demean_parcelwise == False)
    assert(add_parcelwise_mean == False)
if save_r2:
    assert(which_parcel=='a')

resultsfilepath=ospath(f'{hutils.results_path}/r{hutils.datetime_for_filename()}.txt')
c=hutils.clock()   
resultsfile = open(resultsfilepath, 'w')


try:
    t=hutils.cprint(resultsfile) 

    df = hutils.get_hcp_behavioral_data()
    cognitive_measures, rows_with_cognitive, rows_with_3T_taskfMRI, rows_with_3T_rsfmri, rows_with_7T_rsfmri, rows_with_7T_movie = hutils.get_rows_in_behavioral_data()

    """
    df=pd.read_csv(hutils.ospath(f'{hutils.intermediates_path}/BehavioralData.csv'))
    df.loc[df['Subject']==179548,'3T_Full_Task_fMRI']  = False #does not have MSMAll data for WM task

    cognitive_measures = ['Flanker_AgeAdj', 'CardSort_AgeAdj', 'PicSeq_AgeAdj', 'ListSort_AgeAdj', 'ProcSpeed_AgeAdj','PicVocab_AgeAdj', 'ReadEng_AgeAdj','PMAT24_A_CR','IWRD_TOT','VSPLOT_TC'] 
    rows_with_cognitive = ~df[cognitive_measures].isna().any(axis=1)
    rows_with_3T_taskfMRI = (df['3T_Full_Task_fMRI']==True)
    rows_with_3T_rsfmri = (df['3T_RS-fMRI_Count']==4)
    rows_with_7T_rsfmri = (df['7T_RS-fMRI_Count']==4)
    rows_with_7T_movie = (df['fMRI_Movie_Compl']==True)
    """

    eligible_rows = rows_with_3T_rsfmri & rows_with_3T_taskfMRI & rows_with_cognitive
    eligible_subjects = [str(i) for i in df.loc[eligible_rows,'Subject']]  

    #eligible_subjects = hutils.all_subs
    #print('hutils.all_subs !!!!')

    sub_slice_string = f'sub{sub_slice.start}to{sub_slice.stop}'
    print(sub_slice_string)
    subs = eligible_subjects[sub_slice]

    clustering = hutils.parcellation_string_to_parcellation(parcellation_string)
    parc_matrix = hutils.parcellation_string_to_parcmatrix(parcellation_string)
    df_subs = hutils.dataframe_get_subs(df,subs)
    y = PCA(n_components=1).fit_transform(df_subs[cognitive_measures]).squeeze().astype(np.float16)
    if y_quantile_transform:
        y = hutils.do_quantile_transform(y)

    for align_option in align_options:
        if (align_option in ['anat','both']) or ((align_option == 'func') and not(X_data_pre_aligned)) or add_parcelwise_mean: #get original imgs
            t.print(f'{c.time()} get original img start')
            if X_data == 'task':
                original_imgs,X_data_string = hutils.get_task_data(subs,hutils.tasks,MSMAll=MSMAll)
            elif X_data == 'rest_FC':
                original_imgs,X_data_string = hutils.get_movie_or_rest_data(subs,'rest_FC',runs=[0,1,2,3],fwhm=0,clean=True,MSMAll=MSMAll,FC_parcellation_string='S1000',FC_normalize=False)
            t.print(f'{c.time()} get original img end')
        if align_option in ['func','both']: #get aligned imgs
            if X_data_pre_aligned:
                print(f'{c.time()} get pre-aligned imgs start')
                imgs_aligned = hutils.get_pre_aligned_X_data(pre_aligned_folder,subs)
                print(f'{c.time()} get pre-aligned imgs end')
                X_data_aligned_string = pre_aligned_folder 
            else:
                t.print(f'{c.time()} transform img start')
                prefix = ospath(f'{hutils.intermediates_path}/alignpickles3/{func_aligner_folder}')
                imgs_aligned = Parallel(n_jobs=-1,prefer='processes')(delayed(putils.apply_aligner_to_img)(original_img,prefix,sub) for original_img,sub in zip(original_imgs,subs))
                X_data_aligned_string = f"{func_aligner_folder}{X_data_string}"
                t.print(f'{c.time()} transform img end')
            if add_parcelwise_mean:
                imgs_aligned, X_data_aligned_string = putils.func_add_parcelwise_mean(c, t, clustering, original_imgs, imgs_aligned, X_data_aligned_string)
        if align_option == 'anat':
            imgs = original_imgs
            X_string = X_data_string
        elif align_option == 'func':
            imgs = imgs_aligned
            X_string = X_data_aligned_string
        elif align_option == 'both': #horizontally concatenate anat and func aligned task data
            X_string = f"anat&{X_data_aligned_string}"
            imgs = Parallel(n_jobs=-1,prefer='threads')(delayed(np.hstack)([img1,img2]) for img1,img2 in zip(original_imgs,imgs_aligned))
            clustering = np.hstack([clustering,clustering])
            from scipy.sparse import hstack
            parc_matrix = hstack([parc_matrix,parc_matrix])

        log2str = hutils.logical2str
        output_string = f"{X_string}_{sub_slice_string}_{parcellation_string}_{classifier}_y{log2str[y_quantile_transform]}_X{log2str[X_impute]}{log2str[X_pca_features]}{log2str[X_StandardScaler]}{log2str[X_demean_parcelwise]}_{which_parcel}"
        t.print(output_string)

        if len(imgs)==2:
            print('pairwise')
        else:
            t.print(f'{c.time()} template or anat')
            if X_demean_parcelwise:
                imgs = Parallel(n_jobs=-1, prefer='threads')(delayed(hutils.standardize_image_parcelwise)(img, clustering, parc_matrix,demean=True, unit_variance=False) for img in imgs)
                t.print(f'{c.time()} ?demeaned done')

            imgs = np.stack(imgs) #(nsubs,ncontrasts,nvertices)

            feature_groups = putils.get_feature_groups(what_feature_groups, clustering, imgs, which_parcel)
            pipeline = putils.construct_pipeline(X_impute,X_StandardScaler,X_pca_features,classifier,feature_groups)

            if type(which_parcel)==int:
                parc_imgs = imgs[:,:,clustering==which_parcel]
                X = np.reshape(parc_imgs,(len(subs),-1))
                t.print(f'{c.time()} start CV')
                r2_scores = putils.do_prediction(pipeline,X, y,n_jobs=n_jobs_cross_val)
                putils.print_r2_result(t,r2_scores)
            elif which_parcel=='w':
                X = np.reshape(imgs,(len(subs),-1))
                del imgs
                t.print(f'{c.time()} start CV')
                r2_scores = putils.do_prediction(pipeline,X, y,n_jobs=n_jobs_cross_val)
                putils.print_r2_result(t,r2_scores)
            elif which_parcel=='a':
                all_parc_imgs = [imgs[:,:,clustering==parcel] for parcel in range(clustering.max()+1)]
                del imgs
                t.print(f'{c.time()} singleparcel done')
                all_X = [arr.reshape(len(subs),-1) for arr in all_parc_imgs]
                del all_parc_imgs
                t.print(f'{c.time()} X rearrange done') 
                t.print(f'{c.time()} start CV')
                all_r2_scores = np.stack(Parallel(n_jobs=-1,prefer="processes")(delayed(putils.do_prediction)(pipeline,X, y,n_jobs=1) for X in all_X))
                xm = all_r2_scores.mean(axis=1) #mean across folds
                print(f"max accuracy is {xm.max():.3f} for parcel {xm.argmax()}")
                xmp = xm.copy()
                xmp[xmp<0]=0 #set negative r2 values to 0

                if save_r2: #save r2 scores to file
                    hutils.mkdir(hutils.ospath(f'{hutils.intermediates_path}/predict'))
                    np.save(hutils.ospath(f'{hutils.intermediates_path}/predict/{output_string}r2.npy'),all_r2_scores.astype(np.float32))

            t.print(f'{c.time()} end CV\n')
    
finally:
    resultsfile.close()