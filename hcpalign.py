if __name__=='__main__':
    import nibabel as nib
    nib.imageglobals.logger.level = 40  #suppress pixdim error msg
    import hcp_utils as hcp
    import numpy as np, pandas as pd
    import os, itertools
    from sklearn.svm import LinearSVC
    
    import hcpalign_utils as hutils
    from Connectome_Spatial_Smoothing import CSS as css
    from joblib import Parallel, delayed
    from sklearn.model_selection import KFold
    import psutil
    import matplotlib.pyplot as plt

    from fmralign.template_alignment import TemplateAlignment
    from fmralign.surf_pairwise_alignment import SurfacePairwiseAlignment

    import generic_utils as gutils

    hcp_folder=hutils.hcp_folder
    intermediates_path=hutils.intermediates_path
    results_path=hutils.results_path
    available_movies=hutils.movies 
    available_rests=hutils.rests
    available_tasks=hutils.tasks 

    def align_and_classify(\
        c,t,verbose,save_string, subs, imgs_align, imgs_decode, n_bags=1, n_jobs=-1,MSMAll=False, method='pairwise',alignment_method='scaled_orthogonal',alignment_kwargs={}, per_parcel_kwargs={}, gamma=0, absValueOfAligner=False, scramble_aligners=False,post_decode_fwhm=0, imgs_template=None, align_template_to_imgs=False, lowdim_template=False,n_bags_template=1, gamma_template=0,
        args_template={'n_iter':1,'do_level_1':False,'normalize_imgs':None,'normalize_template':None,'remove_self':False,'level1_equal_weight':False},\
        kfolds=5,load_pickle=False, save_pickle=False,do_classification=True,\
        plot_type='open_in_browser', plot_impulse_response=False, plot_contrast_maps=False, plot_scales=False,plot_isc=False,return_aligner=False,imgs_decode_meanstds=None):
        """
        c: a hcpalign_utils.clock object
        t: If a hcpalign_utils.cprint object is given here, will print to both console and text file 
        verbose: bool
            True to print progress, memory usage
        save_string: string
            string for saving outputs
        subs: list of subject IDs
        imgs_align: list of alignment data (subs_indices) (nvertices,nsamples) for each subject
        imgs_decode: list of decoding data (subs_indices) (ncontrasts,nvertices) for each subject
        n_bags: integer, optional (default = 1)
            Number of bags to use for bagging. If n_bags > 1, then make n_bags bootstrap resamples (sampling rows) of source and target image data. Each bag produces a different alignment matrix which are subsequently averaged. 
        n_jobs: processor cores for a single source to target pairwise aligment (1 parcel per thread) (my PC has 12), ie for the 'inner loop'. -1 means use all cores
        MSMAll: False for MSMSulc
        method: anat, intra_subject, pairwise, template
        alignment_method: scaled_orthogonal, permutation, optimal_tarnsport, ridge_cv
        alignment_kwargs: dict
            Additional keyword arguments to pass to the alignment method
        per_parcel_kwargs: dict
            extra arguments, unique value for each parcel. Dictionary of keys (argument name) and values (list of values, one for each parcel) For each parcel, the part of per_parcel_kwargs that applies to that parcel will be added to alignment_kwargs
        gamma: regularization parameter for surf_pairwise_alignment: default 0
        absValueOfAligner: to take elementwise absolute value of the aligner
        scramble_aligners:  to transform subjects' task maps using someone else's aligner  
        post_decode_fwhm smooths the left-out contrast maps predicted through alignment
        imgs_template: list (nsubjects) of alignment data (nsamples,nvertices) for each subject who is used to make template for alignment
        align_template_to_imgs: bool
            True to align from template to imgs_align, rather than vice versa
        lowdim_template: bool
            True to use PCA/ICA to make template
        args_template: dict
            Passed to TemplateAlignment
        kfolds: default 5 (for classification)
        load_pickle: bool
            to load precalculated aligner
        save_pickle: bool
            to save aligner as pickle object
        do_classification: bool
            Whether to do classification of decode data (or just ISC)
        plot_type: 'open_in_browser' or 'save_to_html'
        plot_impulse_response #to plot aligned image of circular ROIs
        plot_contrast_maps #to plot task contrast maps (predicted through alignment)
        plot_scales #to plot scale parameter for Procrustes aligner
        plot_isc: bool
            to plot inter-subject correlation
        args_diffusion: only relevant if align_with='diffusion'
            sift2: True or False
            tckfile e.g. 'tracks_5M' (default with sift2=True),'tracks_5M_sift1M.tck' (default with sift2=False) 'tracks_5M_sift1M.tck' 'tracks_5M_50k.tck', 'tracks_5M_sift1M_200k.tck'
            diffusion_targets_nparcs: False, or a number of parcels [100,300,1000,3000,10000] Number of connectivity targets for DA     
            diffusion_targets_nvertices: number of random vertices to use as targets for diffusion alignment
            fwhm_circ: 
    imgs_decode_meanstds: None or array of shape (nsamples,nparcels*2)
        Mean value and standard deviation for each parcel in each sample, to add to decode data before decoding
        """
        
        #so that print will print to text file too
        if t is not None:
            print=t.print 
        def vprint(string):
            if verbose: print(string)
        
        n_subs=len(subs) #number of subjects
        subs_indices = np.arange(len(subs)) #list from 0 to n_subs-1
        labels = [np.array(range(i.shape[0])) for i in imgs_decode]  
        post_decode_smooth=hutils.make_smoother_100610(post_decode_fwhm)

        #Set up for saving pickled data and for plotting
        if alignment_method=='optimal_transport':
            import dill
            pickle=dill
        else:
            import pickle      
        hutils.mkdir(f'{intermediates_path}/alignpickles')
        save_pickle_filename=gutils.ospath(f'{intermediates_path}/alignpickles/{save_string}.p')                          
        if plot_impulse_response or plot_contrast_maps or plot_scales or plot_isc:
            plot_dir=f'{results_path}/figures/hcpalign/{save_string}'
            p=hutils.surfplot(plot_dir,plot_type=plot_type)
        if load_pickle: assert(os.path.exists(gutils.ospath(save_pickle_filename)))

        #Get parcellation and classifier
        if parcellation_string[0]=='I': assert(MSMAll==True)
        clustering = hutils.parcellation_string_to_parcellation(parcellation_string)
        classifier=LinearSVC(max_iter=10000,dual='auto')     
        n_splits=min(kfolds,n_subs) #for cross validation
        kf=KFold(n_splits=n_splits)
        classification_scores=[]

        ###PAIRWISE ALIGNMENT###
        if method == 'pairwise':   
            vprint(hutils.memused())    
            if load_pickle:
                vprint('loading aligners')
                aligners = pickle.load(open(gutils.ospath(save_pickle_filename), "rb" ))
            else:      
                vprint(f'{c.time()}: Calculate aligners')   
                aligners = hutils.get_all_pairwise_aligners(subs,imgs_align,alignment_method,clustering,n_bags,n_jobs,alignment_kwargs,per_parcel_kwargs,gamma,absValueOfAligner)
                if save_pickle: 
                    pickle.dump(aligners,open(save_pickle_filename,"wb"))
            vprint(hutils.memused())  
            vprint(f'{c.time()}: Aligning decode data start')
            all_aligned_sources_decode = hutils.transform_all_decode_data(subs,imgs_decode,aligners,post_decode_smooth,None,stack=False) #list (ntargetsubjects) of list (remaining source subjects) of arrays (nsamples,nvertices)
            vprint(hutils.memused())  

            #Pearson correlation between subjects' brain maps. Given any 2 source subjects aligned to a target subject, we can find the pearson correlation between their brain maps (whole-brain or within-parcel)
            all_aligned_sources_decode_parcel = [[[array[:,clustering==i] for array in list_of_arrays]  for i in np.unique(clustering)] for list_of_arrays in all_aligned_sources_decode] # list (ntargetsubjects) of list (nparcels) of list(nsourcesubjects) of arrays (nsamples,nvertices in each parcel)
            vprint(f'{c.time()} CORRS: get corrs per parcel start')
            imgs_decode_aligned_parcel_corrs = np.stack([hutils.corr_rows_parcel(i) for i in all_aligned_sources_decode_parcel])  #array (ntargetsubjects, nsamples,n_source_subjectpairs,nparcels) containing correlation across all vertices
            vprint(f'{c.time()} CORRS: get corrs whole brain start')
            imgs_decode_aligned_corrs = np.dstack([hutils.corr_rows(i,prefer='processes') for i in all_aligned_sources_decode]) #array (ntargetsubjects,nsamples,n_source_subjectpairs) containing correlation across all vertices
            vprint(f'{c.time()} CORRS: find means start')
            corrs_mean = np.mean(imgs_decode_aligned_corrs) #whole-brain correlations averaged across target subjects, source subject pairs, and samples
            corrs_mean_parcel = imgs_decode_aligned_parcel_corrs.mean(axis=(0,1,2)) #parcel-specific correlations, averaged across target subjects, source subject pairs and samples
            corrs_mean_parcel_mean = corrs_mean_parcel.mean()
            print(f'Correlation whole-brain {corrs_mean:.3f}, per-parcel {corrs_mean_parcel_mean:.3f}')

            if imgs_decode_meanstds is None:
                all_aligned_sources_decode = hutils.transform_all_decode_data(subs,imgs_decode,aligners,post_decode_smooth,None,stack=True)
            else: #add parcel-specific means and stds to decode data
                vprint(f'{c.time()}: Aligning decode data start')
                all_aligned_sources_decode = hutils.transform_all_decode_data(subs,imgs_decode,aligners,post_decode_smooth,imgs_decode_meanstds)  
                imgs_decode = [np.hstack([x,y]) for x,y in zip(imgs_decode,imgs_decode_meanstds)]

            vprint(f'{c.time()}: Classification start')  
            classification_scores = Parallel(n_jobs=-1)(delayed(hutils.classify_pairwise)(target,subs_indices,labels,imgs_decode[target],all_aligned_sources_decode[target],classifier) for target in subs_indices)  

            imgs_decode_aligned = [imgs_decode,all_aligned_sources_decode] #for the return value
            aligners = None  #for return value           

        elif method=='template':
        ###TEMPLATE ALIGNMENT###          
            if load_pickle:
                vprint('loading aligner')
                aligners = pickle.load(open(gutils.ospath(save_pickle_filename), "rb" ))
            else:                
                aligners=TemplateAlignment(alignment_method,clustering=clustering,alignment_kwargs=alignment_kwargs,per_parcel_kwargs=per_parcel_kwargs)
                if lowdim_template:
                    aligners.make_lowdim_template(imgs_template,clustering,n_bags=n_bags_template,method='pca')
                else:
                    aligners.make_template(imgs_template,n_bags=n_bags_template,**args_template,gamma=gamma_template)
                vprint('{} Make template done'.format(c.time())) 
                if align_template_to_imgs:
                    vprint('Fitting from template to images')
                    aligners.fit_template_to_imgs(imgs_align,n_bags=n_bags,gamma=gamma)
                else:
                    aligners.fit_to_template(imgs_align,n_bags=n_bags,gamma=gamma)
                vprint(hutils.memused()) 

                """
                print(aligners.estimators[0].fit_[1].R[3,6:10])
                from tkalign_utils import randomise_but_preserve_row_col_sums_template as randomise
                aligners = randomise(aligners)
                print(aligners.estimators[0].fit_[1].R[3,6:10])
                """
                
                if absValueOfAligner:
                    [aligners.estimators[j].absValue() for j in range(len(aligners.estimators))]
                vprint('{} Template align done'.format(c.time())) 
                if return_aligner: 
                    return aligners
                if save_pickle: 
                    vprint('saving aligner')
                    pickle.dump(aligners,open(save_pickle_filename,"wb"))

            if False:
                print('Only saving aligners')
                return [0], [0], [0], aligners

            #aligners.estimators is list(subs_indices) of SurfacePairwiseAlignments, from each subj to template       
            imgs_decode_aligned=[post_decode_smooth(aligners.transform(imgs_decode[i],i)) for i in range(len(imgs_decode))]
            if plot_impulse_response:
                ratio_within_roi_first = hutils.do_plot_impulse_responses(p,'FirstSub',aligners.estimators[0]) #aligner for plot is sub 0 to template
                ratio_within_roi_last = hutils.do_plot_impulse_responses(p,'LastSub',aligners.estimators[-1]) #aligner for plot is last sub to template
                t.print(f'Ratios within ROI: First sub {ratio_within_roi_first:.2f}, Last sub {ratio_within_roi_last:.2f}')

            if plot_contrast_maps:
                for i in range(1):
                    for contrast in [3]: #Visualise predicted contrast map   
                        p.plot(imgs_decode[i][contrast,:],'Con{}_sub{}'.format(contrast,i))
                        p.plot(imgs_decode_aligned[i][contrast,:],'_Con{}_subTemplate_from_sub{}'.format(contrast,i))
            if plot_scales:
                p.plot(hutils.aligner_get_scale_map(aligners.estimators[0])) #plot scale parameter for Procrustes aligner

            if False:
                print('Skip classification. Only saving aligners and imgs_decode_aligned')
                return [0], [0], imgs_decode_aligned, aligners
            del aligners
            vprint('{} Template transform done'.format(c.time()))
            hutils.getloadavg()
           
        elif method=='anat':
            imgs_decode_aligned=[post_decode_smooth(ntask) for ntask in imgs_decode]
        del imgs_align

        if method in ['template','anat']: 

            #Inter-subject correlation (ISC)
            vprint(f'{c.time()} ISC start')
            ISC = hutils.get_ISC(imgs_decode_aligned).astype(np.float32) #array (nvertices,nsubjects)
            ISCm = ISC.mean(axis=1) #ISCs at each vertex, averaged across subject pairs
            print(f'Mean ISC (across subjects/vertices): {ISC.mean():.3f}') #ISC averaged across subject pairs and vertices

            vprint(f"{c.time()} ISC done, {hutils.memused()}")
            #p=hutils.surfplot('',plot_type='open_in_browser')
            #p.plot(imgs_decode_aligned[0][-1:])
            if plot_isc:
                p.plot(ISCm,'ISCm',vmax=1) 


            if plot_impulse_response and method == 'pairwise': 
                aligner = next(iter(aligners.values()))
                ratio_within_roi = hutils.do_plot_impulse_responses(p,'',aligner)
                t.print(f'Ratio within ROI {ratio_within_roi:.2f}')

            if do_classification:
                if imgs_decode_meanstds is not None: #add parcel-specific means and stds to decode data
                    imgs_decode_aligned = [np.hstack([x,y]) for x,y in zip(imgs_decode_aligned,imgs_decode_meanstds)]

                """
                vprint(f'{c.time()}: ADD ANATOMY')
                imgs_decode_anat=[post_decode_smooth(ntask) for ntask in imgs_decode]
                imgs_decode_aligned = [np.hstack([x,y]) for x,y in zip(imgs_decode_aligned,imgs_decode_anat)]
                vprint(f'{c.time()}: END ADD ANATOMY')
                """

                #To see accuracy for all subjects     
                X=np.vstack(imgs_decode_aligned)
                y=np.hstack(labels)
                num_of_tasks=imgs_decode_aligned[0].shape[0]
                subjects=np.hstack([np.tile([i],num_of_tasks) for i in range(n_subs)])
                from sklearn.model_selection import GroupKFold,cross_val_score, GridSearchCV
                gkf=GroupKFold(n_splits=n_splits)       
                classification_scores=cross_val_score(classifier,X,y,groups=subjects,cv=gkf,n_jobs=-1)       
                        
                #To see accuracy for subsets of subjects
                """
                for gp in [range(10,20)]:
                    X=np.vstack([imgs_decode_aligned[i] for i in gp])
                    y=np.hstack([labels[i] for i in gp])
                    num_of_tasks=imgs_decode_aligned[0].shape[0]
                    subjects=np.hstack([np.tile([i],num_of_tasks) for i in range(len(gp))])
                    from sklearn.model_selection import GroupKFold,cross_val_score, GridSearchCV
                    gkf=GroupKFold(n_splits=n_splits)       
                    classification_scores=cross_val_score(classifier,X,y,groups=subjects,cv=gkf,n_jobs=-1) 
                    vprint('Subgroup mean classification accuracy {:.2f}'.format(np.mean([np.mean(i) for i in classification_scores])))  
                """
                vprint(f'{c.time()} Classifications done')
            else:
                classification_scores = [0]
        
        vprint(hutils.memused())
        return classification_scores, ISC,imgs_decode_aligned, None


###########################################################
    filename = hutils.datetime_for_filename()
    resultsfilepath=gutils.ospath(f'{results_path}/r{filename}.txt')
    with open(resultsfilepath,'w') as resultsfile:
        t=gutils.cprint(resultsfile) 
        c=gutils.clock()     


        if True: #True for Home PC and functional alignment alone, False for IQ prediction
            subjects = hutils.all_subs
            print("USING HTUILS.ALL_SUBS (NEEDED FOR TKALIGN, HCPALIGN)")
        else: 
            print("USING HTUILS.ALL_SUBS (NEEDED FOR PREDICTING IQ)")
            df=pd.read_csv(hutils.gutils.ospath(f'{hutils.intermediates_path}/BehavioralData.csv'))
            df.loc[df['Subject']==179548,'3T_Full_Task_fMRI']  = False #does not have MSMAll data for WM task
            cognitive_measures = ['Flanker_AgeAdj', 'CardSort_AgeAdj', 'PicSeq_AgeAdj', 'ListSort_AgeAdj', 'ProcSpeed_AgeAdj','PicVocab_AgeAdj', 'ReadEng_AgeAdj','PMAT24_A_CR','IWRD_TOT','VSPLOT_TC'] 
            rows_with_3T_taskfMRI = (df['3T_Full_Task_fMRI']==True)
            rows_with_3T_rsfmri = (df['3T_RS-fMRI_Count']==4)
            rows_with_7T_rsfmri = (df['7T_RS-fMRI_Count']==4)
            rows_with_7T_movie = (df['fMRI_Movie_Compl']==True)
            rows_with_cognitive = ~df[cognitive_measures].isna().any(axis=1)
            eligible_rows = rows_with_3T_rsfmri & rows_with_3T_taskfMRI
            subjects = [str(i) for i in df.loc[eligible_rows,'Subject']]

        #### General Parameters

        accs = [] #store accuracies
        ISCs = np.zeros((59412,0)).astype(np.float32) #store inter-subject correlations (nvertices,nsubjects)

        for n in [10,20]:
            sub_slice=slice(0,n)
            parcellation_string = 'S300' #S300, K1000, MMP, R10, I300
            MSMAll=False
            save_pickle=False
            load_pickle=False #use saved aligner
            verbose=True
            post_decode_fwhm=0

            #### Parameters for doing functional alignment
            method='template' #anat, intra_subject, pairwise, template
            alignment_method='ridge_cv' #scaled_orthogonal, permutation, optimal_transport, ridge_cv
            alignment_kwargs = {'alphas':[1000]}
            per_parcel_kwargs={}
            n_bags=1
            gamma=0

            #### Parameters for alignment data
            align_with='movie'
            runs=[0,1,2,3]
            align_fwhm=0
            align_clean=True
            FC_parcellation_string = 'S1000'
            FC_normalize=True

            #### Parameters for decode data
            decode_with = 'tasks' #tasks, movie
            decode_ncomponents = None #400
            decode_standardize = None #None, 'wholebrain' or 'parcel'                
            decode_demean=True #not relevant if decode_standardize==None
            decode_unit_variance=False

            use_parcelmeanstds = False #add parcel-specific means and stds back for classification

            for use_parcelmeanstds in [False,True]:

                if not(use_parcelmeanstds):
                    print('not use_parcelmeanstds')

                #### Parameters for making template (ignored if method!='template')

                #subs_template_slice=slice(0,2)
                subs_template_slice = slice(n,n*2)
                    
                lowdim_template=True
                align_template_to_imgs=False
                n_bags_template=1
                gamma_template=0

                args_template_dict = {'hyperalignment':{'n_iter':0,'do_level_1':True, 'normalize_imgs':'zscore', 'normalize_template':'zscore', 'remove_self':True, 'level1_equal_weight':False},\
                                    'GPA': {'n_iter':1,'do_level_1':False,'normalize_imgs':'rescale','normalize_template':'rescale','remove_self':False,'level1_equal_weight':False}}
                args_template = args_template_dict['GPA']
                #print('TEMPLATE WITH N_ITER 0')
                #args_template = {'n_iter':0,'do_level_1':False,'normalize_imgs':'rescale','normalize_template':'rescale','remove_self':False,'level1_equal_weight':False}

                #### Get data
                subs_template = subjects[subs_template_slice]
                subs_template_slice_string = f'sub{subs_template_slice.start}to{subs_template_slice.stop}'
                imgs_template,template_string = hutils.get_template_making_alignment_data(c,method,subs_template,subs_template_slice_string,align_with,runs,align_fwhm,align_clean,MSMAll,load_pickle,align_template_to_imgs,lowdim_template,args_template,n_bags_template,gamma_template,FC_parcellation_string,FC_normalize)

                subs = subjects[sub_slice] 
                sub_slice_string = f'sub{sub_slice.start}to{sub_slice.stop}'
                imgs_decode,decode_string,imgs_decode_meanstds = hutils.get_decode_data(c,subs,decode_with,align_fwhm,align_clean,MSMAll,decode_ncomponents,decode_standardize,decode_demean,decode_unit_variance,parcellation_string,use_parcelmeanstds)

                imgs_align,align_string = hutils.get_alignment_data(c,subs,method,align_with,runs,align_fwhm,align_clean,MSMAll,load_pickle,FC_parcellation_string=FC_parcellation_string,FC_normalize=FC_normalize)          

                if decode_with=='movie':
                    do_classification=False
                else:
                    do_classification=True

                print(f"{c.time()} Getting all data done")
                print(hutils.memused())  

                #gammas_folder = 'gammasAmovf0123t0_D7tasksf&ms_S300_Tmovf0123t0sub20to40_L_TempRidg_gam1alphas[1000]_sub0to20_0'
                #gammas_parcel = np.load(gutils.ospath(f'{results_path}/figures/hcpalign/{gammas_folder}/best_gamma.npy'))

                gammas = [0]


                
                #Preparation for ProMises model
                """
                nparcs=parcellation_string[1:]
                gdists_path=hutils.gutils.ospath(f'{hutils.intermediates_path}/geodesic_distances/gdist_full_100610.midthickness.32k_fs_LR.S{nparcs}.p') #Get saved geodesic distances between vertices (for vertices in each parcel separately)
                import pickle
                with open(gdists_path,'rb') as file:
                    gdists = pickle.load(file)
                promises_k=0 #k parameter in ProMises model
                #for promises_k in [0,.01,.03,.1,.3,1,3,10]:
                promises_F = [np.exp(-i) for i in gdists] #local distance matrix in ProMises model
                alignment_kwargs = {'promises_k':promises_k}
                per_parcel_kwargs = {'promises_F':promises_F}
                """

                for gamma in gammas:

                    method_string=hutils.alignment_method_string(method,alignment_method,alignment_kwargs,per_parcel_kwargs,n_bags,gamma)
                    save_string = f"A{align_string}_D{decode_string}_{parcellation_string}{template_string}_{method_string}_{sub_slice_string}_{post_decode_fwhm}"

                    t.print(f"{c.time()}: Start {save_string}")
                    scores, ISC, imgs_decode_aligned, aligners = align_and_classify(c,t,verbose,save_string, subs, imgs_align, imgs_decode, method=method ,alignment_method=alignment_method,alignment_kwargs=alignment_kwargs,per_parcel_kwargs=per_parcel_kwargs,gamma=gamma,post_decode_fwhm=post_decode_fwhm,save_pickle=save_pickle,load_pickle=load_pickle,n_bags=n_bags,n_jobs=+1,imgs_template=imgs_template,align_template_to_imgs=align_template_to_imgs,lowdim_template=lowdim_template,n_bags_template=n_bags_template,gamma_template=gamma_template,args_template=args_template,do_classification=do_classification,plot_type='open_in_browser',plot_impulse_response=False, plot_contrast_maps=False,plot_isc=False,imgs_decode_meanstds=imgs_decode_meanstds,kfolds=5)
                    t.print(f"{c.time()}: Done with {save_string}")
                    mean_accuracy = np.mean([np.mean(i) for i in scores])
                    t.print(f'Classification accuracies: mean {mean_accuracy:.3f}, folds [', end= "")
                    for score in scores:
                        t.print(f"{score:.3f},", end="")
                    t.print(']\n') 

                    ISCs = np.hstack([ISCs,ISC])
                    accs.append(np.mean(scores))

                    if False: #save aligner for each subject separately in alignpickles3
                        t.print(f"{c.time()}: Downsample aligner start")
                        for i in range(len(aligners.estimators)):
                            aligners.estimators[i] = hutils.aligner_downsample(aligners.estimators[i],dtype='float32')
                        t.print(f"{c.time()}: Downsample aligner end")
                        save_string3 = f"A{align_string}_{parcellation_string}{template_string}_{method_string}"
                        hutils.mkdir(f'{intermediates_path}/alignpickles3')
                        hutils.mkdir(f'{intermediates_path}/alignpickles3/{save_string3}')
                        import pickle
                        prefix = gutils.ospath(f'{intermediates_path}/alignpickles3/{save_string3}')
                        save_sub_aligner = lambda estimator,sub: pickle.dump(estimator,open(f'{prefix}/{sub}.p',"wb"))
                        _=Parallel(n_jobs=-1,prefer='threads')(delayed(save_sub_aligner)(estimator,sub) for estimator,sub in zip(aligners.estimators,subs))
                        #load_sub_aligner = lambda sub: pickle.load(open(f'{prefix}/{sub}.p',"rb"))
                        #_=Parallel(n_jobs=-1,prefer='threads')(delayed(pickle.load(open(f'{prefix}/{sub}.p',"wb")))(sub) for sub in subs)

                    if False: #save aligned decode data
                        imgs_decode_aligned = [i[:,0:59412] for i in imgs_decode_aligned] #remove mean and stds
                        save_string2 = f"A{align_string}_D{decode_string}_{parcellation_string}{template_string}_{method_string}_{post_decode_fwhm}"
                        hutils.mkdir(f'{intermediates_path}/alignpickles2')
                        hutils.mkdir(f'{intermediates_path}/alignpickles2/{save_string2}')
                        _=Parallel(n_jobs=-1,prefer='threads')(delayed(np.save)(gutils.ospath(f'{intermediates_path}/alignpickles2/{save_string2}/{sub}.npy'),img) for sub,img in zip(subs,imgs_decode_aligned))
                        #_ = Parallel(n_jobs=-1,prefer='threads')(delayed(np.load)(gutils.ospath(f'{intermediates_path}/alignpickles2/{save_string2}/{sub}.npy')) for sub in subs)

                print(hutils.memused())

                """
                #To save parcel-specific outcome measures including best gamma value
                best_gamma = np.array([gammas[i] for i in np.argmax(corrs,axis=0)]) #best performing gamma value for each parcel
                t.print(f'Gammas: {gammas}')
                parc_matrix = hutils.parcellation_string_to_parcmatrix('S300')
                plot_dir=f'{results_path}/figures/hcpalign/gammas{save_string}'
                p=hutils.surfplot(plot_dir,plot_type='save_as_html')
                p.plot(best_gamma @ parc_matrix,savename='gammas')

                np.save(gutils.ospath(f'{plot_dir}/best_gamma.npy'),best_gamma)
                np.save(gutils.ospath(f'{plot_dir}/corrs.npy'),corrs)
                np.save(gutils.ospath(f'{plot_dir}/gammas.npy'),np.array(gammas))
                z = np.load(gutils.ospath(f'{plot_dir}/corrs.npy'))
                """


        t.print(f'Mean ISC (across subjects/vertices): {ISCs.mean():.3f}')
        ISCs_subjectmeans = ISCs.mean(axis=0)
        t.print(f'Mean ISC (per subject): {[round(i,3) for i in ISCs_subjectmeans]}')
        t.print(f'Mean accuracies: {[round(i,3) for i in accs]}')

        #Plot ISC averaged across subjects 
        plot_isc = False
        if plot_isc:
            plot_dir=gutils.ospath(f'{results_path}/figures/hcpalign/{filename}')
            os.mkdir(plot_dir)
            np.save(gutils.ospath(f'{plot_dir}/ISCs_vertexmeans.npy'),ISCs.mean(axis=1))
            #p=hutils.surfplot(plot_dir,plot_type='save_as_html')
            #p.plot(ISCs.mean(axis=1),'ISCs_vertexmeans',vmax=1)

        print('\a') #beep sounds 