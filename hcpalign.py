if __name__=='__main__':

    import nibabel as nib
    nib.imageglobals.logger.level = 40  #suppress pixdim error msg
    import hcp_utils as hcp
    import numpy as np
    import os, itertools
    from sklearn.svm import LinearSVC
    
    import hcpalign_utils as hutils
    from hcpalign_utils import ospath
    from Connectome_Spatial_Smoothing import CSS as css
    from joblib import Parallel, delayed
    from sklearn.model_selection import KFold
    import psutil

    from fmralign.template_alignment import TemplateAlignment
    from fmralign.surf_pairwise_alignment import SurfacePairwiseAlignment

    hcp_folder=hutils.hcp_folder
    intermediates_path=hutils.intermediates_path
    results_path=hutils.results_path
    available_movies=hutils.movies 
    available_rests=hutils.rests
    available_tasks=hutils.tasks 

    def align_and_classify(\
        c,t,verbose,save_string, subs, imgs_align, imgs_decode, n_jobs=-1,MSMAll=False, method='pairwise',alignment_method='scaled_orthogonal',alignment_kwargs={}, per_parcel_kwargs={}, gamma=0, absValueOfAligner=False, scramble_aligners=False,post_decode_fwhm=0, imgs_template=None, lowdim_template=False,\
        args_template={'n_iter':1,'do_level_1':False,'normalize_imgs':None,'normalize_template':None,'remove_self':False,'level1_equal_weight':False},\
        kfolds=5,load_pickle=False, save_pickle=False,\
        plot_any=False, plot_impulse_response=False, plot_contrast_maps=False, plot_scales=False,return_aligner=False):
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
        n_jobs: processor cores for a single source to target pairwise aligment (1 parcel per thread) (my PC has 12), ie for the 'inner loop'. -1 means use all cores
        nparcs: no. of parcels. Subjects aligned within each parcel
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
        lowdim_template: bool
            True to use PCA/ICA to make template
        args_template: dict
            Passed to TemplateAlignment
        kfolds: default 5 (for classification)
        load_pickle: bool
            to load precalculated aligner
        save_pickle: bool
            to save aligner as pickle object
        plot_any #plot anything at all, or not
        plot_impulse_response #to plot aligned image of circular ROIs
        plot_contrast_maps #to plot task contrast maps (predicted through alignment)
        plot_scales #to plot scale parameter for Procrustes aligner
        args_diffusion: only relevant if align_with='diffusion'
            sift2: True or False
            tckfile e.g. 'tracks_5M' (default with sift2=True),'tracks_5M_sift1M.tck' (default with sift2=False) 'tracks_5M_sift1M.tck' 'tracks_5M_50k.tck', 'tracks_5M_sift1M_200k.tck'
            diffusion_targets_nparcs: False, or a number of parcels [100,300,1000,3000,10000] Number of connectivity targets for DA     
            diffusion_targets_nvertices: number of random vertices to use as targets for diffusion alignment
            fwhm_circ: 
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
        save_pickle_filename=ospath(f'{intermediates_path}/alignpickles/{save_string}.p')                          
        if plot_any:
            plot_dir=f'/mnt/d/FORSTORAGE/Data/Project_Hyperalignment/figures/hcpalign/{save_string}'
            p=hutils.surfplot(plot_dir,plot_type='open_in_browser')
        if load_pickle: assert(os.path.exists(ospath(save_pickle_filename)))

        #Get parcellation and classifier
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
                aligners = pickle.load(open(ospath(save_pickle_filename), "rb" ))
            else:      
                vprint(f'{c.time()}: Calculate aligners')   
                aligners = hutils.get_all_pairwise_aligners(subs,imgs_align,alignment_method,clustering,n_jobs,alignment_kwargs,per_parcel_kwargs,gamma,absValueOfAligner)
                if save_pickle: 
                    pickle.dump(aligners,open(save_pickle_filename,"wb"))
            vprint(hutils.memused())  
            vprint(f'{c.time()}: Aligning decode data start')
            all_aligned_sources_decode = hutils.transform_all_decode_data(subs,imgs_decode,aligners,post_decode_smooth)   
            vprint(hutils.memused())  
            vprint(f'{c.time()}: Classification start')  
            classification_scores = Parallel(n_jobs=-1)(delayed(hutils.classify_pairwise)(target,subs_indices,labels,imgs_decode[target],all_aligned_sources_decode[target],classifier) for target in subs_indices)               

        elif method=='template':
        ###TEMPLATE ALIGNMENT###           
            if load_pickle:
                vprint('loading aligner')
                aligners = pickle.load(open(ospath(save_pickle_filename), "rb" ))
            else:                
                aligners=TemplateAlignment(alignment_method,clustering=clustering,alignment_kwargs={},per_parcel_kwargs={})
                if lowdim_template:
                    aligners.make_lowdim_template(clustering,imgs_template)
                else:
                    aligners.make_template(imgs_template,**args_template,gamma=gamma)
                aligners.fit_to_template(imgs_align,gamma=gamma)
                vprint(hutils.memused()) 
                if absValueOfAligner:
                    [aligners.estimators[j].absValue() for j in range(len(aligners.estimators))]
                vprint('{} Template align done'.format(c.time())) 
                if return_aligner: 
                    return aligners
                if save_pickle: 
                    vprint('saving aligner')
                    pickle.dump(aligners,open(save_pickle_filename,"wb"))
                
            #aligners.estimators is list(subs_indices) of SurfacePairwiseAlignments, from each subj to template       
            imgs_decode_aligned=[post_decode_smooth(aligners.transform(imgs_decode[i],i)) for i in range(len(imgs_decode))]
            if plot_any:
                if plot_impulse_response:
                    source=0 #aligner for plot is sub 0 to template
                    aligner = aligners.estimators[source] 
                    hutils.do_plot_impulse_responses(p,'',aligner)
                if plot_contrast_maps:
                    for i in range(1):
                        for contrast in [3]: #Visualise predicted contrast map   
                            p.plot(imgs_decode[i][contrast,:],'Con{}_sub{}'.format(contrast,i))
                            p.plot(imgs_decode_aligned[i][contrast,:],'_Con{}_subTemplate_from_sub{}'.format(contrast,i))
                if plot_scales:
                    p.plot(hutils.aligner_get_scale_map(aligners.estimators[source])) #plot scale parameter for Procrustes aligner

            del aligners
            vprint('{} Template transform done'.format(c.time()))
            hutils.getloadavg()
           
        elif method=='anat':
            imgs_decode_aligned=[post_decode_smooth(ntask) for ntask in imgs_decode]
        del imgs_align

        if method in ['template','anat']: 
            
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
        
        if plot_any and plot_impulse_response and method == 'pairwise': 
            aligner = next(iter(aligners.values()))
            hutils.do_plot_impulse_responses(p,'',aligner)
        
        vprint(hutils.memused())
        vprint(f'{c.time()} Classifications done')
        return classification_scores


###########################################################
    
    resultsfilepath=ospath(f'{results_path}/r{hutils.datetime_for_filename()}.txt')
    with open(resultsfilepath,'w') as resultsfile:
        t=hutils.cprint(resultsfile) 
        c=hutils.clock()   

        #### General Parameters
        sub_slice = slice(0,2)
        parcellation_string = 'S300' #S300, K1000, MMP
        MSMAll=False
        save_pickle=False
        load_pickle=False #use saved aligner
        verbose=False
        post_decode_fwhm=0

        #### Parameters for alignment data
        align_with='movie'
        runs=[0]
        align_fwhm=0
        align_clean=True

        #### Parameters for making template (ignored if method!='template')
        subs_template_slice=slice(0,2)
        lowdim_template=False
        args_template = {'n_iter':1,'do_level_1':False,'normalize_imgs':None,'normalize_template':None,'remove_self':False,'level1_equal_weight':False}

        #### Parameters for doing functional alignment
        method='pairwise' #anat, intra_subject, pairwise, template
        alignment_method='scaled_orthogonal' #scaled_orthogonal, permutation, optimal_transport, ridge_cv
        alignment_kwargs = {'scaling':True}
        per_parcel_kwargs={}
        #gamma=0

        subs,sub_slice_string,subs_template,subs_template_slice_string = hutils.get_subjects(sub_slice,subs_template_slice) #get subject IDs
        imgs_align,imgs_decode,align_string,decode_string = hutils.get_alignment_data(c,subs,method,align_with,runs,align_fwhm,align_clean,MSMAll,load_pickle)
        imgs_template,template_string = hutils.get_template_making_alignment_data(c,method,subs_template,subs_template_slice_string,align_with,runs,align_fwhm,align_clean,MSMAll,load_pickle,lowdim_template,args_template)

        print(f"{c.time()} Getting all data done")
        print(hutils.memused())
        for gamma in [0,0.1]:
            method_string=hutils.alignment_method_string(method,alignment_method,alignment_kwargs,per_parcel_kwargs,gamma)
            save_string = f"A{align_string}_D{decode_string}_{parcellation_string}{template_string}_{method_string}_{sub_slice_string}_{post_decode_fwhm}"

            t.print(f"Start {save_string}")
            scores = align_and_classify(c,t,verbose,save_string, subs, imgs_align, imgs_decode, method=method ,alignment_method=alignment_method,alignment_kwargs=alignment_kwargs,per_parcel_kwargs=per_parcel_kwargs,gamma=gamma,post_decode_fwhm=post_decode_fwhm,save_pickle=save_pickle,load_pickle=load_pickle,n_jobs=+1,imgs_template=imgs_template,lowdim_template=lowdim_template,args_template=args_template,plot_any=False, plot_impulse_response=False, plot_contrast_maps=False)

            t.print(f"\nDone with {save_string}")
            t.print(f'Classification accuracies: ', end= "")
            for score in scores:
                t.print(f"{score:.2f} ", end="")
            t.print('\nMean classification accuracy {:.2f}'.format(np.mean([np.mean(i) for i in scores])))

            t.print('')  

    
    print('\a') #beep sounds 