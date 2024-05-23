"""
Code for gyral bias in fMRI surface data
Use env py390

The code is divided into sections which you can (in)activate by setting the relevant boolean to True or False
Can use either real HCP fMRI data or noise data
"""

import numpy as np, pandas as pd, matplotlib.pyplot as plt
import os
import hcpalign_utils as hutils
from hcpalign_utils import ospath
from joblib import Parallel, delayed
import biasfmri_utils as butils
import nilearn.plotting as plotting
import nibabel as nib
import pickle
import hcp_utils as hcp
from scipy import stats

if __name__=='__main__':

    c = hutils.clock()
    #Set paths
    hcp_folder=hutils.hcp_folder
    intermediates_path=hutils.intermediates_path
    results_path=hutils.results_path
    project_path = "D:\\FORSTORAGE\\Data\\Project_GyralBias"
    biasfmri_intermediates_path = ospath(f'{project_path}/intermediates')

    ### GENERAL PARAMETERS
    sub_slice = slice(0,1)
    real_or_noise = 'real' # 'real' or 'noise
    this_parc = 1 #which parcel for within-parcel analysis
    surface = 'midthickness' #which surface for calculating distances, e.g. 'white','inflated','pial','midthickness'
    which_subject_visual = '100610' #which subject for visualization. '100610', '102311', 'standard'
    surface_visual = 'white'
    MSMAll = False
    parc_string='S300'


    ### PARAMETERS FOR TESTS OF CORRELATIONS BETWEEN TWO MAPS
    to_normalize = False #subtract parcel means from data like sulcal depth, local correlations, etc, before doing correlations
    null_method = 'no_null' #no_null, spin_test, eigenstrapping
    n_perm = 100 #number of surrogates for spin test or eigenstrapping

    ### PARAMETERS FOR SPECIFIC SUB-SECTIONS

    which_neighbours = 'local' #'local','distant'
    distance_range=(1,10) #Only relevant if which_neighbours=='distant'. Geodesic distance range in mm, e.g. (0,4), (2,4), (3,5), (4,6). (1,10) is default.
    load_neighbours = True
    save_neighbours = False

    ### PARAMETERS FOR NOISE OR REAL DATA
    subjects=hutils.all_subs[sub_slice]
    nsubjects = len(subjects)
    print(f'{c.time()}: subs {sub_slice}, data is {real_or_noise}, parc {parc_string}, surface {surface}, visualise on {which_subject_visual} {surface_visual}')
    if real_or_noise == 'noise':
        noise_source = 'surface' #'volume' or 'surface'. 'volume' means noise data in volume space projected to 'surface'. 'surface' means noise data generated in surface space
        smooth_noise_fwhm = 2 #mm of surface smoothing. Try 0 or 2
        ntimepoints = 1000 #number of timepoints, default 1000
        print(f'{c.time()}: Noise data, source: {noise_source}, smooth: {smooth_noise_fwhm}mm, ntimepoints: {ntimepoints}, test first-half, retest second-half')
    elif real_or_noise == 'real':
            img_type = 'rest' #'movie', 'rest', 'rest_3T'
            runs = [0]
            #assert(len(runs)%2==0) #even number of runs, to split into test and retest
            print(f'{c.time()}: Real HCP data, img_type: {img_type}, MSMAll: {MSMAll}, runs {runs}, test is first half, retest is second half')


    ### GET DATA
    import getmesh_utils
    if which_subject_visual =='standard':
        p=hutils.surfplot('',mesh = hcp.mesh[surface_visual], plot_type='open_in_browser')
    else:
        vertices_visual,faces_visual = getmesh_utils.get_verts_and_triangles(which_subject_visual,surface_visual,MSMAll)
        p = hutils.surfplot('',mesh=(vertices_visual,faces_visual),plot_type = 'open_in_browser')
    mask = hutils.get_fsLR32k_mask() #boolean mask of gray matter vertices. Excludes medial wall
    parc_labels = hutils.parcellation_string_to_parcellation(parc_string)
    parc_matrix = hutils.parcellation_string_to_parcmatrix(parc_string)
    nparcs = parc_labels.max()+1


    print(f'{c.time()}: Get meshes')
    import getmesh_utils

    try_different_meshes = False
    if try_different_meshes:
        folder='MNINonLinear'
        version='native'
        mesh = getmesh_utils.get_verts_and_triangles(subjects[0],surface,MSMAll,folder=folder,version=version)
        mesh_visual = getmesh_utils.get_verts_and_triangles(subjects[0],'very_inflated',MSMAll,folder=folder,version=version)
        neighbour_vertices,neighbour_distances = butils._get_all_neighbour_vertices(mesh,None)   
        neighbour_distances_mean = np.array([np.mean(i) for i in neighbour_distances])
        p2 = hutils.surfplot('',mesh=mesh_visual,plot_type='open_in_browser')
        p2.plot(np.log10(neighbour_distances_mean))
        sulc = butils.get_sulc(subjects[0],version=version)
        corr = stats.pearsonr(sulc,neighbour_distances_mean)
        print(f'Correlation between sulcal depth and mean neighbour distance is {corr[0]:.3f}, p={corr[1]:.3f}')
        assert(0)

    meshes = [getmesh_utils.get_verts_and_triangles(subject,surface,MSMAll,folder='MNINonLinear',version='fsaverage_LR32k') for subject in subjects]
    meshes = [(hutils.cortex_64kto59k(vertices),hutils.cortex_64kto59k_for_triangles(faces)) for vertices,faces in meshes] #downsample from 64k to 59k
    all_vertices, all_faces = zip(*meshes)
    sulcs = [butils.get_sulc(i)[mask] for i in subjects] #list (subjects) of sulcal depth maps



    print(f'{c.time()}: Get fMRI data')   
    if real_or_noise == 'noise':
        if noise_source=='volume':
            #Get noise data in volume space which has been projected to surface
            def get_vol2surf_noisedata(which_subject):
                noise_data_prefix = f'{which_subject}.32k_fs_LR.func'
                noise_data_path = ospath(f'{project_path}/{noise_data_prefix}.npy')
                noise = np.load(noise_data_path).astype(np.float32) #noise data in volume space projected to surface (vol 2 surf)
                if noise.shape[1] > 59412:
                    print("Removing non-gray vertices from noise fMRI data")
                    noise = noise[:,mask]
                return noise     
            ims = Parallel(n_jobs=-1,prefer='threads')(delayed(get_vol2surf_noisedata)(subject) for subject in subjects)
        elif noise_source=='surface':
            #Generate noise data in surface space
            ims = [np.random.randn(ntimepoints,59412).astype(np.float32) for i in range(nsubjects)]
        if smooth_noise_fwhm>0: #smooth the noise data using geodesic distances from subject-specific meshes
            fwhm_values_for_gdist = np.array([3,5,10]) #fwhm values for which geodesic distances have been pre-calculated
            fwhm_for_gdist = fwhm_values_for_gdist[np.where(fwhm_values_for_gdist>=smooth_noise_fwhm)[0][0]] #find smallest value greater than fwhm in the above list
            print(f'{c.time()}: Generate smoothers start')
            skernels = Parallel(n_jobs=-1,prefer='threads')(delayed(butils.get_smoothing_kernel)(subject,surface,fwhm_for_gdist,smooth_noise_fwhm,MSMAll) for subject in subjects)
            print(f'{c.time()}: Smooth noise data start')
            ims = Parallel(n_jobs=-1,prefer='threads')(delayed(butils.smooth)(im,skernel) for im,skernel in zip(ims,skernels))
            print(f'{c.time()}: Smoothing finished')
    elif real_or_noise == 'real':
        ims,ims_string = hutils.get_movie_or_rest_data(subjects,img_type,runs=runs,fwhm=0,clean=True,MSMAll=MSMAll)
    print(f'{c.time()}: Each subject data shape is {ims[0].shape}')

    ### Bias in fMRI-based parcellation (do left hemisphere alone)
    do_bias_parcellation = False
    if do_bias_parcellation:
        use_precomputed_parcellation=True
        if use_precomputed_parcellation:
            parcellation_string = 'S300'
            print(f"Using pre-computed parcellation labels from {parcellation_string}")
            if nsubjects==1: #use HCP standard sulc values
                print("Using HCP standard sulcal depth values")

        ngrayl = len(hcp.vertex_info.grayl) #left hemisphere only
        tstats_all_subjects = []
        cohend_all_subjects = []
        for nsubject in range(nsubjects): #range(nsubjects)

            sulc = sulcs[nsubject] 
            #sulc = sulcs[(nsubject+1)%nsubjects] #uncomment to compare to the sulc map of a different subject

            if use_precomputed_parcellation:
                #Use pre-computed parcellation labels
                labels = hutils.parcellation_string_to_parcellation(parcellation_string,[subjects[nsubject]])
                if labels.ndim==2: labels=np.squeeze(labels)
                labels = labels[0:ngrayl]
                labels = np.expand_dims(labels,1) #change from (nvertices,) to (nvertices,1). 
                if nsubjects==1: #use HCP standard sulc values
                    sulc = -hcp.mesh.sulc[mask] #because standard sulc values are flipped
                faces = butils.triangles_removenongray(hcp.mesh.midthickness_left[1],mask)
                _,edges = butils.faces2connectivity(faces)
            else:
                #Use clustering to generate your own functional parcellation
                n_clusters = 50 #how many functional clusters
                n_repeats_per_subject = 1 #how many different random parcellations to find per subject
                imgt = ims[nsubject].T
                faces = meshes[nsubject][1]
                print(f'{c.time()}: Faces to structural adjacency matrix')
                connectivity,edges = butils.faces2connectivity(faces)
                imgt = imgt[0:ngrayl,:]
                connectivity = connectivity[:,0:ngrayl][0:ngrayl,:]    
                print(f'{c.time()}: Agglomerative clustering')
                def do_agglomerative_clustering():
                    from sklearn.cluster import AgglomerativeClustering
                    clustering = AgglomerativeClustering(n_clusters=n_clusters, metric='euclidean', connectivity=connectivity, linkage='ward').fit(imgt)
                    return clustering.labels_
                labels = Parallel(n_jobs=1,prefer='threads')(delayed(do_agglomerative_clustering)() for i in range(n_repeats_per_subject))
                labels = np.vstack(labels).T


            #Remove edges which are outside the left hemisphere
            sulc_left = sulc[0:ngrayl] #left-sided sulcal depth map 
            edges_left = butils.find_edges_left(edges, ngrayl)

            #Get border vertices
            print(f'{c.time()}: Find border vertices start')
            border = butils.get_border_vertices(ngrayl,edges_left,labels)
            border = np.sum(border,axis=1)
            border_bool = border>0
            labels = labels[:,0] #relevant if n_repeats_per_subject>1

            #Convert L hemisphere data to bihemispheric brain data (with zeros in the R hemi)
            border_full = np.zeros(59412)
            border_full[0:ngrayl] = border
            parcels_full = np.zeros(59412)
            parcels_full[0:ngrayl] = labels
            sulc_border = sulc_left[border_bool] #sulcal depth values at the border of parcels
            sulc_nonborder = sulc_left[~border_bool] #sulcal depth values not at the border of parcels

            cohend_all_subjects.append(butils.get_cohen_d(sulc_border,sulc_nonborder))
            tstats_all_subjects.append(stats.ttest_ind(sulc_border,sulc_nonborder)[0])

            plot_single_subject_parcellations = False
            if plot_single_subject_parcellations:
                #p.plot(sulc)
                p.plot((border_full+1)/2,cmap='inferno',vmin=0,vmax=1) #Plot border points
                p.plot(parcels_full,cmap='tab20',vmin=None,vmax=None) #Plot parcellation
                fig,axs=plt.subplots(figsize=(4,3))
                ax=axs
                vp = ax.violinplot([sulc_border,sulc_nonborder],showmeans=True,showextrema=False)
                ax.set_ylabel('Sulcal depth')
                ax.set_xticks([1,2],labels=['Border','Non-border'])
                ax.set_xlabel('Vertex location')
                fig.tight_layout()
                plt.show(block=False)
                sulc_both = np.zeros(59412)
                sulc_both[0:ngrayl] = sulc_left
                print(f'{c.time()}: Do spin test start')
                sulc_both_nulls = butils.do_spin_test(sulc_both,mask,n_perm)
                print(f'{c.time()}: Do spin test done')
                sulc_left_nulls = sulc_both_nulls[0:ngrayl]
                cohen_d, t_stat, p_value = butils.ttest_ind_with_nulldata_given(border,sulc_left,sulc_left_nulls)
                print(f"Sulcal depth at parcel borders {np.mean(sulc_border):.3f} vs non-border vertices {np.mean(sulc_nonborder):.3f}: cohens d {cohen_d:.3f}, t(29694)={t_stat:.3f}, spin test p={p_value:.3f}")
                import neuromaps
                corr,pval = neuromaps.stats.compare_images(sulc_left,-border,nulls=sulc_left_nulls,metric='pearsonr')
                print(f"Correlation is {corr:.3f}, spin test p={pval:.3f}")
        
        print(f'{c.time()}: Func parcellation. Sulcal depth in border vs. non-border vertices in each participant') 
        print(f"Cohen\'s d in each participants:")
        print(cohend_all_subjects)
        print(f't-statistic in each participants:')
        print(tstats_all_subjects)
        if len(cohend_all_subjects)>1:
            result = stats.ttest_1samp(tstats_all_subjects,0,alternative='greater')
            print(f'T({result.df})={result.statistic:.3f}, p={result.pvalue:.3f}')
        
        
        #x is "matched". y is "mismatched", where sulc and parcel borders belong to different participants
        """
        x=[5.972067161978508, 4.200001724847725, 7.541103585007607, 9.647505744232744, 9.257547053985983, 10.310557605592088, 2.4880528770743906, 7.177858058572761, 7.286142525423939, 0.7765097580619749] #data from parcellation_string=='I300'
        y=[2.587842844948114, 1.4626501408007213, 1.7848729840837478, 3.9404892558753404, 4.366622999248193, 5.400840682065983, 3.8781698476385755, 0.472326956298086, 2.3287401266091416, 1.2342674147144064]
        result = stats.ttest_rel(x,y,alternative='greater')         
        print(f'2nd level, matched vs. mismatched: T({result.df})={result.statistic:.3f}, p={result.pvalue:.3f}')
        assert(0)
        """

    ### Will the largest PCA components be biased towards the sulci?
    do_pca = False
    if do_pca:
        #calculate PCA components for each array in ims separately 
        print(f'{c.time()}: PCA start')
        from sklearn.decomposition import PCA
        nsubject = 0
        img = ims[nsubject]
        sulc = sulcs[nsubject]
        pca = PCA(n_components=50)
        pca.fit(img)
        pca.components_.shape #shape (ncomponents,nvertices)

        corrs_with_sulc = []
        for component in pca.components_:
            #p.plot(component)
            corrs_with_sulc.append(np.corrcoef(component,sulc)[0,1])
        print(f"How many components are gyral biased? {np.sum(np.array(corrs_with_sulc)>0)}/{len(corrs_with_sulc)}")
        assert(0)

    get_vertex_areas = False
    if get_vertex_areas:
        print(f'{c.time()}: Get vertex areas start')
        vertex_areas = Parallel(n_jobs=-1,prefer='threads')(delayed(hutils.get_vertex_areas)(mesh) for mesh in meshes)
        print(f'{c.time()}: Get vertex areas end')

        plot_vertex_areas = False
        if plot_vertex_areas:
            nsubject=0
            #p.plot(vertex_areas[nsubject])
            fig,axs=plt.subplots(figsize=(4,4))
            axs.scatter(sulcs[nsubject],vertex_areas[nsubject],1,alpha=0.05,color='k')
            axs.set_xlabel('Sulcal depth')
            axs.set_ylabel('Vertex area')
            axs.set_title(f'Correlation: {np.corrcoef(sulcs[nsubject],vertex_areas[nsubject])[0,1]:.3f}')
            fig.tight_layout()
            plt.show(block=False) 
    else:
        vertex_areas = [None]*nsubjects 

    ### Parcellate
    do_parcellate = True
    if do_parcellate:
        print(f'{c.time()}: Parcellate')    
        ims_parc = Parallel(n_jobs=1,prefer='threads')(delayed(butils.parcel_mean)(im,parc_matrix,vertex_area) for im, vertex_area in zip(ims,vertex_areas)) #list (nsubjects) of parcellated time series (ntimepoints,nparcels)
        ims_singleparc = [im[:,parc_labels==1] for im in ims] #subjects' single-parcel time series (ntimepoints,nverticesInParcel)
        sulcs_parc = [butils.parcel_mean(sulc,parc_matrix) for sulc in sulcs] #subjects' parcellated sulcal depth maps (nparcels,)
        sulcs_singleparc = [sulc[parc_labels==1] for sulc in sulcs] #subjects' single-parcel sulcal depth maps (nverticesInParcel,)

    ### Parcel-mean values might be a bit biased towards the sulci within each parcel
    do_parcelmean = False
    if do_parcelmean:
        print(f'{c.time()}: Parcel-mean values biased towards the sulci within each parcel')
        corrs_vert_parc = Parallel(n_jobs=-1,prefer='threads')(delayed(butils.corr_between_vertex_and_parcelmean)(img,img_parc,parc_labels) for img,img_parc in zip(ims,ims_parc)) #correlations between each vertex and its parcel's mean
        for nsubject in range(nsubjects):
            cc = corrs_vert_parc[nsubject]
            #ccs=butils.smooth(cc,skernels[nsubject]) #smooth the correlation values
            sulc = sulcs[nsubject]
            sulc_ranks = np.argsort(np.argsort(sulc)) #ranks of sulcal depth values (0 is sulcus, max is gyrus)
            ccn = butils.subtract_parcelmean(cc,parc_matrix)
            sulcn = butils.subtract_parcelmean(sulc,parc_matrix)
            #p.plot(cc)
            #p.plot(sulcn)
            for pairs in ([['sulc','cc']]): #,['sulc_ranks','cc'],['sulcn','ccn']]):
                xname = pairs[0]
                yname = pairs[1]
                xvals = eval(xname)
                yvals = eval(yname)
                p.plot(yvals,cmap='inferno')
                corr,pval = butils.corr_with_nulls(xvals,yvals,mask,null_method,n_perm)
                print(f'{xname} vs {yname}: R={corr:.3f}, p={pval:.3f}')
                fig,ax=plt.subplots(figsize=(3.5,3.5))
                ax.scatter(xvals,yvals,1,alpha=0.05,color='k')
                ax.set_xlabel('Sulcal depth')
                ax.set_ylabel('Correlation with parcel mean')
                fig.tight_layout()
        assert(0)

    ### Get neighbours and correlations
    do_neighbours = True
    if do_neighbours:
        func = lambda subject,mesh: butils.get_subjects_neighbour_vertices(c, subject,surface,mesh,biasfmri_intermediates_path, which_neighbours, distance_range,load_neighbours, save_neighbours,MSMAll)
        print(f'{c.time()}: Get neighbour vertices')  
        temp = Parallel(n_jobs=-1,prefer='threads')(delayed(func)(subject,mesh) for subject,mesh in zip(subjects,meshes))
        all_neighbour_vertices, all_neighbour_distances, all_neighbour_distances_mean = zip(*temp)
        print(f'{c.time()}: Get corr neighbour start')   
        ims_adjcorr, ims_adjcorr_full = zip(*Parallel(n_jobs=1,prefer='threads')(delayed(butils.get_corr_with_neighbours)(all_neighbour_vertices[i],ims[i],parallelize=True) for i in range(nsubjects)))
        ims_adjcorr_parc = [butils.parcel_mean(im,parc_matrix) for im in ims_adjcorr] #subjects' parcel-mean local neighbourhood correlations (nparcels,)
        ims_adjcorr_singleparc = [im[parc_labels==1] for im in ims_adjcorr] #subjects' single-parcel local neighbourhood correlations (nverticesInParcel,)
        print(f'{c.time()}: Get corr neighbour end')   

        ### Interpolate neighbourhood correlations to get plot of correlation vs. geodesic distance
        do_interpolate_corrs_linearspline = False #interpolate with linear spline
        do_interpolate_corrs_exp = False #interpolate with exponential function
        if do_interpolate_corrs_exp or do_interpolate_corrs_linearspline:
            assert(which_neighbours=='distant')
            nsubject = 0      
            nvertices = len(ims_adjcorr[0])
            interp_dists = butils.get_interp_dists(distance_range) #distance values at which to interpolate
            print(f'Interpolate at distances {interp_dists}mm')
        if do_interpolate_corrs_linearspline:
            """
            Fit linear spline to each vertex's correlation-distance curve. Store interpolated correlations at integer intervals. Plot interpolated correlations vs. sulcal depth. Plot spatial map of interpolated correlation at a particular geodesic distance
            """
            print(f'{c.time()}: Fit and interpolate linear-spline start') 
            interp_corrs = butils.corr_dist_fit_linear_spline(c,interp_dists,all_neighbour_distances[nsubject],ims_adjcorr_full[nsubject])
            print(f'{c.time()}: Fit and interpolate linear-spline end')   
            fig,axs = butils.corr_dist_plot_linear_spline(c,interp_dists,interp_corrs,sulcs[nsubject],mask,null_method,n_perm)
            fig,axs=butils.corr_dist_plot_samples(all_neighbour_distances[nsubject],ims_adjcorr_full[nsubject],distance_range,interp_dists,interp_corrs)
            for i in range(len(interp_dists)): p.plot(interp_corrs[:,i])
        if do_interpolate_corrs_exp:
            """
            Fit exponential function to each vertex's correlation-distance curve. Store expfit parameters for each vertex. Plot expfit parameters on brain. Test whether expfit parameters are associated with sulcal depth.
            """
            print(f'{c.time()}: Fit exponentials start')  
            expfit_params = butils.corr_dist_fit_exp(c,distance_range,all_neighbour_distances[nsubject],ims_adjcorr_full[nsubject])
            print(f'{c.time()}: Fit exponentials end') 
            print(f'{c.time()}: Interpolate exponentials start')  
            interp_dists = np.array(list(range(distance_range[0],distance_range[1]+1)))
            interp_corrs = butils.interpolate_from_expfit_params(expfit_params,interp_dists)
            print(f'{c.time()}: Interpolate exponentials end')  
            fig,axs=butils.corr_dist_plot_samples(all_neighbour_distances[nsubject],ims_adjcorr_full[nsubject],distance_range,interp_dists,interp_corrs)
            fig,axs = butils.corr_dist_plot_exp(c,expfit_params,sulcs[nsubject],mask,null_method,n_perm)
            for i in range(len(expfit_params[1])): p.plot(expfit_params[:,i]) #plot on brain map
        if do_interpolate_corrs_exp or do_interpolate_corrs_linearspline:
            plt.show(block=False)
            assert(0)


        plot_neighbours = True
        if plot_neighbours:
            for i in range(nsubjects):
                ims_adjcorr[i][np.isnan(ims_adjcorr[i])] = 0 #replace NaNs with 0s
                #ims_adjcorr[i][ims_adjcorr[i]<0] = 0 #set negative correlations to 0

            parc_means = butils.parcel_mean(np.reshape(ims_adjcorr[0],(1,-1)),parc_matrix)
            #p.plot(parc_means @ parc_matrix) #Plot parcel-means of local correlations for subject 0

            if to_normalize:
                #im = ims_adjcorr[0]
                smoother = hutils.get_searchlight_smoother(15,sub='102311',surface='midthickness')
                normalize = lambda x: x - smoother(x)
                #normalize = lambda x: butils.subtract_parcelmean(x,parc_matrix)
                ims_adjcorr = [normalize(i) for i in ims_adjcorr]
                all_neighbour_distances_mean = [normalize(i) for i in all_neighbour_distances_mean]
                sulcs = [normalize(i) for i in sulcs]


            compare_to_ReHoPaperFig4 = False
            if compare_to_ReHoPaperFig4:
                #replicate fig 4 of https://link.springer.com/article/10.1007/s00429-014-0795-8
                im = ims_adjcorr[0]
                labels_yeo = butils.get_Yeo17Networks_fsLR32k()[mask]
                names_yeo = ['0','VisCent','VisPeri','SommMotA','SomMotB','DorsAttnA','DorsAttnB','SalVentAttnA','SalVentAttnB','LimbicB','LimbicA','ContA','ContB','ContC','DefaultA','DefaultB','DefaultC','TempPar']
                #VentAttnA, ContA,DorsAttnB, DefaultB, VentAttnB, ContB, DefaultA, LimbicB.
                parc_means = np.zeros(labels_yeo.max()+1)
                parc_stds = np.zeros(labels_yeo.max()+1)
                for i in range(labels_yeo.max()+1): parc_means[i] = np.mean(im[labels_yeo==i])
                for i in range(labels_yeo.max()+1): parc_stds[i] = np.std(im[labels_yeo==i])

                names_yeo_new = ['SalVentAttnA','ContA','DorsAttnB','DefaultB','SalVentAttnB','ContB','DefaultA','LimbicB']
                parc_means_new = [parc_means[names_yeo.index(name)] for name in names_yeo_new]

                fig,axs=plt.subplots(3)
                ax=axs[0]
                ax.bar(names_yeo,parc_means)
                ax.set_ylabel('Local correlation (mean)')
                ax=axs[1]
                ax.bar(names_yeo,parc_stds)
                ax.set_ylabel('Local correlation (std)')
                ax=axs[2]
                ax.plot(names_yeo_new,parc_means_new)
                ax.set_ylabel('Local correlation (mean)')
                fig.tight_layout()
                plt.show(block=False)
                assert(0)

            depth_vs_corrs_stats=False
            if depth_vs_corrs_stats:
                corrs = np.zeros((nsubjects,2)) #stores correlation between sulcal depth and local correlations. First column uses same subjects for both variables, second column uses different subjects (mismatched)
                for i in range(nsubjects):
                    corr,pval = stats.pearsonr(sulcs[i],ims_adjcorr[i])
                    corrs[i,0] = corr
                    corr,pval = stats.pearsonr(sulcs[i],ims_adjcorr[(i+1)%nsubjects])
                    corrs[i,1] = corr
                ttest = stats.ttest_rel(corrs[:,0],corrs[:,1],alternative='less')
                print(f"Correlation of sulcal depth with local correlations: same subjects mean {np.mean(corrs[:,0]):.3f} vs different subjects mean {np.mean(corrs[:,1]):.3f}, paired t-test, t({ttest.df})={ttest.statistic:.3f}, p={ttest.pvalue:.3f}")

                #Stripplot of the correlation between sulcal depth and local correlations, for the same subjects and different subjects. Each column of variable "corrs" will be a different column in the stripplot
                import seaborn as sns #need env nilearn with package seaborn
                import pandas as pd
                columns = ['Same\nparticipants','Different\nparticipants']
                corrs_df = pd.DataFrame(corrs,columns=columns)                   
                corrs_df = corrs_df.melt(value_vars=columns,var_name='Participants',value_name='Correlation') #convert to single column format
                fig,ax=plt.subplots(figsize=(3,4))
                sns.stripplot(x='Participants',y='Correlation',data=corrs_df,ax=ax)
                ax.set_xlabel('')
                ax.set_ylabel('Association between sulcal\ndepth and local correlation')
                fig.tight_layout()
                plt.show(block=False)
                assert(0)

            for i in [0]:
                p.plot(sulcs[i])
                p.plot(all_neighbour_distances_mean[i],cmap='inferno')
                p.plot(ims_adjcorr[i],cmap='inferno')
                if nsubjects>1:
                    p.plot(ims_adjcorr[(i+1)%nsubjects],cmap='inferno')
                fig,axs=plt.subplots(2,2,figsize=(7,7))
                ax=axs[0,0]
                ax.scatter(all_neighbour_distances_mean[i],ims_adjcorr[i],1,alpha=0.05,color='k')
                ax.set_xlabel('Inter-vertex distance (mm)')
                ax.set_ylabel('Correlation with neighbours')
                corr,pval = butils.corr_with_nulls(all_neighbour_distances_mean[i],ims_adjcorr[i],mask,null_method,n_perm)
                ax.set_title(f'R={corr:.3f}, p={pval:.3f}')
                ax=axs[0,1]
                ax.scatter(sulcs[i],all_neighbour_distances_mean[i],1,alpha=0.05,color='k')
                ax.set_xlabel('Sulcal depth')
                ax.set_ylabel('Inter-vertex distance (mm)')
                corr,pval = butils.corr_with_nulls(sulcs[i],all_neighbour_distances_mean[i],mask,null_method,n_perm)
                ax.set_title(f'R={corr:.3f}, p={pval:.3f}')
                ax=axs[1,0]
                ax.scatter(sulcs[i],ims_adjcorr[i],1,alpha=0.05,color='k')
                ax.set_xlabel('Sulcal depth')
                ax.set_ylabel('Correlation with neighbours')
                corr,pval = butils.corr_with_nulls(sulcs[i],ims_adjcorr[i],mask,null_method,n_perm)
                ax.set_title(f'R={corr:.3f}, p={pval:.3f}')
                ax=axs[1,1]
                ax.scatter(sulcs[i],ims_adjcorr[(i+1)%nsubjects],1,alpha=0.05,color='k')
                ax.set_xlabel('Sulcal depth')
                ax.set_ylabel('Correlation with neighbours')
                corr,pval = butils.corr_with_nulls(sulcs[i],ims_adjcorr[(i+1)%nsubjects],mask,null_method,n_perm)
                ax.set_title(f'Correlation of this sub\'s sulc with \nlocalcorrs of {subjects[(i+1)%nsubjects]}: R={corr:.3f}, p={pval:.3f}')
                fig.suptitle(f'Subject {subjects[i]}')
                fig.tight_layout()
                plt.show(block=False)

            assert(0) #because we've modified the data

        ### Is there a bias in the group mean local neighbourhood correlations?
        do_groupmean = False
        if do_groupmean:
            neighbour_distances_submean = np.mean(all_neighbour_distances_mean,axis=0) #mean of all subjects' local neighbourhood distances
            ims_adjcorr_mean=np.mean(ims_adjcorr,axis=0) #mean of all subjects' local neighbourhood correlations
            p.plot(butils.fillnongray(ims_adjcorr_mean,mask))
            p.plot(butils.fillnongray(neighbour_distances_submean,mask))

        #Correlation between sulc and local correlations. Looking at parcel-mean data, or single-parcel data
        do_correlations=False
        if do_correlations:
            corrs_sulc_adjcorr_parc = [np.corrcoef(sulc,adjcorr)[0,1] for sulc,adjcorr in zip(sulcs_parc,ims_adjcorr_parc)]
            corrs_sulc_adjcorr_singleparc = [np.corrcoef(sulc,adjcorr)[0,1] for sulc,adjcorr in zip(sulcs_singleparc,ims_adjcorr_singleparc)]

            print('Correlation (within-subject, across parcels) of (parcel-averaged) sulcal depth vs local correlations')
            print(corrs_sulc_adjcorr_parc)

            print('Correlation (within-subject, within single parcel, across vertices) of sulcal depth vs local correlations')
            print(corrs_sulc_adjcorr_singleparc)

    ### Analyse reliability and identifiability of functional connectivity
    do_FC = False
    if do_FC:
        print(f'{c.time()}: Calculate FC')   

        ### Parameters for identifiability plots ###
        figsize = (4,4)
        x_axis_label = 'Test scans'
        y_axis_label = 'Retest scans'
        regress_ident = False #whether to regress out rows/columns in identifiability
        ntimepoints_per_half = ims[0].shape[0]//2 #number of time points in the test and retest halves of each subject's data


        ### Identifiability - parcel level ###    
        ims_parc = [array[0:ntimepoints_per_half,:] for array in ims_parc] + [array[ntimepoints_per_half:,:] for array in ims_parc] #split each subject's data into test and retest halves
        ims_pfc = Parallel(n_jobs=-1,prefer='processes')(delayed(np.corrcoef)(im.T) for im in ims_parc) #list (nsubjects) of parcellated FC Matrices (nparcels,nparcels)

        ims_pfcv = [i.ravel() for i in ims_pfc] #list (nsubjects) of parcellated FC matrices vectorized, shape (nparcls*nparcels,)      
        ims_pfcvx = ims_pfcv[0:nsubjects] #re-split into run 1 and run 2
        ims_pfcvy = ims_pfcv[nsubjects:]

        #Correlation between test and retest scans, across all subject pairs
        print(f'{c.time()}: Get identifiability parcellated')  
        import tkalign_utils as tutils
        prefix = f'Parcellated FC (reg={regress_ident})'
        corrs = tutils.ident_plot(ims_pfcvx,x_axis_label,ims_pfcvy,y_axis_label,reg=regress_ident,normed=False,figsize=figsize,title=prefix) 
        print(f'{prefix}, test-retest identifiability is {tutils.identifiability(corrs):.2f}%')

        ### Identifiability - single parcel level ###
 
        all_parc_labels = list(range(1,11)) #which parcels, default [1], or list(range(1,nparcs))
        all_min_mm = [0,1,2,3,4,5,6,7,8,9,10,15,20] #which cutoff distances, default [0]
        results = np.zeros((len(all_parc_labels),len(all_min_mm))) #save identifiability values
        for n_parc_labels, parc_label in enumerate(all_parc_labels):
            print(f'{c.time()}: Identifiability: single parcel #{parc_label}')
            mask_singleparc = (parc_labels==parc_label)
            gdists = [butils.get_geodesic_distances_within_masked_mesh(mesh,mask_singleparc) for mesh in meshes] #get geodesic distances within the single parcel
            gdists = [i.ravel() for i in gdists]
            gdists_use = np.mean(np.stack(gdists),axis=0)

            ims_singleparc = [im[:,mask_singleparc] for im in ims] #subjects' single-parcel time series (ntimepoints,nverticesInParcel)
            ims_singleparc = [array[0:ntimepoints_per_half,:] for array in ims_singleparc] + [array[ntimepoints_per_half:,:] for array in ims_singleparc] #split each subject's data into test and retest halves
            ims_sfc = Parallel(n_jobs=-1,prefer='processes')(delayed(np.corrcoef)(im.T) for im in ims_singleparc) #list (nsubjects) of within-single-parcel FC Matrices (nvertices,nvertices)

            for n_min_mm,min_mm in enumerate(all_min_mm):
                ims_sfcv = [i.ravel() for i in ims_sfc] #list (nsubjects) of single parcel FC matrices vectorized, shape (nverticesInParcel*nverticesInParcel,)   

                gdist_mask = gdists_use > min_mm
                ratio_admissible = 100*np.sum(gdist_mask)/len(gdists_use)
                ims_sfcv = [i[gdist_mask] for i in ims_sfcv] #only keep the FC values from vertex pairs greater than a given geodesic distance from each other

                ims_sfcvx = ims_sfcv[0:nsubjects] #re-split into run 1 and run 2
                ims_sfcvy = ims_sfcv[nsubjects:]

                #Correlation between test and retest scans, across all subject pairs
                #print(f'{c.time()}: Get identifiability single parcel pairs>{min_mm}mm')  
                import tkalign_utils as tutils
                prefix = f'Single parcel FC pairs>{min_mm}mm {ratio_admissible:.1f}% (reg={regress_ident})'
                corrs = tutils.ident_plot(ims_sfcvx,x_axis_label,ims_sfcvy,y_axis_label,reg=regress_ident,normed=False,figsize=figsize,title=prefix,make_plot=False)
                identifiability = tutils.identifiability(corrs)
                print(f'{prefix}, test-retest ident is {identifiability:.0f}%')
                results[n_parc_labels,n_min_mm] = identifiability

        #Strip-plot of identifiability at each cutoff value
        results_df = pd.DataFrame(results,columns=all_min_mm)                   
        results_dfmelt = results_df.melt(value_vars=all_min_mm,var_name='cutoff',value_name='Identifiability') #convert to single column format
        fig,ax=plt.subplots(figsize=(3,4))
        import seaborn as sns
        sns.lineplot(data=results_dfmelt,x="cutoff",y="Identifiability",color='black',ax=ax)
        ax.axhline(y=10,color='r')
        ax.set_ylim(0,100)
        ax.set_xlabel('Distance threshold (mm)')
        ax.set_ylabel('Identifiability (%)')
        fig.tight_layout()

        print(f'{c.time()}: Finished with identifiability')  
        plt.show(block=False)