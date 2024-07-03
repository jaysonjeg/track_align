"""
Script to test different measures of fMRI parcel homogeneity

TO DO:
Test expfit to make sure it works okay
Use more runs, more subjects

RESULTS:
Run 0:
-----
S300
mean_FC: 0.170
mean_FC_min_distance cutoff 10mm: 0.122
expfit_decay: 0.202

S1000:
mean_FC: 0.254

MMP
mean_FC: 0.176
mean_FC_min_distance cutoff 10mm: 0.118

K100:
expfit_decay: 0.189
----

Runs 0,1,2,3:
Results given as (subject 1, subject 2, etc)
-----
S300. 
mean_FC:                            .209, .233, .214, .236, .229
mean_FC_min_distance cutoff 10mm:   .157, .180, .166, .181, .181
expfit_decay:                       .179, .176, .197, .172, .184

MMP:
mean_FC:                            .209, .235, .210, .241, .229
mean_FC_min_distance cutoff 10mm:   .127, .170, .151, .177, .171
-----
"""

import numpy as np

import generic_utils as gutils
import brainmesh_utils as bmutils
import homo_utils


c = gutils.clock()

#SET PATHS
import hcpalign_utils as hutils
hcp_folder=hutils.hcp_folder
intermediates_path=hutils.intermediates_path
results_path=hutils.results_path
project_path = "D:\\FORSTORAGE\\Data\\Project_GyralBias"
biasfmri_intermediates_path = gutils.ospath(f'{project_path}/intermediates')
parcs_dir = gutils.ospath(f"{project_path}/intermediates/parcellations")

### GENERAL PARAMETERS
sub_slice = slice(0,2)
this_parc = 1 #which parcel for within-parcel analysis
surface = 'midthickness' #which surface for calculating distances, e.g. 'white','inflated','pial','midthickness'

which_subject_visual = '100610' #which subject for visualization. '100610', '102311', 'standard'
surface_visual = 'white'
MSMAll = False
parc_string='S300'

img_type = 'rest' #'movie', 'rest', 'rest_3T'
runs = [0]
print(f'{c.time()}: Real HCP data, img_type: {img_type}, MSMAll: {MSMAll}, runs {runs}, ')

# Parameters for homogeneity calculation
single_parcel = False #whether to calculate homogeneity for a single parcel or all parcels
function = homo_utils.homo_meanFC #which function in homo_utils to use, e.g. 'homo_meanFC', 'homo_mean_FC_min_distance','homo_expfit_decay'
args = []
kwargs = {}
if function==homo_utils.homo_meanFC_min_distance:
    kwargs = {'min_distance': 10}


### GET MESHES AND DATA
subjects=hutils.all_subs[sub_slice]
nsubjects = len(subjects)

vertices_visual,faces_visual = bmutils.hcp_get_mesh(which_subject_visual,surface_visual,MSMAll)
p = hutils.surfplot('',mesh=(vertices_visual,faces_visual),plot_type = 'open_in_browser')
mask = hutils.get_fsLR32k_mask() #boolean mask of gray matter vertices. Excludes medial wall
parc_labels = hutils.parcellation_string_to_parcellation(parc_string)
#nparcs = parc_labels.max()+1-parc_labels.min()

### GET PARCELLATIONS 

atlas_names = []
atlases = []

#Atlases saved on my PC
atlas_names.append('Kong2022_17n_300')
atlases.append(hutils.parcellation_string_to_parcellation('S300'))
atlas_names.append('MMP')
atlases.append(hutils.parcellation_string_to_parcellation('M'))
atlas_names.append('kmeans_300')
atlases.append(hutils.parcellation_string_to_parcellation('K300'))


#Surface atlases from netneurotools
"""
from netneurotools import datasets as nntdata
atlas_names.append('schaefer2018_fslr32k_7n_300')
atlas_path=nntdata.fetch_schaefer2018('fslr32k',data_dir=parcs_dir)['300Parcels7Networks']
#atlas = homo_utils.dlabel_filepath_to_array(atlas_path,mask)
atlases.append(homo_utils.dlabel_filepath_to_array(atlas_path,mask))

atlas_names.append('cammoun_fslr32k_250')
atlas_path=nntdata.fetch_cammoun2012('fslr32k',data_dir=parcs_dir)['scale250']
atlases.append(homo_utils.dlabel_filepath_to_array(atlas_path,mask))
"""

#Volume deterministic atlases from nilearn (https://nilearn.github.io/dev/modules/datasets.html), projected to surface
import nilearn

atlas_names.append('schaefer2018_vol_7n_300')
atlas_path = nilearn.datasets.fetch_atlas_schaefer_2018(n_rois=300, yeo_networks=7, resolution_mm=1, data_dir=parcs_dir, verbose=1)['maps']
atlases.append(homo_utils.atlas_vol2surf(atlas_path,mask))


atlas_names.append('basc_vol_325')
atlas_path = nilearn.datasets.fetch_atlas_basc_multiscale_2015(data_dir=parcs_dir, resolution=325, version='sym')['map']
atlases.append(homo_utils.atlas_vol2surf(atlas_path,mask))

atlas_names.append('destrieux_vol')
atlas_path = nilearn.datasets.fetch_atlas_destrieux_2009(lateralized=True, data_dir=parcs_dir, legacy_format=True)['maps']
atlases.append(homo_utils.atlas_vol2surf(atlas_path,mask))

atlas_names.append('harvard_vol_thr0_1mm')
atlas_path = nilearn.datasets.fetch_atlas_harvard_oxford("cort-maxprob-thr0-1mm", data_dir=parcs_dir, symmetric_split=False)['maps']
atlases.append(homo_utils.atlas_vol2surf(atlas_path,mask))

atlas_names.append('juelich_vol_thr0_1mm')
atlas_path = nilearn.datasets.fetch_atlas_juelich("maxprob-thr0-1mm", data_dir=parcs_dir, symmetric_split=False)['maps']
atlases.append(homo_utils.atlas_vol2surf(atlas_path,mask))



"""
atlas_names.append('talairach_vol_tissue')
atlas_path = nilearn.datasets.fetch_atlas_talairach('tissue')['maps']
atlases.append(homo_utils.atlas_vol2surf(atlas_path,mask))

atlas_names.append('talairach_vol_brodmann')
atlas_path = nilearn.datasets.fetch_atlas_talairach('ba')['maps']
atlases.append(homo_utils.atlas_vol2surf(atlas_path,mask))

atlas_names.append('dosenbach2010_vol')
atlas_path = nilearn.datasets.fetch_coords_dosenbach_2010()
atlases.append(homo_utils.atlas_vol2surf(atlas_path,mask))

atlas_names.append('power2011_vol')
atlas_path = nilearn.datasets.fetch_coords_power_2011()
atlases.append(homo_utils.atlas_vol2surf(atlas_path,mask))

atlas_names.append('seitzman2018_vol')
atlas_path = nilearn.datasets.fetch_coords_seitzman_2018()
atlases.append(homo_utils.atlas_vol2surf(atlas_path,mask))
"""

#Print information about each atlas
atlas_verts_per_parcel = [[np.sum(atlas==value) for value in np.unique(atlas)] for atlas in atlases]
atlas_min_verts_per_parcel = [np.min(lists) for lists in atlas_verts_per_parcel]
atlas_max_verts_per_parcel = [np.max(lists) for lists in atlas_verts_per_parcel]
atlas_mean_verts_per_parcel = [np.mean(lists) for lists in atlas_verts_per_parcel]

print(f"Atlas names: {atlas_names}")
print(f"Atlas - no of parcels: {[len(lists) for lists in atlas_verts_per_parcel]}")
print(f"Atlas - smallest parcels: {[np.min(lists) for lists in atlas_verts_per_parcel]}")
print(f"Atlas - largest parcels: {[np.max(lists) for lists in atlas_verts_per_parcel]}")
print(f"Atlas - median parcels {[np.median(lists) for lists in atlas_verts_per_parcel]}")


"""
from neuromaps import datasets
fslr = datasets.fetch_atlas(atlas='fslr', density='32k')
print(fslr.keys())
import nibabel as nib
lsphere, rsphere = fslr['sphere']
lvert, ltri = nib.load(lsphere).agg_data()
print(lvert.shape, ltri.shape)
annot = datasets.available_annotations(space='fsLR', den='32k')
"""

print(f'{c.time()}: Get meshes')
meshes = [bmutils.hcp_get_mesh(subject,surface,MSMAll,folder='MNINonLinear',version='fsaverage_LR32k') for subject in subjects]
meshes = [bmutils.reduce_mesh((vertices,faces),mask) for vertices,faces in meshes] #reduce to only gray matter vertices
all_vertices, all_faces = zip(*meshes)

print(f'{c.time()}: Get fMRI data')
ims,ims_string = hutils.get_movie_or_rest_data(subjects,img_type,runs=runs,fwhm=0,clean=True,MSMAll=MSMAll)

print(f'{c.time()}: Get vertex areas')
from joblib import Parallel,delayed
vertex_areas = Parallel(n_jobs=-1,prefer='processes')(delayed(bmutils.get_vertex_areas)(mesh) for mesh in meshes)

print(f'{c.time()}: Get parcel areas')
parc_areas = [[np.median(bmutils.get_parcel_areas(vertex_area,parc_labels)) for vertex_area in vertex_areas] for parc_labels in atlases] #get parcel areas (median across parcels)
parc_areas = [np.array(parc_areas) for parc_areas in parc_areas] #convert to numpy array
parc_areas = np.stack(parc_areas) #shape (atlases, subjects)


print(f"{c.time()}: Get gyral bias in parcel boundaries")
import hcp_utils as hcp
import biasfmri_utils as butils
from scipy import stats
ngrayl = len(hcp.vertex_info.grayl) #left hemisphere only
sulcs_left = [butils.hcp_get_sulc(subject)[mask][0:ngrayl] for subject in subjects]
_,edges = bmutils.triangles2edges(meshes[0][1]) #all subjects have the same mesh triangles
edges_left = butils.find_edges_left(edges, ngrayl)

tstats = np.zeros((len(atlases),nsubjects),dtype=np.float32) #saves t-statistic for each parcellation and each subject
sulcs_border = np.zeros((len(atlases),nsubjects),dtype=np.float32) #saves mean sulcal depth at parcel borders
cohends = np.zeros((len(atlases),nsubjects),dtype=np.float32) #saves Cohen's d
pvals = np.zeros((len(atlases),nsubjects),dtype=np.float32) #saves p-values
for n_atlas in range(len(atlases)):
    parc_labels = atlases[n_atlas]
    border = bmutils.get_border_vertices(edges_left,parc_labels[0:ngrayl])
    border_bool = border>0
    for nsubject in range(nsubjects):
        sulc_left = sulcs_left[nsubject]
        sulc_border = sulc_left[border_bool] #sulcal depth values at the border of parcels
        sulc_nonborder = sulc_left[~border_bool] #sulcal depth values not at the border of parcels

        #Get p-value with spin test
        """
        sulc_both = np.zeros(59412)
        sulc_both[0:ngrayl] = sulc_left
        import biasfmri_utils as butils
        sulc_both_nulls = butils.do_spin_test(sulc_both,mask,100)
        sulc_left_nulls = sulc_both_nulls[0:ngrayl]
        cohen_d, t_stat, p_value = butils.ttest_ind_with_nulldata_given(border,sulc_left,sulc_left_nulls)
        pvals[n_atlas,nsubject] = p_value
        #print(f"Sulcal depth at parcel borders {np.mean(sulc_border):.3f} vs non-border vertices {np.mean(sulc_nonborder):.3f}: cohens d {cohen_d:.3f}, t(29694)={t_stat:.3f}, spin test p={p_value:.3f}")
        #assert(0)
        """

        sulcs_border[n_atlas,nsubject] = np.mean(sulc_border)
        cohends[n_atlas,nsubject] = butils.get_cohen_d(sulc_border,sulc_nonborder)
        tstats[n_atlas,nsubject] = stats.ttest_ind(sulc_border,sulc_nonborder)[0]

assert(0)

print(f'{c.time()}: Get parcel homogeneity loop start')
homos=np.zeros((len(atlases),nsubjects),dtype=np.float32) #saves fMRI homogeneity for each parcellation and each subject (median across parcels)

for n_atlas in range(len(atlases)):
    parc_labels = atlases[n_atlas]
    print(f"Doing atlas {n_atlas}")
    for nsubject in range(nsubjects):

        #print(f'{c.time()}: Get parcel homogeneity start')
        #print(f"subject is {nsubject}")
        data = ims[nsubject] #fMRI data for single subject (timepoints x vertices)
        mesh = meshes[nsubject]

        if single_parcel:
            data_singleparc = data[:,parc_labels==this_parc]#fMRI data for the single parcel (timepoints x vertices)
            if function != homo_utils.homo_meanFC: #get geodesic distances
                gdists = bmutils.get_gdists_singleparc(mesh,parc_labels,this_parc)
                kwargs['gdists'] = gdists
            homo = function(data_singleparc,*args,**kwargs)
            print(f"Homogeneity in this parcel is {homo:.3f}")
        else:
            if function != homo_utils.homo_meanFC: #get geodesic distances
                parcel_masks = [(parc_labels==parc_index) for parc_index in np.unique(parc_labels)] #get mask for each parcel
                parcel_meshes = [bmutils.reduce_mesh(mesh,parcel_mask) for parcel_mask in parcel_masks] #get separate mesh for each parcel
                from joblib import Parallel, delayed
                gdists = Parallel(n_jobs=-1,prefer='processes')(delayed(bmutils.get_gdists)(*parcel_mesh) for parcel_mesh in parcel_meshes[:]) #10 sec on a 12 core 64 GB RAM machine, for Schaefer 300
                kwargs['gdists'] = gdists   
            homos_allparcs = homo_utils.allparcs(data,parc_labels,function,*args,**kwargs)
            homos_median = np.median(homos_allparcs)
            #print(f"Median homogeneity across all parcels is {homos_median:.3f}")
            homos[n_atlas,nsubject] = homos_median

        #print(f'{c.time()}: Get parcel homogeneity end')

print(f'{c.time()}: Finished')

homos_mean_across_subs = homos.mean(axis=1)
tstats_mean_across_subs = tstats.mean(axis=1)
pvals_mean_across_subs = pvals.mean(axis=1)
parc_areas_mean_across_subs = parc_areas.mean(axis=1)


print(f"homos_mean_across_subs\n\t{homos_mean_across_subs}")
print(f"tstats_mean_across_sub\n\t{tstats_mean_across_subs}")
print(f"pvals_mean_across_subs\n\t{pvals_mean_across_subs}")
print(f"parc_areas_mean_across_sub\n\t{parc_areas_mean_across_subs}")

result = stats.spearmanr(tstats_mean_across_subs,homos_mean_across_subs)
print(f"Sp corr (each dot a parcellation) between mean (across subs) tstat and mean (across subs) homogeneity:\n\tstat={result.statistic:.3f}, p={result.pvalue:.3f}")

print('')
for i in range(nsubjects):
    result = stats.spearmanr(tstats[:,i],homos[:,i])
    print(f"Sp corr (each dot a parcellation) between tstat and homogeneity in subject {i}:\n\tstat={result.statistic:.3f}, p={result.pvalue:.3f}")

print('')
for i in range(len(atlases)):
    result = stats.spearmanr(tstats[i,:],homos[i,:])
    print(f"Sp corr (each dot a subject) between tstat and homogeneity in parcellation {i}:\n\tstat={result.statistic:.3f}, p={result.pvalue:.3f}")

import matplotlib.pyplot as plt
fig,ax=plt.subplots()
ax.scatter(parc_areas_mean_across_subs,homos_mean_across_subjs)
ax.set_xlabel('median parcel surface area')
ax.set_ylabel('median parcel homogeneity')
plt.show(block=False)