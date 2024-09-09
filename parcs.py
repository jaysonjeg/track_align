"""
Script to test different parcellations, and measures of fMRI parcel homogeneity
env py390 
"""

import numpy as np
from joblib import Parallel, delayed
import hcp_utils as hcp
import generic_utils as gutils
import cortex_utils as cutils
import brainmesh_utils as bmutils
import hcpalign_utils as hutils
import parcs_utils

import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 12})

c = gutils.clock()

#SET PATHS

hcp_folder='/mnt/d/FORSTORAGE/Data/HCP_S1200'
intermediates_path='/mnt/d/FORSTORAGE/Data/Project_Hyperalignment/AWS_studies/files0/intermediates'
results_path='/mnt/d/FORSTORAGE/Data/Project_Hyperalignment/AWS_studies/files0/results'
project_path = "D:\\FORSTORAGE\\Data\\Project_GyralBias"
biasfmri_intermediates_path = gutils.ospath(f'{project_path}/intermediates')
parcs_dir = gutils.ospath(f"{project_path}/intermediates/parcellations")
all_subs=['100610','102311','102816','104416','105923','108323','109123','111312','111514','114823','115017','115825','116726','118225','125525']

### GENERAL PARAMETERS
sub_slice = slice(0,2)
surface = 'midthickness' #which surface for calculating distances, e.g. 'white','inflated','pial','midthickness'
hemi='both' #which hemisphere, e.g. 'L','both'. 'L' default, but need 'both' for spin test

which_subject_visual = '100610' #which subject for visualization. '100610', '102311', 'standard'
surface_visual = 'inflated' #default white
MSMAll = False

img_type = 'rest' #'movie', 'rest', 'rest_3T'
runs = [0]
print(f'{c.time()}: Real HCP data, img_type: {img_type}, MSMAll: {MSMAll}, runs {runs}, ')

centre = np.median #np.mean, np.median
n_nulls = 0 #default 0, set to >0 to make spun versions of a single parcellation
gyral_bias_spin_test=False #whether to calculate spin-tested p-value for boundaries being gyral

# Parameters for homogeneity calculation
single_parcel = False #whether to calculate homogeneity for a single parcel or all parcels
this_parc = 1 #which parcel for within-parcel analysis
function = parcs_utils.homogeneity_meanFC #which function in parcs_utils to use, e.g. 'homogeneity_meanFC', 'homogeneity_mean_FC_min_distance','homogeneity_expfit_decay'
args = []
kwargs = {}
if function==parcs_utils.homogeneity_meanFC_min_distance:
    kwargs = {'min_distance': 10}


### GET MESHES AND DATA
subjects=all_subs[sub_slice]
nsubjects = len(subjects)

mask_both = cutils.get_fsLR32k_mask(hemi='both')
mask_left = cutils.get_fsLR32k_mask(hemi='L')
if hemi=='both':
    mask = mask_both
    hemi_mask = mask_both
else:
    ntotalvertices = hcp.vertex_info.num_meshl + hcp.vertex_info.num_meshr
    mask = np.zeros(ntotalvertices,dtype=bool) #mask has zeros for all R hemisphere vertices AND L hemisphere non-gray vertices
    hemi_mask = mask_left
    mask[0:len(hemi_mask)] = hemi_mask
ngrays = np.sum(hemi_mask) #number of gray matter vertices

vertices_visual,faces_visual = bmutils.hcp_get_mesh(which_subject_visual,surface_visual,MSMAll,hemi=hemi)
vertices_visual,faces_visual = bmutils.reduce_mesh((vertices_visual,faces_visual),hemi_mask)
p = cutils.surfplot('',mesh=(vertices_visual,faces_visual),plot_type = 'open_in_browser')

vertices_visual_left,faces_visual_left = bmutils.hcp_get_mesh(which_subject_visual,surface_visual,MSMAll,hemi='L')
vertices_visual_left,faces_visual_left = bmutils.reduce_mesh((vertices_visual_left,faces_visual_left),mask_left)
p_left = cutils.surfplot('',mesh=(vertices_visual_left,faces_visual_left),plot_type = 'open_in_browser')

### GET PARCELLATIONS 

atlas_names = []
atlases = []


#Existing parcellations from Pan

pan_parcellation_names = ['Brodmann78','Smith88','Flechsig92','Kleist98','Julich257',\
                          'Desikan70','AAL82','Mars82','HarvardOxford96','Destrieux150',\
                            'Brainnetome210','Cammoun219','Glasser360',\
                                'Shen200','Craddock300','Aicha344',\
                                    'Schaefer300','SchaeferHomotopic300']
pan_parcellation_names = ['Craddock300','Schaefer300','SchaeferHomotopic300']

for name in pan_parcellation_names:
    atlas_names.append(name)
    atlases.append(parcs_utils.get_parcellation_pan(parcs_dir,name)[mask_both])

"""
#Geometric parcellations from Pan
Geom_nparcels = [100,300]
for nparcels in Geom_nparcels:
    atlas_names.append(f'Geom{nparcels}')
    atlases.append(parcs_utils.get_parcellation_geom(parcs_dir,int(nparcels/2))[mask_both])
    #atlases.append(parcs_utils.get_parcellation_geom(parcs_dir,int(nparcels/2)))

#Equal area parcellations
EA_parcellation_names = ['EqualAreaPCA_150Parcels_2-5-5-3'] #'EqualAreaPCA_150Parcels_2-5-3-5'
for name in EA_parcellation_names:
    atlas_names.append(f"EA{int(name[13:16])*2}_{name[24:]}")
    atlases.append(parcs_utils.get_parcellation_EA(parcs_dir,name)[mask_both])
"""

#Atlases saved on my PC
"""
atlas_names.append('Kong2022_17n_300')
atlases.append(cutils.parcellation_string_to_parcellation('S300'))
atlas_names.append('MMP')
atlases.append(cutils.parcellation_string_to_parcellation('M'))
"""
atlas_names.append('kmeans_300')
atlases.append(cutils.parcellation_string_to_parcellation('K300'))


#multiple random k-means parcellations

for i in range(5): 
    atlas_names.append(f'kmeans_300_{i}')
    atlases.append(cutils.kmeans(300,i))  

    
#Surface atlases from netneurotools
"""
from netneurotools import datasets as nntdata
atlas_names.append('schaefer2018_fslr32k_7n_300')
atlas_path=nntdata.fetch_schaefer2018('fslr32k',data_dir=parcs_dir)['300Parcels7Networks']
#atlas = parcs_utils.dlabel_filepath_to_array(atlas_path,mask)
atlases.append(parcs_utils.dlabel_filepath_to_array(atlas_path,mask_both))

atlas_names.append('cammoun_fslr32k_250')
atlas_path=nntdata.fetch_cammoun2012('fslr32k',data_dir=parcs_dir)['scale250']
atlases.append(parcs_utils.dlabel_filepath_to_array(atlas_path,mask_both))
"""

#Volume deterministic atlases from nilearn (https://nilearn.github.io/dev/modules/datasets.html), projected to surface

"""
import nilearn

atlas_names.append('schaefer2018_vol_7n_300')
atlas_path = nilearn.datasets.fetch_atlas_schaefer_2018(n_rois=300, yeo_networks=7, resolution_mm=1, data_dir=parcs_dir, verbose=1)['maps']
atlases.append(parcs_utils.atlas_vol2surf(atlas_path,mask_both))

atlas_names.append('basc_vol_325')
atlas_path = nilearn.datasets.fetch_atlas_basc_multiscale_2015(data_dir=parcs_dir, resolution=325, version='sym')['map']
atlases.append(parcs_utils.atlas_vol2surf(atlas_path,mask_both))

atlas_names.append('destrieux_vol')
atlas_path = nilearn.datasets.fetch_atlas_destrieux_2009(lateralized=True, data_dir=parcs_dir, legacy_format=True)['maps']
atlases.append(parcs_utils.atlas_vol2surf(atlas_path,mask_both))

atlas_names.append('harvard_vol_thr0_1mm')
atlas_path = nilearn.datasets.fetch_atlas_harvard_oxford("cort-maxprob-thr0-1mm", data_dir=parcs_dir, symmetric_split=False)['maps']
atlases.append(parcs_utils.atlas_vol2surf(atlas_path,mask_both))

atlas_names.append('juelich_vol_thr0_1mm')
atlas_path = nilearn.datasets.fetch_atlas_juelich("maxprob-thr0-1mm", data_dir=parcs_dir, symmetric_split=False)['maps']
atlases.append(parcs_utils.atlas_vol2surf(atlas_path,mask_both))
"""

"""
atlas_names.append('talairach_vol_tissue')
atlas_path = nilearn.datasets.fetch_atlas_talairach('tissue')['maps']
atlases.append(parcs_utils.atlas_vol2surf(atlas_path,mask_both))

atlas_names.append('talairach_vol_brodmann')
atlas_path = nilearn.datasets.fetch_atlas_talairach('ba')['maps']
atlases.append(parcs_utils.atlas_vol2surf(atlas_path,mask_both))

atlas_names.append('dosenbach2010_vol')
atlas_path = nilearn.datasets.fetch_coords_dosenbach_2010()
atlases.append(parcs_utils.atlas_vol2surf(atlas_path,mask_both))

atlas_names.append('power2011_vol')
atlas_path = nilearn.datasets.fetch_coords_power_2011()
atlases.append(parcs_utils.atlas_vol2surf(atlas_path,mask_both))

atlas_names.append('seitzman2018_vol')
atlas_path = nilearn.datasets.fetch_coords_seitzman_2018()
atlases.append(parcs_utils.atlas_vol2surf(atlas_path,mask_both))
"""

if len(atlases)==1 and n_nulls>0:
     print('Generating spun version of the base parcellation')
     import biasfmri_utils as butils
     atlas_nulls = butils.do_spin_test(atlases[0],mask_both,n_nulls)
     atlases = atlases + [atlas_nulls[:,i] for i in range(n_nulls)]
     atlas_names = atlas_names + [f'{atlas_names[0]}_null{i}' for i in range(n_nulls)]

#If any atlas has > 59412 vertices, reduce to gray matter vertices
for i in range(len(atlases)):
    atlas = atlases[i]
    if len(atlas)>59412:
        print(f"Removing non-gray from {atlas_names[i]}")
        atlases[i] = atlas[mask] 
        
if hemi=='L': #reduce to left hemisphere
    for i in range(len(atlases)):
        atlases[i] = atlases[i][0:ngrays]

atlases_verts_per_parcel = [[np.sum(atlas==value) for value in np.unique(atlas)] for atlas in atlases]
num_parcs = [len(lists) for lists in atlases_verts_per_parcel]
nverts_perparc = [centre(lists) for lists in atlases_verts_per_parcel]

#atlases_verts_per_parcel is is a list (one for each parcellation) containing the number of vertices in each parcel. The first parcellation is the reference one. Iterate through each parcellation (2nd element onwards) in atlases_verts_per_parcel. In each parcellation, iterate through the parcels. Check if the number of vertices in the parcel is the same as the number of 


print(f'{c.time()}: Get meshes')
meshes = [bmutils.hcp_get_mesh(subject,surface,MSMAll,hemi='both',folder='MNINonLinear',version='fsaverage_LR32k') for subject in subjects]
meshes = [bmutils.reduce_mesh((vertices,faces),mask) for vertices,faces in meshes] #reduce to only gray matter vertices
all_vertices, all_faces = zip(*meshes)

print(f'{c.time()}: Get fMRI data')
ims,ims_string = hutils.get_movie_or_rest_data(subjects,img_type,runs=runs,fwhm=0,clean=True,MSMAll=MSMAll)
if hemi=='L':
    ims = [im[:,0:ngrays] for im in ims]

print(f'{c.time()}: Get vertex areas')
from joblib import Parallel,delayed
vertex_areas = Parallel(n_jobs=-1,prefer='processes')(delayed(bmutils.get_vertex_areas)(mesh) for mesh in meshes)

print(f'{c.time()}: Get parcel areas') #get parcel areas (median across parcels)
parc_areas = [[centre(bmutils.get_parcel_areas(vertex_area,atlas)) for vertex_area in vertex_areas] for atlas in atlases] 
parc_areas = [np.array(parc_areas) for parc_areas in parc_areas] #convert to numpy array
parc_areas = np.stack(parc_areas) #shape (atlases, subjects)

parc_areas_var = [[np.var(bmutils.get_parcel_areas(vertex_area,atlas)) for vertex_area in vertex_areas] for atlas in atlases] #variance across parcels in parcel areas
parc_areas_var = [np.array(parc_areas_var) for parc_areas_var in parc_areas_var] #convert to numpy array
parc_areas_var = np.stack(parc_areas_var) #shape (atlases, subjects)

#Find 10th and 90th quantile of parcel areas (in the first parcellation) for each subject
cutoff_quantiles = [0.0,1.0] #default [0,1], or otherwise [0.1,0.9]
areas = [[bmutils.get_parcel_areas(vertex_area,atlas) for vertex_area in vertex_areas] for atlas in atlases] 
area_quantiles = []
for i in range(len(subjects)):
    area_quantiles.append(np.quantile(areas[0][i],cutoff_quantiles))
valid = [[((area[i]>=area_quantiles[i][0]) & (area[i]<=area_quantiles[i][1]))for i in range(len(subjects))] for area in areas] #list (atlases) of list (subjects) of boolean arrays determining whether each parcel's is within the 10th and 90th quantile of parcel areas for the reference parcellation


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
borders = [] #save border vertices for each parcellation
for n_atlas in range(len(atlases)):
    atlas = atlases[n_atlas]
    border = bmutils.get_border_vertices(edges_left,atlas[0:ngrayl])
    border_bool = border>0
    borders.append(border_bool)
    for nsubject in range(nsubjects):
        sulc_left = sulcs_left[nsubject]
        sulc_border = sulc_left[border_bool] #sulcal depth values at the border of parcels
        sulc_nonborder = sulc_left[~border_bool] #sulcal depth values not at the border of parcels

        #Get p-value with spin test
        sulc_both = np.zeros(59412)
        sulc_both[0:ngrayl] = sulc_left
        import biasfmri_utils as butils
        sulcs_border[n_atlas,nsubject] = np.mean(sulc_border)
        cohends[n_atlas,nsubject] = butils.get_cohen_d(sulc_border,sulc_nonborder)
        tstats[n_atlas,nsubject] = stats.ttest_ind(sulc_border,sulc_nonborder)[0]

        if gyral_bias_spin_test:
            sulc_both_nulls = butils.do_spin_test(sulc_both,mask,100)
            sulc_left_nulls = sulc_both_nulls[0:ngrayl]
            cohen_d, t_stat, p_value = butils.ttest_ind_with_nulldata_given(border,sulc_left,sulc_left_nulls)
            pvals[n_atlas,nsubject] = p_value

print(f'{c.time()}: Get parcel homogeneity loop start')
homogeneities=np.zeros((len(atlases),nsubjects),dtype=np.float32) #saves fMRI homogeneity for each parcellation and each subject (median across parcels)
homogeneities_n = np.zeros((len(atlases),nsubjects),dtype=np.float32) #saves fMRI homogeneity, normalized as in https://academic.oup.com/cercor/article/28/9/3095/3978804

for n_atlas in range(len(atlases)):
    atlas = atlases[n_atlas]
    print(f"{c.time()}: Doing atlas {n_atlas}")
    for nsubject in range(nsubjects):

        #print(f'{c.time()}: Get parcel homogeneity start')
        #print(f"subject is {nsubject}")
        data = ims[nsubject] #fMRI data for single subject (timepoints x vertices)
        mesh = meshes[nsubject]

        if single_parcel:
            data_singleparc = data[:,atlas==this_parc]#fMRI data for the single parcel (timepoints x vertices)
            if function != parcs_utils.homogeneity_meanFC: #get geodesic distances
                gdists = bmutils.get_gdists_singleparc(mesh,atlas,this_parc)
                kwargs['gdists'] = gdists
            homogeneity = function(data_singleparc,*args,**kwargs)
            print(f"Homogeneity in this parcel is {homogeneity:.3f}")
        else:
            if function != parcs_utils.homogeneity_meanFC: #get geodesic distances
                parcel_masks = [(atlas==parc_index) for parc_index in np.unique(atlas)] #get mask for each parcel
                parcel_meshes = [bmutils.reduce_mesh(mesh,parcel_mask) for parcel_mask in parcel_masks] #get separate mesh for each parcel
                gdists = Parallel(n_jobs=-1,prefer='processes')(delayed(bmutils.get_gdists)(*parcel_mesh) for parcel_mesh in parcel_meshes) #10 sec on a 12 core 64 GB RAM machine, for Schaefer 300
                kwargs['gdists'] = gdists   
            
            valid_indices = valid[n_atlas][nsubject]
            homogeneities_allparcs = parcs_utils.allparcs(data,atlas,function,*args,**kwargs)[valid_indices]
            homogeneities[n_atlas,nsubject] = centre(homogeneities_allparcs)
            weights = list(np.array(atlases_verts_per_parcel[n_atlas])[valid_indices])
            weightsum = np.sum(weights)
            homogeneities_n[n_atlas,nsubject] = np.sum(homogeneities_allparcs*weights)/weightsum

        #print(f'{c.time()}: Get parcel homogeneity end')

print(f'{c.time()}: Printing outputs')

homogeneities_m = centre(homogeneities,axis=1)
homogeneities_n_m = centre(homogeneities_n,axis=1)
tstats_m = centre(tstats,axis=1)
pvals_m = centre(pvals,axis=1)
parc_areas_m = centre(parc_areas,axis=1)
parc_areas_var_log_m = np.log10(centre(parc_areas_var,axis=1))

#combine atlas_names and above variables into pandas dataframe
import pandas as pd
df = pd.DataFrame({'atlas':atlas_names,'nparcs':num_parcs,'vertsparc':nverts_perparc,'area':parc_areas_m,'LogAreaVar':parc_areas_var_log_m,'hom':homogeneities_m,'homn':homogeneities_n_m,'tstat':tstats_m,'pval':pvals_m})
#round each column of df to 3 decimal places
df = df.round({'hom':4,'homn':4,'tstat':2,'pval':3,'LogAreaVar':3,'area':0,'vertsparc':0})
print(df)
print('')

#Correlation between each pair of columns in df
correlations = df.iloc[:,1:].corr(method='spearman')
print('Correlations across parcels')
print(correlations)

"""
for i in range(len(atlases)):
    #p.plot(atlases[i],cmap='prism')
    p_left.plot(borders[i],cmap='plasma')

result = stats.spearmanr(tstats_m,homogeneities_m)
print(f"Sp corr (each dot a parcellation) between median (across subs) tstat and median (across subs) homogeneity:\n\tstat={result.statistic:.3f}, p={result.pvalue:.3f}")

print('')
for i in range(nsubjects):
    result = stats.spearmanr(tstats[:,i],homogeneities[:,i])
    print(f"Sp corr (each dot a parcellation) between tstat and homogeneity in subject {i}:\n\tstat={result.statistic:.3f}, p={result.pvalue:.3f}")

    
if len(subjects)>2:
    print('')
    for i in range(len(atlases)):
        result = stats.spearmanr(tstats[i,:],homogeneities[i,:])
        print(f"Sp corr (each dot a subject) between tstat and homogeneity in parcellation {i}:\n\tstat={result.statistic:.3f}, p={result.pvalue:.3f}")
"""


import matplotlib.pyplot as plt
fig,axs=plt.subplots(2,2,figsize=(6,6))

ax=axs[0,0]
ax.scatter(parc_areas_m,homogeneities_n_m)
ax.set_xlabel('median parcel surface area')
ax.set_ylabel('weighted homogeneity')
#ax.set_xlim(left=0)
#ax.set_ylim(bottom=0)

ax=axs[0,1]
ax.scatter(num_parcs,homogeneities_n_m)
ax.set_xlabel('no of parcels')
ax.set_ylabel('weighted homogeneity')
#ax.set_xlim(left=0)
#ax.set_ylim(bottom=0)

ax=axs[1,0]
ax.scatter(tstats_m,homogeneities_n_m)
ax.set_xlabel('t-statistic (gyral bias)')
ax.set_ylabel('weighted homogeneity')
#ax.set_xlim(left=0)

ax=axs[1,1]
ax.scatter(tstats_m,homogeneities_m)
ax.set_xlabel('t-statistic (gyral bias)')
ax.set_ylabel('Raw homogeneity')

"""
ax=axs[1,1]
meanFCn = [i/j for i,j in zip(homogeneities_n_m,num_parcs)]
ax.scatter(num_parcs,meanFCn)
ax.set_xlabel('no of parcels')
ax.set_ylabel('meanFCn')
#ax.set_xlim(left=0)
"""
fig.tight_layout()
plt.show(block=False)