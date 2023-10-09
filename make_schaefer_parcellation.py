"""
Make random kmeans parcellations in HCP surface grayordinates for cortex (59412)
Use Euclidean distance between vertex xyz coordinates on HCP sphere.surg.gii surface files for subject 100610. This should approximate surface geodesic distance
Uses same number of clusters in each hemisphere
Save pickle in 'D:\FORSTORAGE\Data\Project_Hyperalignment\intermediates\kmeansparcellation'
"""

import nibabel as nib
nib.imageglobals.logger.level = 40  #suppress pixdim error msg
import numpy as np
import os
import hcp_utils as hcp
import hcpalign_utils as hutils
from hcpalign_utils import ospath
import pickle
c=hutils.clock()

"""
save_folder= 'D:\FORSTORAGE\Data\Project_Hyperalignment\intermediates\schaeferparcellation'
for n_clusters in [100,200,300,400,500,600,700,800,900,1000]: 
    print([sub,n_clusters])
    print(c.time())

    labels=hutils.Schaefer_original(n_clusters)
    #matrix=hutils.Schaefer_matrix(n_clusters) #removes the 0-valued vertices
    matrix=hutils.parc_char_matrix(labels)[1] #includes 0-valued vertices

    #remove empty parcels (none in Schaefer)
    nonempty_parcels = np.array((matrix.sum(axis=1)!=0)).squeeze()
    matrix2 = matrix[nonempty_parcels,:]
    labels2=hutils.reverse_parc_char_matrix(matrix2)
    save=ospath(f'{save_folder}/schaefer_{n_clusters}parcs')
    pickle.dump(labels2,open(f'{save}.p',"wb"))
    pickle.dump(matrix2,open(f'{save}_matrix.p',"wb"))
"""  

#visualise
p=hutils.surfplot(hutils.results_path,mesh=hcp.mesh.inflated,plot_type='save_as_html',cmap='prism')
shuf_cmap = hutils.shuffle_colormap('tab20',upsample=500)


for nparcs in [100,300,600,1000]:
    labels=hutils.Schaefer(nparcs)
    matrix=hutils.Schaefer_matrix(nparcs)
    p.plot(labels,cmap=shuf_cmap,savename=f'schaefer_{nparcs}parcs')

#Code to find mean diameter (max euc dist between two points) across parcels in a parcellation
"""
from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist
verts_mesh=hcp.mesh.midthickness[0]
gray=hutils.vertexmap_59kto64k()
verts=verts_mesh[gray,:]

for align_nparcs in [100,200,300,500,1000]:
    align_labels=hutils.Schaefer(align_nparcs)
    align_parc_matrix=hutils.Schaefer_matrix(align_nparcs) 

    results=[]
    print(f'Start at {c.time()}')
    for nlabel in range(align_labels.max()):
        v=verts[align_labels==nlabel,:]
        if len(v)>4:
            hull=ConvexHull(v)
            hullpoints=v[hull.vertices,:]
            hdist = cdist(hullpoints, hullpoints, metric='euclidean')
            bestpair = np.unravel_index(hdist.argmax(), hdist.shape)
            maxdist=np.linalg.norm(hullpoints[bestpair[0]] - hullpoints[bestpair[1]])
            results.append(maxdist)
    print(f'Done at {c.time()}')
    print(f'S={align_nparcs}, mean diameter is {np.mean(results)}')
"""

def get_Schaefer_Kong2022_components_abbrevs():
    folder='D:\\FORSTORAGE\\Data\\Project_Hyperalignment\\SchaeferParcellations'
    filename='brain_parcellation-Schaefer2018_LocalGlobal-Parcellations-README.csv'
    filepath=hutils.ospath(f'{folder}\\{filename}')
    import csv
    with open(filepath, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        data = [row for row in reader]
    data=np.array(data)
    components_abbrev = {data[i,0]:data[i,1] for i in range(data.shape[0])}
    components_abbrev['PCC']='posterior cingulate cortex'
    return components_abbrev

#Code to extract parcel name, and 17 network names from Schaefer2018_300Parcels_Kong2022_17Networks_order_info.txt
def get_Schaefer_parcelnames(nparcs=300):
    components_abbrev = get_Schaefer_Kong2022_components_abbrevs()
    networks_abbrev={'VisualA':'Visual A', 'VisualB':'Visual B','VisualC':'Visual C','Aud':'Auditory','SomMotA':'Somatomotor A','SomMotB':'Somatomotor B','Language':'Language','SalVenAttnA':'Salience/VenAttn A','SalVenAttnB':'Salience/VenAttn B','ContA':'Control A' ,'ContB':'Control B','ContC':'Control C','DefaultA':'Default A','DefaultB':'Default B','DefaultC':'Default C','DorsAttnA':'Dorsal Attention A','DorsAttnB':'Dorsal Attention B','':''} #https://academic.oup.com/cercor/article/31/10/4477/6263393?login=false

    folder='D:\\FORSTORAGE\\Data\\Project_Hyperalignment\\SchaeferParcellations\\HCP\\fslr32k\\cifti'
    filename=f'Schaefer2018_{nparcs}Parcels_Kong2022_17Networks_order_info.txt'
    filepath=ospath(f'{folder}\\{filename}')

    array = np.empty(shape=(nparcs,4),dtype='<U30')
    import re
    with open(filepath) as f:
        lines = f.readlines()
        lines2=lines[0::2] #take every other line
        assert(len(lines2) == nparcs)

        for i in range(nparcs):
            string=lines2[i]
            match = re.search(r"^17networks_(LH|RH)_([a-zA-Z]*)(_[a-zA-Z]*)?_(\d)*", string)
            assert(match)
            array[i,0] = match.group(1)
            array[i,1] = networks_abbrev[match.group(2)]
            if match.group(3) is not None:
                array[i,2] = components_abbrev[match.group(3)[1:]]
            else:
                array[i,2]='somatomotor' #renamed this because somatormotor vertices seem to have ''. They should probably be precentral or postcentral
            array[i,3] = match.group(4)
    return array

