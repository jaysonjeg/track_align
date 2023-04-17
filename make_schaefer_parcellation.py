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
"""
labels=hutils.Schaefer(300)
matrix=hutils.Schaefer_matrix(300)
p=hutils.surfplot('',mesh=hcp.mesh.inflated,plot_type='open_in_browser',cmap='prism')
p.plot(labels,cmap='prism')
"""

#Code to find mean diameter (max euc dist between two points) across parcels in a parcellation
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
