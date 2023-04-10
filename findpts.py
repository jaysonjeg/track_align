import hcpalign_utils
from hcpalign_utils import ospath
import os
import numpy as np, pandas as pd

c=hcpalign_utils.clock()

hcp_folder=hcpalign_utils.hcp_folder
intermediates_path=hcpalign_utils.intermediates_path
available_movies=hcpalign_utils.movies 
available_rests=hcpalign_utils.rests
available_tasks=hcpalign_utils.tasks 

csv7Tpath=ospath(f'{hcp_folder}/sessionSummaryCSV_7T')
csv3Tpath=ospath(f'{hcp_folder}/sessionSummaryCSV_1200Release')

modalities_7T=['movie7T','diff3T','rest7T']
modalities_3T=['tasks3T']
modalities=modalities_7T+modalities_3T

def flatten(listoflists):
    return [item for sublist in listoflists for item in sublist]

all_descs={} #strings that map to 'Scan description' column in the csv
all_descs['movie7T']=['BOLD_MOVIE1_AP','BOLD_MOVIE2_PA','BOLD_MOVIE3_PA','BOLD_MOVIE4_AP']
all_descs['rest7T']=['BOLD_REST1_PA','BOLD_REST2_AP','BOLD_REST3_PA','BOLD_REST4_AP']
all_descs['diff3T']=['DWI_AP_dir71','DWI_PA_dir71','DWI_AP_dir72','DWI_PA_dir72']

tasks3T=['WM','MOTOR','GAMBLING','LANGUAGE','SOCIAL','RELATIONAL','EMOTION']
temp=[[f'tfMRI_{task}_LR',f'tfMRI_{task}_RL'] for task in tasks3T]
all_descs['tasks3T']=flatten(temp)
scans_7T=flatten([all_descs[i] for i in modalities_7T])
scans_3T=flatten([all_descs[i] for i in modalities_3T])
scans=flatten([all_descs[i] for i in modalities])

df =pd.DataFrame(columns=['id','include']) #whether subject has all necessary data or not
dfa=pd.DataFrame(columns=['id']+modalities) #summary data for each modality (e.g. all 3T tasks put together)
dfb=pd.DataFrame(columns=['id']+scans) #detailed data for each scan in each modality (e.g. tfMRI_WM_LR)

df.set_index('id',inplace=True)
dfa.set_index('id',inplace=True)
dfb.set_index('id',inplace=True)

print(f'Get dfb start at {c.time()}')

for filename in os.listdir(csv7Tpath): #for each subject with 7T data
    subname=filename[0:6]
    print(subname)
    
    dfsub7T=pd.read_csv(ospath(f'{csv7Tpath}/{filename}'))
    col=dfsub7T['Scan Description'].astype('string')  
    for scan in scans_7T:
        dfb.loc[subname,scan] = (col.isin([scan]).sum()>0)

    dfsub3T=pd.read_csv(ospath(f'{csv3Tpath}/{subname}.csv'))
    col=dfsub3T['Scan Description'].astype('string')
    for scan in scans_3T:
        dfb.loc[subname,scan] = (col.isin([scan]).sum()>0)

    
print(f'Get dfb finish at {c.time()}')
subnames=np.array(list(dfb.index))


for subname in subnames:
    for modality in modalities:
        descs=all_descs[modality]
        dfa.loc[subname,modality] = dfb.loc[subname,descs].all() 
        
print(f'Get dfa finish at {c.time()}')
for subname in subnames:
    df.loc[subname,'include'] = dfa.loc[subname,modalities].all()
    
included_subs = subnames[df.values.squeeze()]
np.savetxt('included_subs.csv',included_subs,fmt='%s')

#x=np.loadtxt('included_subs.csv',dtype='str')