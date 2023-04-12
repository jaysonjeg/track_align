"""
Given outputs of tkalign.py using 'RD' and 'RT', try to compute directionality
"""

import hcpalign_utils as hutils
import matplotlib.pyplot as plt, tkalign_utils as utils
import numpy as np
from scipy.stats import ttest_1samp
import matplotlib.pyplot as plt

##### Global variables ######

align_nparcs=300
nparcs=301
nblocks=5

##### Functions ######

def get_a(z):
    b,f,a=utils.load_f_and_a(z)
    return a
def reshape(a):
    if a.ndim==1:
        return np.transpose(np.reshape(a,(nblocks,nparcs)))
    elif a.ndim==3:
        return np.transpose( np.reshape(a,(a.shape[0],a.shape[1],nblocks,nparcs)) , (0,1,3,2)) #Now a is n_subs_test * n_subs_test * nparcs * nblocksperparc

        
def removenans(arr):
    arr = arr[~np.isnan(arr)]
    return np.array(arr)
def count_negs_for_each_parc(a):
    if a.ndim==4:
        out= [utils.count_negs(a[:,:,i,:]) for i in range(a.shape[2])]    
    elif a.ndim==3:
        out= [utils.count_negs(a[:,:,i]) for i in range(a.shape[2])]
    elif a.ndim==2:
        out= [utils.count_negs(a[i,:]) for i in range(a.shape[0])]
    return np.array(out)
def minus(a,b):
    #subtract corresponding elements in two lists
    return [i-j for i,j in zip(a,b)]
def get_vals(a):
    array=[]
    for i in range(a.shape[2]):
        if a.ndim==4:
            values = removenans(a[:,:,i,:])
        elif a.ndim==3:
            values = removenans(a[:,:,i])
        array.append(values)
    return np.array(array)
def tstat(a):
    array=[]
    for i in range(a.shape[0]):
        t,p=ttest_1samp(a[i,:],0)
        array.append(t)
    return np.array(array)
def corr(x):
    #Correlations between columns in x, averaged across all pairs of columns (ignoring corr between a col and itself)
    values = np.corrcoef(x)
    non_ones = values!=1
    return np.mean(values[non_ones])
def trinarize(mylist,cutoff):
    """
    Given a list li, convert values above 'cutoff' to 1, values below -cutoff to -1, and all other values to 0
    """
    li=np.copy(mylist)
    greater=li>cutoff
    lesser=li<-cutoff
    neither= np.logical_not(np.logical_or(greater,lesser))
    li[greater]=1
    li[lesser]=-1
    li[neither]=0
    return li


##### Initial preparations ######

strings=['0-10','10-20','20-30','30-40','40-50']
zd=[f'0-50_rest_{string}_RD.npy' for string in strings]
zt=[f'0-50_rest_{string}_RT.npy' for string in strings]
zb=[f'0-50_rest_{string}.npy' for string in strings]
zd = [hutils.ospath(f'{hutils.intermediates_path}/tkalign_corrs/direction/{i}') for i in zd] #RD
zt = [hutils.ospath(f'{hutils.intermediates_path}/tkalign_corrs/direction/{i}') for i in zt] #RT
zb = [hutils.ospath(f'{hutils.intermediates_path}/tkalign_corrs/maxhpmult/{i}') for i in zb] #RDRT
ad = [get_a(i) for i in zd]
at = [get_a(i) for i in zt]
ab = [get_a(i) for i in zb]
blocks,_,_=utils.load_f_and_a(zd[0]) #get blocks

##### Get the outcome measures ######

#subtract nonscrambled
adn=[utils.subtract_nonscrambled_from_a(i) for i in ad]
atn=[utils.subtract_nonscrambled_from_a(i) for i in at]
abn=[utils.subtract_nonscrambled_from_a(i) for i in ab]
adnc=np.concatenate(adn,axis=0) 
atnc=np.concatenate(atn,axis=0) 
abnc=np.concatenate(abn,axis=0) 

#remove the nans
adx=[get_vals(i) for i in adn]
atx=[get_vals(i) for i in atn]
abx=[get_vals(i) for i in abn]
adncx=get_vals(adnc)
atncx=get_vals(atnc)
abncx=get_vals(abnc)

#count_negs, averaged across subject-pairs, for each block
adnN=[count_negs_for_each_parc(i) for i in adx]
atnN=[count_negs_for_each_parc(i) for i in atx]
abnN=[count_negs_for_each_parc(i) for i in abx]

#for each parc
adnrN=[count_negs_for_each_parc(reshape(i)) for i in adn]
atnrN=[count_negs_for_each_parc(reshape(i)) for i in atn]
abnrN=[count_negs_for_each_parc(reshape(i)) for i in abn]
adncN = count_negs_for_each_parc(adncx)
atncN = count_negs_for_each_parc(atncx)
abncN = count_negs_for_each_parc(abncx)

anND=[a-b for a,b in zip(adnN,atnN)] #RD-RT difference for count_negs (av across sub-pairs)
ancND=adncN-atncN #important

axD = [a-b for a,b in zip(adx,atx)] #RD-RT difference for each subjectpair*block values
ancxD=adncx-atncx 

axDT = [tstat(i) for i in axD] #t-test on the difference
ancxDT=tstat(ancxD) #important

##### Use correlations to find if repeated runs are similar ######


#similarity between (0,5) and (5,10)
print(corr(adnN)) #RD: count_negs for each block, summed across sub-pairs
print(corr(atnN))
print(corr(abnN))

print(corr(adnrN)) #count_negs for each parc, summed across blocks and sub-pairs
print(corr(atnrN))
print(corr(abnrN))

print(corr(anND)) #count_negs(RD) - count_negs(RT)
print(corr(axDT)) #t-stat for RD-RT difference, calc. for each subpair*block

#similarity between RD and RT
print([np.corrcoef(adnN[i],atnN[i])[0,1] for i in range(len(strings))]) 


##### Compare with null models ######

#Compare ancND with null model. Cutoffs: loose: |values| > 5, tight: |values| > 10
"""
adncx_null=np.random.normal(size=adncx.shape) #assumes 50% positives and negatives
atncx_null=np.random.normal(size=atncx.shape)
adncN_null = count_negs_for_each_parc(adncx_null)
atncN_null = count_negs_for_each_parc(atncx_null)
ancND_null=minus(adncN_null,atncN_null) 
plt.hist(ancND_null,10,color='b',alpha=0.3)
plt.hist(ancND,10,color='r',alpha=0.3)
plt.show()
"""

#Compare ancxDT with null model. Cutoffs: loose: |values| > 1.5, tight: |values| > 2.5
"""
ancxD_null=np.copy(ancxD)
ancxD_null=ancxD_null.ravel()
np.random.shuffle(ancxD_null)
ancxD_null=np.reshape(ancxD_null,ancxD.shape)
ancxDT_null=tstat(ancxD_null)
plt.hist(ancxDT_null,10,color='b',alpha=0.3)
plt.hist(ancxDT,10,color='r',alpha=0.3)
plt.show()
"""

##### Plot on the brain where the directional connections are ######
"""
#The 2 outcome measures
ancNDr=reshape(ancND)
ancxDTr=reshape(ancxDT)

#Let's focus on ancxDTr
t=ancxDTr
tm=t.mean(axis=1)
tm_1p5=trinarize(tm,1.5)
tm_2=trinarize(tm,2)

p=hutils.surfplot(hutils.results_path,plot_type='open_in_browser')
align_parc_matrix=hutils.Schaefer_matrix(align_nparcs)
p.plot(tm @ align_parc_matrix ,savename='tm')
p.plot(tm_1p5 @ align_parc_matrix,savename='tm_1p5')
p.plot(tm_2 @ align_parc_matrix,savename='tm_2')
"""