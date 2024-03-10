"""
Given r^2 scores for each fold in each parcel, plot them on surface
"""

import hcp_utils as hcp
import numpy as np
import os
from sklearn.svm import LinearSVC
import hcpalign_utils as hutils
from joblib import Parallel, delayed

import hcpalign_utils as hutils
def plot_r2(filename,vmax=None):
    x = np.load(hutils.ospath(f'{hutils.intermediates_path}/predict/{filename}.npy'))
    xm = x.mean(axis=1) #mean across folds
    xmp = xm.copy()
    xmp[xmp<0]=0 #set negative r2 values to 0
    parc_matrix = hutils.parcellation_string_to_parcmatrix('S300')
    p=hutils.surfplot('',plot_type='open_in_browser',vmax=vmax)
    p.plot(xmp @ parc_matrix, savename = '', vmax=vmax)

def plot_r2_anat_and_func(suffix,vmax=None):
    plot_r2(f"{'7tasksf'}_{suffix}",vmax=vmax)
    plot_r2(f"{'Aresfcf0123t0S1000_D7tasksf&ms_S300_Tresfcf0123t0S1000sub0to20_G1ffrr_TempScal__0'}_{suffix}",vmax=vmax)

vmax = 0.2


#Task anat vs TempScal
plot_r2_anat_and_func('sub50to450_S300_ridgeCV_yf_Xffft_ar2',vmax=vmax) #default
#plot_r2('Aresfcf0123t0S1000_D7tasksf&ms_S300_Tresfcf0123t0S1000sub0to20_G1ffrr_TempScal__0CIRCSHIFT_sub50to450_S300_ridgeCV_yf_Xffft_ar2',vmax=vmax) #circshifted
#plot_r2_anat_and_func('sub450toNone_S300_ridgeCV_yf_Xffft_ar2',vmax=vmax) #subs 450 to max
#plot_r2_anat_and_func('sub50toNone_S300_ridgeCV_yf_Xffft_ar2',vmax=vmax) #subs 50 to max
plot_r2('7taskst_sub50to450_S300_ridgeCV_yf_Xffft_ar2',vmax=vmax) #MSMAll

#plot_r2_anat_and_func('sub50to450_S300_ridgeCV_yf_Xfttt_ar2',vmax=vmax) #PCA
#plot_r2_anat_and_func('sub50to450_S300_ridgeCV_yf_Xfftt_ar2',vmax=vmax) #standard scaler
#plot_r2_anat_and_func('sub450toNone_S300_ridgeCV_yf_Xfftt_ar2',vmax=vmax) #standard scaler, subs 450 to max

#plot_r2_anat_and_func('sub50to450_S300_ridgeCV_yf_Xffff_ar2',vmax=vmax) #not demean
#plot_r2('7tasksf&Aresfcf0123t0S1000_D7tasksf&ms_S300_Tresfcf0123t0S1000sub0to20_G1ffrr_TempScal__0_sub50to450_S300_ridgeCV_yf_Xffff_ar2',vmax=vmax) #not demean, both anat and func align



#Task TempRidg vs TempScal
plot_r2('Aresfcf0123t0S1000_D7tasksf&ms_S300_Tresfcf0123t0S1000sub0to50_L_TempRidg_alphas[1000]_0_sub50to450_S300_ridgeCV_yf_Xffft_ar2',vmax=vmax) #TempRidge, ffft
#plot_r2('Aresfcf0123t0S1000_D7tasksf& ms_S300_Tresfcf0123t0S1000sub0to20_G1ffrr_TempScal__0_sub50to450_S300_ridgeCV_yf_Xffft_ar2',vmax=vmax) #TempScal, ffft
#plot_r2('Aresfcf0123t0S1000_D7tasksf&ms_S300_Tresfcf0123t0S1000sub0to50_L_TempRidg_alphas[1000]_0_sub50to450_S300_ridgeCV_yf_Xffff_ar2',vmax=vmax) #TempRidge, ffff
#plot_r2('Aresfcf0123t0S1000_D7tasksf&ms_S300_Tresfcf0123t0S1000sub0to20_G1ffrr_TempScal__0_sub50to450_S300_ridgeCV_yf_Xffff_ar2',vmax=vmax) #TempScal, ffff
#plot_r2('7tasksf&Aresfcf0123t0S1000_D7tasksf&ms_S300_Tresfcf0123t0S1000sub0to50_L_TempRidg_alphas[1000]_0_sub50to450_S300_ridgeCV_yf_Xffff_ar2',vmax=vmax)



#Use rsfMRI to decode IQ
#plot_r2('Aresfcf0123t0S1000_D7tasksf&ms_S300_Tresfcf0123t0S1000sub0to20_G1ffrr_TempScal__0_sub50to450_S300_ridgeCV_yf_Xffff_ar2',vmax=vmax) #TempScal, ffff
#plot_r2('resfcf0123t0S1000_sub50to450_S300_ridgeCV_yf_Xffft_ar2',vmax=vmax)
#plot_r2('resfcf0123t0S1000_sub50to450_S300_ridgeCV_yf_Xffff_ar2',vmax=vmax)
#plot_r2('resfcf0123t0S1000_sub50to450_S300_ridgeCV_yf_Xffff_a_norm_r2',vmax=vmax)


"""
#TempRidg, FC_normalize True vs False
plot_r2('Aresfcf0123t0S1000_D7tasksf&ms_S300_Tresfcf0123t0S1000sub0to50_L_TempRidg_alphas[1000]_0_sub50to450_S300_ridgeCV_yf_Xffft_ar2')
plot_r2('Aresfcf0123t0S1000f_D7tasksf&ms_S300_Tresfcf0123t0S1000fsub0to50_L_TempRidg_alphas[1000]_0_sub50to450_S300_ridgeCV_yf_Xffft_ar2')
"""

#Gamma values. Alignment from rsfMRI applied to task fMRI
"""
plot_r2('Aresfcf0123t0S1000_S300_Tresfcf0123t0S1000sub0to50_L_TempRidg_alphas[1000]7tasksf_sub50to450_S300_ridgeCV_yf_Xffft_ar2',vmax=vmax)
#plot_r2('Aresfcf0123t0S1000t_S300_Tresfcf0123t0S1000tsub0to50_L_TempRidg_gam0.02alphas[1000]7tasksf_sub50to450_S300_ridgeCV_yf_Xffft_ar2',vmax=vmax)
plot_r2('Aresfcf0123t0S1000t_S300_Tresfcf0123t0S1000tsub0to50_L_TempRidg_gam0.05alphas[1000]7tasksf_sub50to450_S300_ridgeCV_yf_Xffft_ar2',vmax=vmax)
#plot_r2('Aresfcf0123t0S1000_S300_Tresfcf0123t0S1000sub0to50_L_TempRidg_gam0.1alphas[1000]7tasksf_sub50to450_S300_ridgeCV_yf_Xffft_ar2',vmax=vmax)
plot_r2('Aresfcf0123t0S1000_S300_Tresfcf0123t0S1000sub0to50_L_TempRidg_gam0.2alphas[1000]7tasksf_sub50to450_S300_ridgeCV_yf_Xffft_ar2',vmax=vmax)
plot_r2('Aresfcf0123t0S1000_S300_Tresfcf0123t0S1000sub0to50_L_TempRidg_gam0.5alphas[1000]7tasksf_sub50to450_S300_ridgeCV_yf_Xffft_ar2',vmax=vmax)
#plot_r2('Aresfcf0123t0S1000_S300_Tresfcf0123t0S1000sub0to50_L_TempRidg_gam1alphas[1000]7tasksf_sub50to450_S300_ridgeCV_yf_Xffft_ar2',vmax=vmax)
"""

#MSMAll True improves things further
plot_r2('Aresfcf0123t0S1000_S300_Tresfcf0123t0S1000sub0to50_L_TempRidg_alphas[1000]7tasksf_sub50to450_S300_ridgeCV_yf_Xffft_ar2',vmax=vmax)
plot_r2('Aresfct0123t0S1000t_S300_Tresfct0123t0S1000tsub0to50_L_TempRidg_alphas[1000]7taskst_sub50to450_S300_ridgeCV_yf_Xffft_ar2',vmax=vmax)

assert(0)