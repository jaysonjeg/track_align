"""
Script to make simple figures from data
"""

import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

def dict2df(dictionary,label_name='group',data_name='data'):
    """
    Convert dictionary to a dataframe with two columns: label and data. For each key do the following. For each value in the list, append a row to the dataframe with the key as the label and the value as the data.

    Parameters:
    ----------
    dictionary: keys are labels, values are data. data is a list of numbers.
    """
    import pandas as pd
    df=pd.DataFrame(columns=[label_name,data_name])
    for key,data in dictionary.items():
        for d in data:
            df=df.append({label_name:key,data_name:d},ignore_index=True)
    return df

def make_boxplot_transparent(fig,ax):
    for patch in ax.artists: #Make boxplot transparent
        fc = patch.get_facecolor()
        patch.set_facecolor(mpl.colors.to_rgba(fc, 0.0))
        fig.tight_layout()

def plot(dictionary,label_name='group',data_name='data'):
    """
    Plot a strip plot of the data in the dictionary. Each key is a label, and each value is a list of data. The data is plotted as a strip plot with a boxplot overlaid.
    
    Parameters:
    ----------
    dictionary: keys are labels, values are data. data is a list of numbers.
    label_name: name of the column in the dataframe that contains the labels
    data_name: name of the column in the dataframe that contains the data
    """
    df = dict2df(dictionary,label_name,data_name)
    fig,ax=plt.subplots(1,figsize=(7,4)) #(4,4)
    sns.set_context('talk')#,font_scale=1)
    sns.stripplot(ax=ax,data=df,y=label_name,x=data_name,size=6)
    sns.boxplot(data=df,y=label_name,x=data_name,linewidth=1,width=0.5,showfliers=False)
    make_boxplot_transparent(fig,ax)

def lineplot(xdata,ydata,xlabel=None,ylabel=None,title=None,ylim=None,xlim=None,figsize=(4,4),hline=0.84):
    """
    Make a lineplot of xdata and ydata. xdata and ydata are lists of numbers. xlabel and ylabel are strings.
    """
    fig,ax = plt.subplots(figsize=figsize)
    ax.plot(xdata,ydata,marker='o', color='k')
    if xlabel is not None: ax.set_xlabel(xlabel)
    if ylabel is not None: ax.set_ylabel(ylabel)
    if title is not None: ax.set_title(title)
    if xlim is not None: ax.set_xlim(xlim)
    if ylim is not None: ax.set_ylim(ylim)
    if hline is not None: ax.axhline(hline,linewidth=1, color='k',linestyle='--')
    fig.tight_layout()

"""
#Figure. Out-of-sample vs in-sample template
dictionary = {'In-sample\ntemplate':[0.94,0.94,0.89,0.92,0.86,],\
              'Out-of-sample\ntemplate A':[0.83,0.92,0.78,0.78,0.75,],\
                'Out-of-sample\ntemplate B':[0.78,0.89,0.78,0.75,0.72,],\
                'Anatomical':[0.89, 0.92, 0.81, 0.81, 0.81]}
label_name='Condition' #'Participants used to generate template'
data_name='Classification accuracy'
plot(dictionary,label_name,data_name)


#Figure. Gamma regularization. 1-10 for alignment testing, 11-20 for template generation, 21-30 for parameter optimization 
dictionary = {'In-sample template':[0.94,0.94,0.89,0.92,0.86,],\
              'Out-of-sample template':[0.83,0.92,0.78,0.78,0.75,],\
                'Out-of-sample template\nwith regularization':[0.94,0.94,0.86,0.89,0.83,],} 
label_name='Condition'
data_name='Classification accuracy'
plot(dictionary,label_name,data_name)
lineplot([0.02,0.05,0.1,0.2,0.5,1], [0.20, 0.35, 0.65, 0.79, 0.97,1],'Regularization parameter','Ratio within source ROI')
"""


gammas = [0,0.02,0.05,0.1,0.2,0.5,1]
ylim=[0.7,1.0]

#Pairwise with subs 0-10
lineplot(gammas,[.74,.79,.79,.83,.87,.87,.84],'Regularization parameter','Mean classification accuracy','Pairwise Procrustes',ylim)
lineplot(gammas[:-1],[.72,.77,.82,.86,.89,.89],'Regularization parameter','Mean classification accuracy','Pairwise Ridge',ylim) #alpha 1e4
lineplot(gammas,[.89,.89,.89,.88,.85,.83,.83],'Regularization parameter','Mean classification accuracy','Pairwise Optimal Transport',ylim) #reg 1

#Template with subs 0-10, template generated from subs 10-20
lineplot(gammas, [0.10,0.10,0.18,0.33,0.65,0.97,1.0],'Regularization parameter','Ratio within source ROI',"GPA template Procrustes",hline=None)
lineplot(gammas, [0.81,0.82,0.85,0.87,0.90,0.89,0.84],'Regularization parameter','Mean classification accuracy','GPA template Procrustes',ylim) 
lineplot(gammas[:-1],[.74,.76,.77,.81,.86,.91],'Regularization parameter','Mean classification accuracy','GPA template Ridge',ylim) #alpha 1e3
lineplot(gammas,[.92,.92,.91,.89,.85,.83,.83],'Regularization parameter','Mean classification accuracy','GPA template Optimal Transport',ylim) #reg 1
lineplot(gammas,[.83,.85,.88,.89,.89,.86,.84],'Regularization parameter','Mean classification accuracy','PCA template Procrustes',ylim) 
lineplot(gammas[:-1],[.81,.82,.84,.89,.91,.88],'Regularization parameter','Mean classification accuracy','PCA template Ridge',ylim) #alpha 1e3

#ProMises with subs 0-10, template generated from subs 10-20
lineplot([0,.003,.01,.03,.1,.3,1,3,10],[.81,.8,.8,.8,.84,.9,.87,.86,.84],'Parameter k','Mean classification accuracy','GPA template ProMises',ylim)




plt.show()

