#DEFUNCT

import numpy as np
from fmralignbench.surf_pairwise_alignment import fit_parcellation
from fmralign._utils import piecewise_transform
from sklearn.base import BaseEstimator, TransformerMixin
import warnings

assert(0)

class MySurfacePairwiseAlignment(BaseEstimator, TransformerMixin):
    """
    Adapted from fmralignbenchmark.SurfacePairwiseAlignment
    Inputs:
    vertices: e.g. hcp.struct.cortex
    """
    def __init__(self, alignment_method, clustering, n_jobs=1, parallel_type='threads', verbose=0,reg=0):
        """
        reg is regularization parameter
        reg ranges from 0 to 1. Bigger values are more regularization. 
        Suggest values between 0.05 (1/20) and 0.2 (1/5)
        if reg==0, then in function fit, Y is just Y (default: no regularization)
        if reg==1, then Y = X (maximum regularization)
        if reg==0.2, then Y is the weighted average (0.2*X + 0.8+Y)
        """
        self.alignment_method = alignment_method
        self.clustering = clustering
        self.n_jobs = n_jobs
        self.parallel_type = parallel_type
        self.verbose = verbose
        self.reg=reg
            
    def fit(self, X, Y):
        """Fit data X and Y and learn transformation to map X to Y
        Parameters
        ----------
        X: Niimg-like object
            Source data.
        Y: Niimg-like object
            Target data
        Returns
        -------
        self
        """
        assert(self.clustering.shape[0]==X.shape[1])

        from collections import Counter
        ntimepoints=X.shape[0]
        nVerticesInLargestCluster = Counter(self.clustering)[0]
        if ntimepoints < nVerticesInLargestCluster:
            warnings.warn(f'UserWarning: ntimepoints {ntimepoints} < nVerticesInLargestCluster {nVerticesInLargestCluster}')

        if self.reg:
            Y = np.average([X,Y],axis=0,weights=[self.reg,1-self.reg])

        self.labels_, self.fit_ = fit_parcellation(
            X, Y, self.alignment_method, self.clustering, self.n_jobs, self.parallel_type,self.verbose)
            #clustering needs to be list of ints      
        
        def change_type(fit_,whichtype):
            fit_.R = fit_.R.astype(whichtype)
            return fit_
            
        self.fit_ = [change_type(i,np.float32) for i in self.fit_] #reduce memory usage
        
        return self

    def get_spatial_map_of_scale(self):
        scales=np.ones((len(self.clustering)))
        parcels=np.unique(self.clustering)
        for i in range(len(parcels)):
            try:
                parcel=parcels[i]
                scale=self.fit_[i].scale
                indices=np.where(self.clustering==parcel)[0]
                scales[indices]=scale
            except: 
                x=1
        return scales


    def transform(self, X):
        """Predict data from X
        Parameters
        ----------
        X: Niimg-like object
            Source data
        Returns
        -------
        X_transform: Niimg-like object
            Predicted data
        """
        
        X_transform = piecewise_transform(
            self.labels_, self.fit_, X)
        return X_transform


class LowDimSurfacePairwiseAlignment(MySurfacePairwiseAlignment):
    """ 
    This class does dimensionality reduction (across vertices, separately for each pt) to get spatial components before alignment
    Because this muddles up vertices across multiple parcels, we then treat the whole brain as a single parcel
    """

    def __init__(self, alignment_method, clustering='kmeans', n_jobs=1, verbose=0, reg=0, n_components=60,whiten=False,lowdim_method='pca'):
        super().__init__(alignment_method=alignment_method, clustering=clustering, n_jobs=n_jobs, verbose=verbose,reg=reg)
        
        self.n_components=n_components
        self.whiten=whiten
        self.clustering=np.ones(n_components) #a single cluster
        self.lowdim_method=lowdim_method
            
    def fit(self, X, Y):
        #assert(self.clustering.shape[0]==X.shape[1])

        from collections import Counter
        ntimepoints=X.shape[0]
        nVerticesInLargestCluster = Counter(self.clustering)[0]
        if ntimepoints < nVerticesInLargestCluster:
            warnings.warn(f'UserWarning: ntimepoints {ntimepoints} < nVerticesInLargestCluster {nVerticesInLargestCluster}')

        if self.reg:
            Y = np.average([X,Y],axis=0,weights=[self.reg,1-self.reg])           
            
        from sklearn.decomposition import PCA,FastICA
        
        if self.lowdim_method=='pca':
            self.pcaX=PCA(n_components=self.n_components,whiten=self.whiten)
            self.pcaY=PCA(n_components=self.n_components,whiten=self.whiten)
        elif self.lowdim_method=='ica':
            self.pcaX=FastICA(n_components=self.n_components,max_iter=100000)
            self.pcaY=FastICA(n_components=self.n_components,max_iter=100000)

        X=self.pcaX.fit_transform(X) #reduces array(ntp,nvertices) to array(ntp,n_components)
        Y=self.pcaY.fit_transform(Y)
        
        self.labels_, self.fit_ = fit_parcellation(
            X, Y, self.alignment_method, self.clustering, self.n_jobs, self.verbose)
            #clustering needs to be list of ints      
        
        return self
        
    def transform(self, X):
        
        X=self.pcaX.transform(X)
        
        X_transform = piecewise_transform(
            self.labels_, self.fit_, X)
        
        #return X_transform
        return self.pcaY.inverse_transform(X_transform)