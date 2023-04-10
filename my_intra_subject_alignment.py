"""IntraSubjectAlignment class is a hack of fmralign pairwise_alignment to replicate Tavor 2016

Care is needed, few changes but tricky ones :
* X and Y have different meanings in fit (different sets of contrasts of same subject)
* X_i and Y_i are transposed before fitting Ridge compared to alignment to predict
    Y contrasts values from X ones linearly and uniformly over voxels in a region
* To avoid changes in the code, multisubject case is done through ensembling IntraSubjectAlignment
 fitted individually.

Author : T. Bazeille, B. Thirion
"""
import warnings
import os
import nibabel as nib
nib.imageglobals.logger.level = 40  #suppress pixdim error msg
import numpy as np
#from fmralign.pairwise_alignment import PairwiseAlignment, generate_Xi_Yi
from joblib import Parallel, delayed, Memory
#from nilearn.input_data.masker_validation import check_embedded_nifti_masker
#from sklearn.model_selection import ShuffleSplit
from sklearn.base import clone
from fmralign._utils import _make_parcellation,_intersect_clustering_mask
from fmralign.template_alignment import _rescaled_euclidean_mean
'''
from fmralign.alignment_methods import RidgeAlignment, Identity, Hungarian, \
    ScaledOrthogonalAlignment, OptimalTransportAlignment, DiagonalAlignment, Alignment
from sklearn.linear_model import Ridge
'''

from my_surf_pairwise_alignment import MySurfacePairwiseAlignment
from fmralignbenchmarkmockup.fmralignbench.surf_pairwise_alignment import  generate_Xi_Yi
from fmralignbenchmarkmockup.fmralignbench.intra_subject_alignment import RidgeAl, fit_one_piece_intra, piecewise_transform_intra


def fit_parcellation_intra(X_, Y_, alignment_method,
                               clustering, n_jobs, verbose):
    """ Copy from fmralign.surf_pairwise_alignment except for transposition of fit_one_piece input
    """
    labels = clustering
    fit = Parallel(n_jobs, prefer="threads", verbose=verbose)(
        delayed(fit_one_piece_intra)(
            X_i.T, Y_i.T, alignment_method
        ) for X_i, Y_i in generate_Xi_Yi(labels, X_, Y_, verbose)
    )
    return labels, fit

class MyIntraSubjectAlignment(MySurfacePairwiseAlignment):
    """ This class replicates Tavor 2016 implementation reusing as much as possible
    code from fmralign that was designed to do hyperalignment for this exact purposes.

    Instead of aligning a pair of subject, we search regularities inside single
    subject fixed contrasts given, and some that we search.
    """

    def __init__(self, alignment_method="ridge_cv", 
                 clustering='kmeans',
                 n_jobs=1, verbose=0):
        super().__init__(
            alignment_method=alignment_method,
            clustering=clustering, n_jobs=n_jobs, verbose=verbose)

    def fit(self, X, Y):
        """Fit data X and Y and learn transformation to map X to Y

        Almost the same as pairwise align except for commented line
        Parameters
        ----------
        X: Niimg-like object
           See http://nilearn.github.io/manipulating_images/input_output.html
           source data
        Y: Niimg-like object
           See http://nilearn.github.io/manipulating_images/input_output.html
           target data

        Returns
        -------
        self
        """
        assert(self.clustering.shape[0]==X.shape[1])
        self.Y_shape = Y.shape

        self.labels_, self.fit_ = fit_parcellation_intra(X,Y,self.alignment_method, self.clustering, self.n_jobs, self.verbose)
        return self
        self.fit_, self.labels_ = [], []

    def transform(self, X):
        """Predict data from X
        Almost the same as pairwise align except for commented line
        Parameters
        ----------
        X: Niimg-like object
           See http://nilearn.github.io/manipulating_images/input_output.html
           source data

        Returns
        -------
        X_transform: Niimg-like object
           See http://nilearn.github.io/manipulating_images/input_output.html
           predicted data
        """
        X_transform = piecewise_transform_intra(self.labels_, self.fit_, X, self.Y_shape)
        return X_transform

"""
class EnsembledSubjectsIntraAlignment(MySurfacePairwiseAlignment):
    def __init__(self, alignment_method="ridge_cv", n_pieces=1,
                 clustering='kmeans', n_bags=1, mask=None,
                 smoothing_fwhm=None, standardize=None, detrend=False,
                 target_affine=None, target_shape=None, low_pass=None,
                 high_pass=None, t_r=None,
                 memory=Memory(cachedir=None), memory_level=0,
                 n_jobs=1, verbose=0):
        super().__init__(
            alignment_method=alignment_method, n_pieces=n_pieces,
            clustering=clustering, n_bags=n_bags, mask=mask,
            smoothing_fwhm=smoothing_fwhm, standardize=standardize, detrend=detrend,
            target_affine=target_affine, target_shape=target_shape, low_pass=low_pass,
            high_pass=high_pass, t_r=t_r, memory=memory, memory_level=memory_level,
            n_jobs=n_jobs, verbose=verbose)

    def fit(self, X, Y):
        ''' X and Y must be lists of equal length (number of subjects)
        Inside each element may lists or a Niimgs (all of the same len / shape)
        '''
        self.masker_ = check_embedded_nifti_masker(self)
        self.masker_.n_jobs = self.n_jobs

        if type(self.clustering) == nib.nifti1.Nifti1Image or os.path.isfile(self.clustering):
            # check that clustering provided fills the mask, if not, reduce the mask
            if 0 in self.masker_.transform(self.clustering):
                reduced_mask = _intersect_clustering_mask(
                    self.clustering, self.masker_.mask_img)
                self.mask = reduced_mask
                self.masker_ = check_embedded_nifti_masker(self)
                self.masker_.n_jobs = self.n_jobs
                self.masker_.fit()
                warnings.warn(
                    "Mask used was bigger than clustering provided. Its intersection with the clustering was used instead.")
        # Avoid warning with imgs != None
        # if masker_ has been provided a mask_img
        if self.masker_.mask_img is None:
            self.masker_.fit([X])
        else:
            self.masker_.fit()

        self.fitted_intra = []
        for X_sub, Y_sub in zip(X, Y):
            intra_align = IntraSubjectAlignment(alignment_method=self.alignment_method, n_pieces=self.n_pieces,
                                                clustering=self.clustering, n_bags=self.n_bags, mask=self.masker_,
                                                smoothing_fwhm=self.smoothing_fwhm, standardize=self.standardize, detrend=self.detrend,
                                                target_affine=self.target_affine, target_shape=self.target_shape, low_pass=self.low_pass,
                                                high_pass=self.high_pass, t_r=self.t_r, memory=self.memory, memory_level=self.memory_level,
                                                n_jobs=self.n_jobs, verbose=self.verbose)
            intra_align.fit(X_sub, Y_sub)
            self.fitted_intra.append(intra_align)
        self.Y_shape = intra_align.Y_shape
        return self

    def transform(self, X):
        ''' X is a list. Each elemen of it is same shape as one element in fit(X) list
        Return a list Y of the same form as one element for each test subject.
        '''
        Y = []
        # if we fitted n subjects we have n models in self.fitted_intra
        # for each new subject each sub, we ensemble the prediction of each model
        for X_sub in X:
            Y.append(_rescaled_euclidean_mean([intra_align.transform(
                X_sub) for intra_align in self.fitted_intra], self.masker_))
        return Y
"""