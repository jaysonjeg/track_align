"""
Adapted from fmralign: https://github.com/Parietal-INRIA/fmralign/blob/master/fmralign/template_alignment.py
"""
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
from joblib import delayed, Memory, Parallel
from nilearn.image import index_img, concat_imgs, load_img
from nilearn.input_data.masker_validation import check_embedded_nifti_masker
from fmralign.pairwise_alignment import PairwiseAlignment
from joblib import Parallel,delayed
from my_surf_pairwise_alignment import MySurfacePairwiseAlignment
from hcpalign_utils import now

def _rescaled_euclidean_mean(imgs, scale_average=False):
    """ Make the Euclidian average of images

    Parameters
    ----------
    imgs: list of Niimgs
        Each img is 3D by default, but can also be 4D.
    masker: instance of NiftiMasker or MultiNiftiMasker
        Masker to be used on the data.
    scale_average: boolean
        If true, the returned average is scaled to have the average norm of imgs
        If false, it will usually have a smaller norm than initial average
        because noise will cancel across images

    Returns
    -------
    average_img: Niimg
        Average of imgs, with same shape as one img
    """
    average_img = np.mean(imgs, axis=0)
    scale = 1
    if scale_average:
        X_norm = 0
        for img in imgs:
            X_norm += np.linalg.norm(img)
        X_norm /= len(imgs)
        scale = X_norm / np.linalg.norm(average_img)
    average_img *= scale
    return average_img


#from hcpalign_utils import memused

def _align_images_to_template(imgs, template, alignment_method,
                               clustering, n_bags,
                              memory, memory_level, n_jobs, verbose,reg):
    '''Convenience function : for a list of images, return the list
    of estimators (PairwiseAlignment instances) aligning each of them to a
    common target, the template. All arguments are used in PairwiseAlignment
    '''
   
    def _align_one_image_to_template(img2,alignment_method2,clustering2,n_jobs2,template2,reg):
        piecewise_estimator= MySurfacePairwiseAlignment(alignment_method2, clustering2, n_jobs=n_jobs2,reg=reg)
        piecewise_estimator.fit(img2, template2) 
        #print(f'within align_one_img {memused()}')
        return piecewise_estimator

    piecewise_estimators = Parallel(n_jobs=-1)(delayed(_align_one_image_to_template)(img,alignment_method,clustering,n_jobs,template,reg) for img in imgs)       
    aligned_imgs = [piecewise_estimators[i].transform(imgs[i]) for i in range(len(imgs))] 

    
    return aligned_imgs, piecewise_estimators


def _create_template(imgs, n_iter, scale_template, alignment_method,
                     clustering, n_bags, memory, memory_level,
                     n_jobs, verbose,template_method,reg):
    '''Create template through alternate minimization.  Compute iteratively :
        * T minimizing sum(||R_i X_i-T||) which is the mean of aligned images (RX_i)
        * align initial images to new template T
            (find transform R_i minimizing ||R_i X_i-T|| for each img X_i)


        Parameters
        ----------
        imgs: List of Niimg-like objects
           See http://nilearn.github.io/manipulating_images/input_output.html
           source data. Every img must have the same length (n_sample)
        scale_template: boolean
            If true, template is rescaled after each inference so that it keeps
            the same norm as the average of training images.
        n_iter: int
           Number of iterations in the alternate minimization. Each image is
           aligned n_iter times to the evolving template. If n_iter = 1,
           the template is simply the mean of the input images.
        All other arguments are the same are passed to PairwiseAlignment

        Returns
        -------
        template: list of 3D Niimgs of length (n_sample)
            Models the barycenter of input imgs
    '''

    intrinsic_reg = 1/len(imgs) #the regularization that is implicit in aligning to a group mean. This was not written with Haxby method 1 in mind...
    if reg == 0 or reg==intrinsic_reg:
        adjusted_reg=0
    elif intrinsic_reg > reg:
        print('1 / len(imgs) is greater than reg. Set reg to to larger values or else to zero')
        assert(0)
    else:
        adjusted_reg = reg - intrinsic_reg 
    print(f'Reg is {reg}. Intrinsic reg is {intrinsic_reg:.3f}. Adjusted reg is {adjusted_reg:.3f}')

    aligned_imgs = imgs
    piecewise_estimators=[]
    for iter in range(n_iter):
        print("Template alignment iteration {}/{}".format(iter+1,n_iter))        
        if iter==0 and template_method != 1: #Haxby method
            assert(reg==0) #Haxby method not tested/fixed for regularization
            aligned_imgs=[imgs[0]] 
            current_template = imgs[0]
            for i in range(1,len(imgs)):
                piecewise_estimator= MySurfacePairwiseAlignment(alignment_method, clustering, n_jobs=n_jobs,reg=reg)
                piecewise_estimator.fit(imgs[i], current_template)
                new_img = piecewise_estimator.transform(imgs[i])
                aligned_imgs.append(new_img)
                if template_method==2: #new template is average of all previous aligned images
                    current_template = _rescaled_euclidean_mean(aligned_imgs,scale_template)
                elif template_method==3: #new template is average of previous template and latest image
                    current_template = _rescaled_euclidean_mean([current_template, new_img],scale_template)  
        template = _rescaled_euclidean_mean(
        aligned_imgs, scale_template)

        aligned_imgs,piecewise_estimators = _align_images_to_template(imgs, template, alignment_method, clustering, n_bags, memory, memory_level,n_jobs, verbose,adjusted_reg)
    return template, piecewise_estimators

'''
def _map_template_to_image(imgs, train_index, template, alignment_method,
                           n_pieces, clustering, n_bags, masker,
                           memory, memory_level, n_jobs, verbose):
    """Learn alignment operator from the template toward new images.

    Parameters
    ----------
    imgs: list of 3D Niimgs
        Target images to learn mapping from the template to a new subject
    train_index: list of int
        Matching index between imgs and the corresponding template images to use
        to learn alignment. len(train_index) must be equal to len(imgs)
    template: list of 3D Niimgs
        Learnt in a first step now used as source image
    All other arguments are the same are passed to PairwiseAlignment


    Returns
    -------
    mapping: instance of PairwiseAlignment class
        Alignment estimator fitted to align the template with the input images
    """

    mapping_image = index_img(template, train_index)
    mapping = PairwiseAlignment(n_pieces=n_pieces,
                                alignment_method=alignment_method,
                                clustering=clustering,
                                n_bags=n_bags, mask=masker, memory=memory,
                                memory_level=memory_level,
                                n_jobs=n_jobs, verbose=verbose)
    mapping.fit(mapping_image, imgs)
    return mapping


def _predict_from_template_and_mapping(template, test_index, mapping):
    """ From a template, and an alignment estimator, predict new contrasts

    Parameters
    ----------
    template: list of 3D Niimgs
        Learnt in a first step now used to predict some new data
    test_index:
        Index of the images not used to learn the alignment mapping and so
        predictable without overfitting
    mapping: instance of PairwiseAlignment class
        Alignment estimator that must have been fitted already

    Returns
    -------
    transformed_image: list of Niimgs
        Prediction corresponding to each template image with index in test_index
        once realigned to the new subjects
    """
    image_to_transform = index_img(template, test_index)
    transformed_image = mapping.transform(image_to_transform)
    return transformed_image
'''

class MyTemplateAlignment(BaseEstimator, TransformerMixin):
    """
    Decompose the source images into regions and summarize subjects information \
    in a template, then use pairwise alignment to predict \
    new contrast for target subject.
    """

    def __init__(self, alignment_method="identity",
                 clustering='kmeans', scale_template=False,
                 n_iter=2, save_template=None, n_bags=1,
                 target_affine=None, target_shape=None,
                 memory=Memory(cachedir=None), memory_level=0,
                 n_jobs=1, verbose=0,template_method=1,reg=0):
        '''
        Parameterss
        ----------
        alignment_method: string
            Algorithm used to perform alignment between X_i and Y_i :
            * either 'identity', 'scaled_orthogonal', 'optimal_transport',
            'ridge_cv', 'permutation', 'diagonal'
            * or an instance of one of alignment classes (imported from
            functional_alignment.alignment_methods)
        n_pieces: int, optional (default = 1)
            Number of regions in which the data is parcellated for alignment.
            If 1 the alignment is done on full scale data.
            If > 1, the voxels are clustered and alignment is performed on each
            cluster applied to X and Y.
        clustering : string or 3D Niimg optional (default : kmeans)
            'kmeans', 'ward', 'rena', 'hierarchical_kmeans' method used for
            clustering of voxels based on functional signal,
            passed to nilearn.regions.parcellations
            If 3D Niimg, image used as predefined clustering,
            n_bags and n_pieces are then ignored.
        scale_template: boolean, default False
            rescale template after each inference so that it keeps
            the same norm as the average of training images.
        n_iter: int
           number of iteration in the alternate minimization. Each img is
           aligned n_iter times to the evolving template. If n_iter = 0,
           the template is simply the mean of the input images.
        save_template: None or string(optional)
            If not None, path to which the template will be saved.
        n_bags: int, optional (default = 1)
            If 1 : one estimator is fitted.
            If >1 number of bagged parcellations and estimators used.
        target_affine: 3x3 or 4x4 matrix, optional (default = None)
            This parameter is passed to nilearn.image.resample_img.
            Please see the related documentation for details.
        target_shape: 3-tuple of integers, optional (default = None)
            This parameter is passed to nilearn.image.resample_img.
            Please see the related documentation for details.
        memory: instance of joblib.Memory or string (default = None)
            Used to cache the masking process and results of algorithms.
            By default, no caching is done. If a string is given, it is the
            path to the caching directory.
        memory_level: integer, optional (default = None)
            Rough estimator of the amount of memory used by caching.
            Higher value means more memory for caching.
        n_jobs: integer, optional (default = 1)
            The number of CPUs to use to do the computation. -1 means
            'all CPUs', -2 'all CPUs but one', and so on.
        verbose: integer, optional (default = 0)
            Indicate the level of verbosity. By default, nothing is printed.
        '''
        self.template = None
        self.alignment_method = alignment_method
        self.clustering = clustering 
        self.n_iter = n_iter
        self.scale_template = scale_template
        self.save_template = save_template
        self.n_bags = n_bags
        self.target_affine = target_affine
        self.target_shape = target_shape
        self.memory = memory
        self.memory_level = memory_level
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.template_method=template_method
        self.reg=reg
    
    def fit_to_template(self,imgs):
        """
        Fit new imgs to pre-calculated template
        """
        _,self.estimators = _align_images_to_template(imgs, self.template, self.alignment_method, self.clustering, self.n_bags, self.memory, self.memory_level,self.n_jobs, self.verbose,self.reg)
    
    def fit(self, imgs):
        """
        Learn a template from source images, using alignment.

        Parameters
        ----------
        imgs: List of 4D Niimg-like or List of lists of 3D Niimg-like
            Source subjects data. Each element of the parent list is one subject
            data, and all must have the same length (n_samples).

        Returns
        -------
        self

        Attributes
        ----------
        self.template: 4D Niimg object
            Length : n_samples

        """
        # Assume imgs is a list (nsubjects) of arrays(nvols,ngrayordinates)
        assert(imgs[0].shape[1]==self.clustering.shape[0])
        self.template, self.estimators = \
            _create_template(imgs, self.n_iter, self.scale_template,
                             self.alignment_method,
                             self.clustering, self.n_bags,
                             self.memory, self.memory_level,
                             self.n_jobs, self.verbose,self.template_method,self.reg)
        if self.save_template is not None:
            self.template.to_filename(self.save_template)

    '''
    def transform(self, imgs, train_index, test_index):
        """ Learn alignment between new subject and template calculated during fit,
        then predicts other conditions for this new subject.
        Alignment is learnt between imgs and conditions in the template indexed by train_index.
        Prediction correspond to conditions in the template index by test_index.

        Parameters
        ----------
        imgs: List of 3D Niimg-like objects
            Target subjects known data. Every img must have length (number of sample) train_index.
        train_index: list of ints
            Indexes of the 3D samples used to map each img to the template.
            Every index should be smaller than the number of images in the template.
        test_index: list of ints
            Indexes of the 3D samples to predict from the template and the mapping.
            Every index should be smaller than the number of images in the template.


        Returns
        -------
        predicted_imgs: List of 3D Niimg-like objects
            Target subjects predicted data. Each Niimg has the same length as the list test_index

        """

        
        if isinstance(imgs[0], (list, np.ndarray)) and len(imgs[0]) != len(train_index):
            raise ValueError(' Each element of imgs (Niimg-like or list of Niimgs) \
                             should have the same length as the length of train_index.')
        elif load_img(imgs[0]).shape[-1] != len(train_index):
            raise ValueError(
                ' Each element of imgs (Niimg-like or list of Niimgs) \
                should have the same length as the length of train_index.')

        template_length = self.template.shape[-1]
        if not (all(i < template_length for i in test_index) and all(
                i < template_length for i in train_index)):
            raise ValueError(
                "Template has {} images but you provided a greater index in \
                train_index or test_index.".format(template_length))

        fitted_mappings = Parallel(self.n_jobs, prefer="threads", verbose=self.verbose)(
            delayed(_map_template_to_image)
            (img, train_index, self.template, self.alignment_method,
             self.n_pieces, self.clustering, self.n_bags, self.masker_,
             self.memory, self.memory_level, self.n_jobs, self.verbose
             ) for img in imgs
        )

        predicted_imgs = Parallel(self.n_jobs, prefer="threads", verbose=self.verbose)(
            delayed(_predict_from_template_and_mapping)
            (self.template, test_index, mapping) for mapping in fitted_mappings
        )
        return predicted_imgs
        
     '''
     
    def transform(self,X,index):
        #Transform X with subject 'index''s aligner to the template
        
        return self.estimators[index].transform(X)
        
        
class LowDimTemplateAlignment(MyTemplateAlignment):
    """
    Dimensionality reduction first (across vertices, separately for each subject) to reduce effective no. of vertices. No parcellation.
    """

    def __init__(self,alignment_method="identity",
                 clustering='kmeans', scale_template=False,
                 n_iter=2, save_template=None, n_bags=1,
                 target_affine=None, target_shape=None,
                 memory=Memory(cachedir=None), memory_level=0,
                 n_jobs=1, verbose=0,
                 n_components=20,whiten=True,lowdim_method='pca',reg=0):
                                 
        super().__init__(alignment_method=alignment_method,
                 clustering=clustering, scale_template=scale_template,
                 n_iter=n_iter, save_template=save_template, n_bags=n_bags,
                 target_affine=target_affine, target_shape=target_shape,
                 memory=memory, memory_level=memory_level,
                 n_jobs=n_jobs, verbose=verbose,reg=reg)

        self.n_components=n_components
        self.whiten=whiten
        self.clustering=np.ones(n_components)
        self.lowdim_method=lowdim_method

    def fit(self, imgs):
        # Assume imgs is a list (nsubjects) of arrays(nvols,ngrayordinates)
        #assert(imgs[0].shape[1]==self.clustering.shape[0])      
            
        from sklearn.decomposition import PCA, FastICA
        if self.lowdim_method=='pca':
            self.pcas=[PCA(n_components=self.n_components,whiten=self.whiten) for img in imgs]  
        elif self.lowdim_method=='ica':
            self.pcas=[FastICA(n_components=self.n_components,max_iter=100000) for img in imgs]    
        lowdim_imgs=[pca.fit_transform(img) for pca,img in zip(self.pcas,imgs)]    
        

        
        self.template, self.estimators = \
            _create_template(lowdim_imgs, self.n_iter, self.scale_template,
                             self.alignment_method,
                             self.clustering, self.n_bags,
                             self.memory, self.memory_level,
                             self.n_jobs, self.verbose,self.reg)
        if self.save_template is not None:
            self.template.to_filename(self.save_template)
            
    def transform(self,X,index):
        #Transform X with subject 'index''s aligner to the template
        
        lowdim_X=self.pcas[index].transform(X)
        return self.estimators[index].transform(lowdim_X)