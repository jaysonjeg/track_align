import numpy as np

def make_random_fmri_data(nsubjects=50,ntimepoints=1000,shape=(91,109,91)):
    """
    Generate random fMRI data for nsubjects, each with shape given by shape+[ntimepoints]. Use random numbers drawn from a normal distribution.
    Parameters
    ----------
    nsubjects : int
    ntimepoints : int
    shape : tuple
        e.g. (91,100,91)
    """
    return [np.random.randn(*shape,ntimepoints) for i in range(nsubjects)]


x = make_random_fmri_data()