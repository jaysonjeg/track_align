This code corresponds to the paper "Spurious correlations in surface fMRI"

Script biasfmri_effects.py for main analyses\
Script biasfmri_cause.py for analyses on fsaverage5 mesh\
Script biasfmrivol.py for analyses in volume space
The other files contain utility functions that are called by these main scripts.

I used a Python environment in conda. The environment was created with the following commands.

conda create -n py390 python=3.9\
python -m pip install -U nilearn\
conda install conda-forge::matplotlib\
conda install -c anaconda seaborn\
conda install -c conda-forge pingouin\
pip install hcp-utils\
pip install neuromaps\
[my connectome-spatial-smoothing fork](https://github.com/jaysonjeg/connectome-spatial-smoothing) with pip install -e .

