"""
#03: 8 core 16GB
#04: 32 core 64GB
#05: 64 core 1000 GB
#06: 64 core 128 GB
"""

#Conda stuff

#do these steps manually
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
sh "Miniconda3-latest-Linux-x86_64.sh"
#now restart shell

#do these together
envdir=studies/code/config/py362
conda create --name py362 --file $envdir/spec-file.txt
conda env update --name py362 --file $envdir/environment.yml

#within py362
conda install -c "conda-forge/label/cf201901" psutil=5.6.2
conda install -c mrtrix3 mrtrix3 

#FSL: Get and install FSL (python fslinstaller.py) (from base environment) (for MRTRix)

#s3fs FUSE. Do these together
export LANG=en_US.UTF-8
export LANGUAGE=en_US.UTF-8
export LC_COLLATE=C
export LC_CTYPE=en_US.UTF-8
sudo yum -y update all
sudo yum -y install automake fuse fuse-devel gcc-c++ git libcurl-devel libxml2-devel make openssl-devel
git clone https://github.com/s3fs-fuse/s3fs-fuse.git


cd s3fs-fuse
./autogen.sh
./configure --prefix=/usr --with-openssl

#could try make -C s3fs-fuse
make

#could try sudoe make -C s3fs-fuse install
sudo make install

sudo bash -c "echo 'AKIAXO65CT57OAB3X4GU:aoYyAII7RXqK+bahHcdtelCZo7IkVS0l304j28JS' > /etc/passwd-s3fs"

cd ~
sudo chmod 640 /etc/passwd-s3fs
mkdir hcp

sudo bash -c "echo 'sudo s3fs hcp-openaccess -o use_cache=/tmp -o allow_other -o uid=1000 -o mp_umask=002 -o multireq_max=5 ~/hcp' >> /etc/bashrc"

