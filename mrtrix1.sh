: '
Mostly adapted from github/sina-mansour/neural-identity/mrtrix_tractography
'

#parameters
nlines=1M #default 5M
all_reduce_lines=50k 200k 1M #default 50k 200k 1M

#colours for fancy logging
RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'

#Beginning stuff
echo "Starting" 
start=`date +%s`

for sub in 130518
do
	echo $sub
	: '
	scratch_path="/home/ec2-user/studies/files0/intermediates/diff2_scratch"	
	store_path="/home/ec2-user/hcp/HCP_1200/${sub}"
	tmp_path="/home/ec2-user/studies/files0/intermediates/diff2/${sub}"
	'
	
	scratch_path="/mnt/d/FORSTORAGE/Data/Project_Hyperalignment/HCP_tractography/scratch_dir"
	store_path="/mnt/d/FORSTORAGE/Data/HCP_S1200/${sub}"
	tmp_path="/mnt/d/FORSTORAGE/Data/Project_Hyperalignment/HCP_tractography/${sub}"
	
	mkdir -p ${tmp_path}
	
	# First convert the initial diffusion image to .mif (3mins, 4GiB)
	echo -e "${GREEN}[INFO]${NC} `date`: Converting images to .mif"
	mrconvert "${store_path}/T1w/Diffusion/data.nii.gz" "${tmp_path}/dwi.mif" -fslgrad "${store_path}/T1w/Diffusion/bvecs" "${store_path}/T1w/Diffusion/bvals" -datatype float32 -strides 0,0,0,1

	#Bias corrections
	echo -e "${GREEN}[INFO]${NC} `date`: Bias correction"
	dwibiascorrect ants "${tmp_path}/dwi.mif" "${tmp_path}/dwi_bias.mif" -bias "${tmp_path}/bias_ants_field.mif" -info -scratch $scratch_path
	rm -f "${tmp_path}/dwi.mif"

	# Five tissue type segmentation using premasked 0.7mm T1w in Structural folder. 16mins
	echo -e "${GREEN}[INFO]${NC} `date`: Running five tissue type segmentation"
	mrconvert "${store_path}/T1w/T1w_acpc_dc_restore_brain.nii.gz" "${tmp_path}/T1.mif"
	5ttgen fsl "${tmp_path}/T1.mif" "${tmp_path}/5tt.mif" -premasked -scratch ${scratch_path}


	# Estimate the response function using the dhollander method. 15 mins
	echo -e "${GREEN}[INFO]${NC} `date`: Estimation of response function using dhollander"
	dwi2response dhollander "${tmp_path}/dwi_bias.mif" \
							"${tmp_path}/wm.txt" \
							"${tmp_path}/gm.txt" \
							"${tmp_path}/csf.txt" \
							-voxels "${tmp_path}/voxels.mif"\
							-scratch ${scratch_path}


	# Multi-Shell, Multi-Tissue Constrained Spherical Deconvolution. 7 mins
	echo -e "${GREEN}[INFO]${NC} `date`: Running Multi-Shell, Multi-Tissue Constrained Spherical Deconvolution"
	dwi2fod msmt_csd "${tmp_path}/dwi_bias.mif" \
					 -mask "${store_path}/T1w/Diffusion/nodif_brain_mask.nii.gz" \
					 "${tmp_path}/wm.txt" "${tmp_path}/wmfod.mif" \
					 "${tmp_path}/gm.txt" "${tmp_path}/gmfod.mif" \
					 "${tmp_path}/csf.txt" "${tmp_path}/csffod.mif" \
	
	#Intensity normalisation
	echo -e "${GREEN}[INFO]${NC} `date`: Intensity normalisation"
	mtnormalise "${tmp_path}/wmfod.mif" "${tmp_path}/wmfodn.mif" "${tmp_path}/gmfod.mif" "${tmp_path}/gmfodn.mif" "${tmp_path}/csffod.mif" "${tmp_path}/csffodn.mif" -mask "${store_path}/T1w/Diffusion/nodif_brain_mask.nii.gz" -info	
	
	# Create a mask for surface seeds. Fast. Not using intensity normalised data...
	echo -e "${GREEN}[INFO]${NC} `date`: Creating cortical surface mask"
	python3 -c "import surface_mask as sm; sm.binary_volume_mask_from_surface_vertices('${store_path}/T1w/fsaverage_LR32k/${sub}.L.white.32k_fs_LR.surf.gii','${store_path}/T1w/fsaverage_LR32k/${sub}.R.white.32k_fs_LR.surf.gii', '${store_path}/T1w/T1w_acpc_dc_restore.nii.gz','${tmp_path}/Cortical_surface_mask.nii.gz',cifti_file='/mnt/g/My Drive/PhD/Project_Hyperalignment/Code/neural-identity-master/data/templates/cifti/ones.dscalar.nii',thickness=5, fwhm=2)"	

	# create white matter + subcortical binary mask to trim streamlines with
	# first extract the white matter and subcortical tissues from 5tt image. Fast
	echo -e "${GREEN}[INFO]${NC} `date`: Creating white matter plus subcortical mask"
	mrconvert --coord 3 2 -axes 0,1,2 "${tmp_path}/5tt.mif" "${tmp_path}/5tt-white_matter.mif"
	mrconvert --coord 3 1 -axes 0,1,2 "${tmp_path}/5tt.mif" "${tmp_path}/5tt-subcortical.mif"
	# add both tissues together
	mrmath "${tmp_path}/5tt-white_matter.mif" \
		   "${tmp_path}/5tt-subcortical.mif" \
		   sum \
		   "${tmp_path}/5tt-wm+sc.mif"
	# binarise to create the trim mask
	mrcalc "${tmp_path}/5tt-wm+sc.mif" 0 -gt 1 0 -if "${tmp_path}/5tt-trim.mif"


	tckgen -seed_image "${tmp_path}/Cortical_surface_mask.nii.gz" \
		   -mask "${tmp_path}/5tt-trim.mif" \
		   -select "${nlines}" \
		   -maxlength 300 \
		   -cutoff 0.06 \
		   "${tmp_path}/wmfodn.mif" \
		   "${tmp_path}/tracks_${nlines}.tck"

	tcksift2 "${tmp_path}/tracks_${nlines}.tck"  "${tmp_path}/wmfodn.mif" "${tmp_path}/tracks_${nlines}_sift2act_weights.txt" -act "${tmp_path}/5tt.mif"
	
	
	
	for reduce_lines in ${all_reduce_lines}
	do
		tckedit "${tmp_path}/tracks_${nlines}.tck" -number ${reduce_lines} "${tmp_path}/tracks_${nlines}_${reduce_lines}.tck"
		tcksift2 "${tmp_path}/tracks_${nlines}_${reduce_lines}.tck"  "${tmp_path}/wmfodn.mif" "${tmp_path}/tracks_${nlines}_${reduce_lines}_sift2act_weights.txt" -act "${tmp_path}/5tt.mif"
	done	

	
	echo -e "${GREEN}[INFO]${NC} `date`: Removing unnecessary files."
	#rm "${tmp_path}/dwi_bias.mif"
	#rm "${tmp_path}/wmfod.mif"
	#rm "${tmp_path}/gmfod.mif"
	#rm "${tmp_path}/csffod.mif"
	
	echo -e "${GREEN}[INFO]${NC} `date`: Script finished!"
	
	

done

#Ending stuff
end=`date +%s`
echo '\007'
runtime=$(($((end-start))/60))
echo Finished in $runtime mins