: '
x
Mostly adapted from github/sina-mansour/neural-identity/mrtrix_tractography
For use on AWS EC2
'

#parameters
nlines=5M #default 5M
all_reduce_lines="50k 200k 1M" #default 50k 200k 1M

#colours for fancy logging
RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'

#Beginning stuff
echo "Starting" 
start=`date +%s`

run() {

	sub=$1
	
	scratch_path="/home/ec2-user/tmp/scratch"	
	store_path="/home/ec2-user/hcp/HCP_1200/${sub}"
	tmp_path="/home/ec2-user/studies/files0/intermediates/diff2/${sub}"
	int="/home/ec2-user/tmp/${sub}"

	if [ -d "$tmp_path" ] && [ "$(ls $tmp_path | wc -l)" -ge 8 ]
	then 
		echo $sub already exists and has "$(ls $tmp_path | wc -l)" files
	else
		echo $sub doesnt exist. Doing it now

		mkdir -p scratch_path
		mkdir -p $int/T1w/Diffusion/
		

		echo -e "${GREEN}[INFO]${NC}${sub} `date`: Copying Diff and T1"
		rsync -r $store_path/T1w/Diffusion/ $int/T1w/Diffusion
		cp $store_path/T1w/T1w_acpc_dc_restore_brain.nii.gz $int/T1w
		

		
		# First convert the initial diffusion image to .mif (3mins, 4GiB)
		echo -e "${GREEN}[INFO]${NC}${sub} `date`: Converting images to .mif"
		mrconvert "${int}/T1w/Diffusion/data.nii.gz" "${int}/dwi.mif" -fslgrad "${int}/T1w/Diffusion/bvecs" "${int}/T1w/Diffusion/bvals" -datatype float32 -strides 0,0,0,1 -quiet
		
		#Bias corrections
		echo -e "${GREEN}[INFO]${NC}${sub} `date`: Bias correction"
		dwibiascorrect fsl "${int}/dwi.mif" "${int}/dwi_bias.mif" -bias "${int}/bias_ants_field.mif" -scratch $scratch_path -quiet

		mrconvert "${int}/T1w/T1w_acpc_dc_restore_brain.nii.gz" "${int}/T1.mif" -quiet


		

		# Five tissue type segmentation using premasked 0.7mm T1w in Structural folder. 16mins
		echo -e "${GREEN}[INFO]${NC}${sub} `date`: Running five tissue type segmentation"
		5ttgen fsl "${int}/T1.mif" "${int}/5tt.mif" -premasked -scratch ${scratch_path} -quiet

		
		# Estimate the response function using the dhollander method. 15 mins
		echo -e "${GREEN}[INFO]${NC}${sub} `date`: Estimation of response function using dhollander"
		dwi2response dhollander "${int}/dwi_bias.mif" "${int}/wm.txt" "${int}/gm.txt" "${int}/csf.txt" -voxels "${int}/voxels.mif" -scratch ${scratch_path} -quiet

		# Multi-Shell, Multi-Tissue Constrained Spherical Deconvolution. 7 mins
		echo -e "${GREEN}[INFO]${NC}${sub} `date`: Running Multi-Shell, Multi-Tissue Constrained Spherical Deconvolution"
		dwi2fod msmt_csd "${int}/dwi_bias.mif" -mask "${int}/T1w/Diffusion/nodif_brain_mask.nii.gz"  "${int}/wm.txt" "${int}/wmfod.mif" "${int}/gm.txt" "${int}/gmfod.mif" "${int}/csf.txt" "${int}/csffod.mif" -quiet
		
		#Intensity normalisation
		echo -e "${GREEN}[INFO]${NC}${sub} `date`: Intensity normalisation"
		mtnormalise "${int}/wmfod.mif" "${int}/wmfodn.mif" "${int}/gmfod.mif" "${int}/gmfodn.mif" "${int}/csffod.mif" "${int}/csffodn.mif" -mask "${int}/T1w/Diffusion/nodif_brain_mask.nii.gz" -quiet
		
		
		# Create a mask for surface seeds. Fast. Not using intensity normalised data...
		echo -e "${GREEN}[INFO]${NC}${sub} `date`: Creating cortical surface mask"
		python3 -c "import surface_mask as sm; sm.binary_volume_mask_from_surface_vertices('${store_path}/T1w/fsaverage_LR32k/${sub}.L.white.32k_fs_LR.surf.gii','${store_path}/T1w/fsaverage_LR32k/${sub}.R.white.32k_fs_LR.surf.gii', '${store_path}/T1w/T1w_acpc_dc_restore.nii.gz','${int}/Cortical_surface_mask.nii.gz',cifti_file='/home/ec2-user/studies/code/ones.dscalar.nii',thickness=5, fwhm=2)"	

		# create white matter + subcortical binary mask to trim streamlines with
		# first extract the white matter and subcortical tissues from 5tt image. Fast
		echo -e "${GREEN}[INFO]${NC}${sub} `date`: Creating white matter plus subcortical mask"
		mrconvert --coord 3 2 -axes 0,1,2 "${int}/5tt.mif" "${int}/5tt-white_matter.mif" -quiet
		mrconvert --coord 3 1 -axes 0,1,2 "${int}/5tt.mif" "${int}/5tt-subcortical.mif" -quiet
		# add both tissues together
		mrmath "${int}/5tt-white_matter.mif" \
			"${int}/5tt-subcortical.mif" \
			sum \
			"${int}/5tt-wm+sc.mif" -quiet
		# binarise to create the trim mask
		mrcalc "${int}/5tt-wm+sc.mif" 0 -gt 1 0 -if "${int}/5tt-trim.mif" -quiet
		
		
		
		echo -e "${GREEN}[INFO]${NC}${sub} `date`: tckgen"
		tckgen -seed_image "${int}/Cortical_surface_mask.nii.gz" \
			-mask "${int}/5tt-trim.mif" \
			-select "${nlines}" \
			-maxlength 300 \
			-cutoff 0.06 \
			"${int}/wmfodn.mif" \
			"${int}/tracks_${nlines}.tck" -quiet

		new_tck_prefix=${int}/tracks_${nlines}

		echo -e "${GREEN}[INFO]${NC}${sub} `date`: tcksift2"
		tcksift2 ${new_tck_prefix}.tck ${int}/wmfodn.mif ${new_tck_prefix}_sift2act_weights.txt -act ${int}/5tt.mif -quiet
		tckresample ${new_tck_prefix}.tck ${new_tck_prefix}_end.tck -quiet -endpoints

		mkdir -p $tmp_path

		cp ${new_tck_prefix}_end.tck $tmp_path
		cp ${new_tck_prefix}_sift2act_weights.txt $tmp_path

		for reduce_lines in ${all_reduce_lines}
		do
			new_tck_prefix=${int}/tracks_${nlines}_${reduce_lines}
			tckedit ${int}/tracks_${nlines}.tck -number ${reduce_lines} ${new_tck_prefix}.tck -quiet

			echo -e "${GREEN}[INFO]${NC}${sub} `date`: tcksift2"
			tcksift2 ${new_tck_prefix}.tck ${int}/wmfodn.mif ${new_tck_prefix}_sift2act_weights.txt -act ${int}/5tt.mif -quiet
			tckresample ${new_tck_prefix}.tck ${new_tck_prefix}_end.tck -quiet -endpoints
			cp ${new_tck_prefix}_end.tck $tmp_path
			cp ${new_tck_prefix}_sift2act_weights.txt $tmp_path
		done	

		echo -e "${GREEN}[INFO]${NC} `date`: ${sub} finished!"
		rm -r ~/tmp

	fi
	
}

export -f run

for sub in 102816
do
	run $sub
done


#Ending stuff
end=`date +%s`
echo '\007'
runtime_min=$(($((end-start))/60))
echo Finished in $runtime_min mins