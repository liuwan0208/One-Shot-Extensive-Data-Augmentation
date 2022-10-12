from __future__ import division
import os
import nibabel as nib
import nibabel.processing
import numpy as np
from shutil import copyfile
import shutil
from os.path import join
from random import sample
from dipy.core.gradients import gradient_table
import copy


# bundles = ['AF_left', 'AF_right', 'ATR_left', 'ATR_right', 'CA', 'CC_1', 'CC_2', 'CC_3', 'CC_4', 'CC_5', 'CC_6',
#                    'CC_7', 'CG_left', 'CG_right', 'CST_left', 'CST_right', 'MLF_left', 'MLF_right', 'FPT_left',
#                    'FPT_right', 'FX_left', 'FX_right', 'ICP_left', 'ICP_right', 'IFO_left', 'IFO_right', 'ILF_left',
#                    'ILF_right', 'MCP', 'OR_left', 'OR_right', 'POPT_left', 'POPT_right', 'SCP_left', 'SCP_right',
#                    'SLF_I_left', 'SLF_I_right', 'SLF_II_left', 'SLF_II_right', 'SLF_III_left', 'SLF_III_right',
#                    'STR_left', 'STR_right', 'UF_left', 'UF_right', 'CC', 'T_PREF_left', 'T_PREF_right', 'T_PREM_left',
#                    'T_PREM_right', 'T_PREC_left', 'T_PREC_right', 'T_POSTC_left', 'T_POSTC_right', 'T_PAR_left',
#                    'T_PAR_right', 'T_OCC_left', 'T_OCC_right', 'ST_FO_left', 'ST_FO_right', 'ST_PREF_left',
#                    'ST_PREF_right', 'ST_PREM_left', 'ST_PREM_right', 'ST_PREC_left', 'ST_PREC_right', 'ST_POSTC_left',
#                    'ST_POSTC_right', 'ST_PAR_left', 'ST_PAR_right', 'ST_OCC_left', 'ST_OCC_right']
#
# subjects = ["992774", "991267", "987983", "984472", "983773", "979984", "978578", "965771", "965367", "959574",
#         "958976", "957974", "951457", "932554", "930449", "922854", "917255", "912447", "910241",
#         "904044", "901442", "901139", "901038", "899885", "898176", "896879", "896778", "894673", "889579",
#         "877269", "877168", "872764", "872158", "871964", "871762", "865363", "861456", "859671",
#         "857263", "856766", "849971", "845458", "837964", "837560", "833249", "833148", "826454", "826353",
#         "816653", "814649", "802844", "792766", "792564", "789373", "786569", "784565", "782561", "779370",
#         "771354", "770352", "765056", "761957", "759869", "756055", "753251", "751348", "749361", "748662",
#         "748258", "742549", "734045", "732243", "729557", "729254", "715647", "715041", "709551", "705341",
#         "704238", "702133", "695768", "690152", "687163", "685058", "683256", "680957", "679568", "677968",
#         "673455", "672756", "665254", "654754", "645551", "644044", "638049", "627549", "623844", "622236",
#         "620434", "613538", "601127", "599671", "599469"]  #HCP103
#
# path = '/data1/qilu/HCP105/'
#
# subjects_test_18 =  ["695768", "690152", "687163", "685058", "683256", "680957", "679568", "677968", "673455", "672756",
# 					 "665254", "654754", "645551",  "638049", "627549", "623844", "622236", "613538"]
# # grad_num=191: "644044", "620434" ("601127")
#
# subjects_val_17 = ["765056", "761957", "759869", "756055",  "751348", "748662", "748258", "742549", "734045", "732243",
# 				   "729557", "729254", "715647",  "709551", "705341", "704238", "702133"]
# # grad_num=191:  "753251", "749361", "715041"

# subjects_training_54 = ["992774", "991267", "987983", "984472", "983773",  "978578", "965771", "965367", "959574",
#                     "958976", "957974",  "932554", "930449", "922854", "917255", "912447", "910241",
#                     "904044", "901442", "901139",  "899885", "898176", "896879", "896778", "894673", "889579",
#                      "877269", "877168",  "872158", "871964", "871762", "865363", "861456", "859671",
#                     "857263", "856766", "849971", "845458", "837964", "837560", "833249", "833148", "826454", "826353",
#                     "816653",  "802844",  "792564", "789373", "786569", "784565", "782561", "779370",
#                     "771354", "770352"]


### dwi copy


def generate_bbox_cutout(img_size):
	## select weight: lamda--(0,1) uniform distribution
	alpha = 1
	lamda = np.random.beta(alpha, alpha)
	## generate crop 3D patches coordinates: center points--(0, img_szie) uniform dis
	W = img_size[0]
	H = img_size[1]
	D = img_size[2]
	cut_rate = np.sqrt(1 - lamda)
	cut_x = np.int(W * cut_rate)  # box size in each dimension
	cut_y = np.int(H * cut_rate)
	cut_z = np.int(D * cut_rate)
	cx = np.random.randint(W)  # coordinate of box center
	cy = np.random.randint(H)
	cz = np.random.randint(D)
	bbx1 = np.clip(cx - cut_x // 2, 0, W)
	bbx2 = np.clip(cx + cut_x // 2, 0, W)
	bby1 = np.clip(cy - cut_y // 2, 0, H)
	bby2 = np.clip(cy + cut_y // 2, 0, H)
	bbz1 = np.clip(cz - cut_z // 2, 0, D)
	bbz2 = np.clip(cz + cut_z // 2, 0, D)
	return bbx1,bbx2,bby1,bby2,bbz1,bbz2







init_path='/data4/wanliu/HCP_2.5mm_341k/HCP_for_training_COPY'
subjects = ['620434']
labeled_tracts = 4 # 4,6,12
bundle_mask = 'bundle_masks_'+str(labeled_tracts)+'.nii.gz'
log_save_file= '/data4/wanliu/HCP_2.5mm_341k/HCP_daug/Oneshot_620434_4/CutoutYChange_'+str(labeled_tracts)+'tracts.txt'
target_path  = '/data4/wanliu/HCP_2.5mm_341k/HCP_daug/Oneshot_620434_4/CutoutYChange/HCP_for_training_COPY'


used_data_num = len(subjects)
aug_data_num = 15
start_index = 0


sub = subjects[0]
for i in range(aug_data_num):
	# load data and label
	data_file = join(init_path, sub, 'mrtrix_peaks.nii.gz')
	data_nii = nib.load(data_file)
	data_affine = data_nii.affine
	data = data_nii.get_fdata()

	label_file = join(init_path, sub, bundle_mask)
	label_nii = nib.load(label_file)
	label_affine = label_nii.affine
	label = label_nii.get_fdata()

	# make_dir
	save_dir = join(target_path, 'daug_' + str(i + start_index))
	if os.path.exists(save_dir) == 0:
		os.makedirs(save_dir)
	## generate mask
	bbx1,bbx2,bby1,bby2,bbz1,bbz2 = generate_bbox_cutout(data.shape)

	log = "Daug_image:{}, sub:{}, cutout region:{}\n".format(i + start_index, sub, [bbx1,bbx2,bby1,bby2,bbz1,bbz2])
	print(log)
	with open(log_save_file, 'a') as f:
		f.write(log)

	data[bbx1:bbx2, bby1:bby2, bbz1:bbz2, :] = 0
	data = data.astype(np.float32)
	data_nii = nib.Nifti1Image(data, data_affine)
	nib.save(data_nii, join(save_dir, 'mrtrix_peaks.nii.gz'))


	label[bbx1:bbx2, bby1:bby2, bbz1:bbz2, :] = 0
	label = label.astype(np.float32)
	label_nii = nib.Nifti1Image(label, label_affine)
	nib.save(label_nii, join(save_dir, bundle_mask))
